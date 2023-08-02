"""
@author: natnij
"""
import pickle
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from multiprocessing import cpu_count
from datetime import datetime
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    ConstraintList,
    Binary,
    maximize,
    SolverFactory,
)
from pyomo.gdp import Disjunct, Disjunction
import pandas as pd
import numpy as np
from simpleor.config.config_assignment import (
    Specs,
    RESOURCES_OUTPUT_DIR as pathOut,
    OutputFiles as of,
)
from simpleor.supports.data_loader_assignment import DataProblem
from simpleor.supports.column_names import ColumnNames_assignment as c
from simpleor.utils.log_utils import LogUtilities as utLog
from simpleor.utils.common_utils import CommonUtils as utCom


class ShiftModel:
    def __init__(
        self,
        modelData,
        solverChoice="gdpopt",
        mip="glpk",
        nlp="ipopt",
        solverpath_exe=None,
    ):
        self.solverChoice = solverChoice
        self.mip = mip  # mixed integer programming solver
        self.nlp = nlp  # nonlinear programming solver
        self.solverpath_exe = solverpath_exe
        self.model = self.buildConcreteModel(modelData)

    def buildConcreteModel(self, data):
        model = ConcreteModel()
        model.Idx_employee = range(len(data.Employees))
        model.Idx_shift = range(data.shift.shape[1])
        model.Idx_noGoShiftGroups = range(data.noGoShiftGroups.shape[0])
        model.Idx_noGoNightShifts = range(data.noGoNightShifts.shape[0])
        model.Idx_hour = range(data.shift.shape[0])
        model.Idx_demand = range(len(data.Demand.demand))
        model.x = Var(model.Idx_employee, model.Idx_shift, within=Binary, initialize=1)

        # the minimum continuous shifts per week turns the problem into non-convex
        # and would require getting many logical-OR's via disjunction or
        # a big-M relaxation, which are numerically terrible.
        if np.sum(data.continuousShifts) > 0:
            model.Idx_contShifts = range(data.continuousShifts.shape[0])
            model.contShifts = Var(
                model.Idx_employee, model.Idx_contShifts, initialize=0
            )

            def calContShifts_rule(model, Idx_employee, Idx_contShifts):
                return (
                    sum(
                        data.continuousShifts[Idx_contShifts, i]
                        * model.x[Idx_employee, i]
                        for i in model.Idx_shift
                    )
                    == model.contShifts[Idx_employee, Idx_contShifts]
                )

            model.contShifts_c = Constraint(
                model.Idx_employee, model.Idx_contShifts, rule=calContShifts_rule
            )
            model.disjuncts = list()
            model.disjuncts_c = list()
            for Idx_employee in model.Idx_employee:
                model.disjuncts.append(list())
                for Idx_contShifts in model.Idx_contShifts:
                    model.disjuncts[Idx_employee].append(Disjunct())
                    model.disjuncts[Idx_employee][Idx_contShifts].c = Constraint(
                        expr=model.contShifts[Idx_employee, Idx_contShifts]
                        >= int(data.regDict["minShiftContinuous"])
                    )
                # the disjunct of not-any-shift-scheduled:
                model.disjuncts[Idx_employee].append(Disjunct())
                model.disjuncts[Idx_employee][-1].c = Constraint(
                    expr=sum(model.x[Idx_employee, i] for i in model.Idx_shift) == 0
                )
                model.disjuncts_c.append(
                    Disjunction(expr=model.disjuncts[Idx_employee])
                )

        # switch the disjunct blocks to be the first constraints
        # and the solution is way better.
        if np.sum(data.continuousNightShifts) > 0:
            model.Idx_contNightShifts = range(data.continuousNightShifts.shape[0])
            model.contNightShifts = Var(
                model.Idx_employee, model.Idx_contNightShifts, initialize=0
            )

            def calContNightShifts_rule(model, Idx_employee, Idx_contNightShifts):
                return (
                    sum(
                        data.continuousNightShifts[Idx_contNightShifts, i]
                        * model.x[Idx_employee, i]
                        for i in model.Idx_shift
                    )
                    == model.contNightShifts[Idx_employee, Idx_contNightShifts]
                )

            model.contNightShifts_c = Constraint(
                model.Idx_employee,
                model.Idx_contNightShifts,
                rule=calContNightShifts_rule,
            )
            model.nightDisjuncts = list()
            model.nightDisjuncts_c = list()
            for Idx_employee in model.Idx_employee:
                model.nightDisjuncts.append(list())
                # if there would be night shifts scheduled, they have to be
                # continuously the same shifts at least for the defined minimum days
                for Idx_contNightShifts in model.Idx_contNightShifts:
                    model.nightDisjuncts[Idx_employee].append(Disjunct())
                    model.nightDisjuncts[Idx_employee][
                        Idx_contNightShifts
                    ].c = Constraint(
                        expr=model.contNightShifts[Idx_employee, Idx_contNightShifts]
                        >= int(data.regDict["minNightShiftContinuous"])
                    )
                # the disjunct of no-night-shift-scheduled:
                model.nightDisjuncts[Idx_employee].append(Disjunct())
                model.nightDisjuncts[Idx_employee][-1].c = Constraint(
                    expr=sum(
                        model.contNightShifts[Idx_employee, i]
                        for i in model.Idx_contNightShifts
                    )
                    == 0
                )
                model.nightDisjuncts_c.append(
                    Disjunction(expr=model.nightDisjuncts[Idx_employee])
                )

        if np.sum(data.noGoShiftGroups.shape) > 0:

            def noGoShiftGroups_rule(model, Idx_noGoShiftGroups, Idx_employee):
                return (
                    sum(
                        data.noGoShiftGroups[Idx_noGoShiftGroups, i]
                        * model.x[Idx_employee, i]
                        for i in model.Idx_shift
                    )
                    <= 1
                )

            model.noGoShiftGroups_c = Constraint(
                model.Idx_noGoShiftGroups, model.Idx_employee, rule=noGoShiftGroups_rule
            )

        if np.sum(data.noGoNightShifts) > 0:

            def noGoNightShifts_rule(model, Idx_noGoNightShifts, Idx_employee):
                return sum(
                    data.noGoNightShifts[Idx_noGoNightShifts, i]
                    * model.x[Idx_employee, i]
                    for i in model.Idx_shift
                ) <= int(data.regDict["maxNightShiftContinuous"])

            model.noGoNightShifts_c = Constraint(
                model.Idx_noGoNightShifts, model.Idx_employee, rule=noGoNightShifts_rule
            )

        def minMaxShifts_rule(model, Idx_employee):
            minshift = max(
                [
                    data.regDict["minShiftsPerWeek"]
                    - int(sum(data.nonAvail[Idx_employee]) / 24),
                    0,
                ]
            )
            return (
                minshift,
                sum(model.x[Idx_employee, i] for i in model.Idx_shift),
                data.regDict["maxShiftsPerWeek"],
            )

        model.maxShifts_c = Constraint(model.Idx_employee, rule=minMaxShifts_rule)

        model.onDuty = Var(
            model.Idx_employee, model.Idx_hour, within=Binary, initialize=1
        )

        def calOnDuty_rule(model, Idx_employee, Idx_hour):
            shift = np.array(data.shift.iloc[Idx_hour, :], dtype=int)
            return (
                sum(model.x[Idx_employee, i] * shift[i] for i in model.Idx_shift)
                == model.onDuty[Idx_employee, Idx_hour]
            )

        model.onDuty_c = Constraint(
            model.Idx_employee, model.Idx_hour, rule=calOnDuty_rule
        )

        def onDuty_rule(model, Idx_employee):
            minhour = max(
                [data.regDict["minWorkHour"] - sum(data.nonAvail[Idx_employee]), 0]
            )
            return (
                minhour,
                sum(model.onDuty[Idx_employee, i] for i in model.Idx_hour),
                data.regDict["maxWorkHour"],
            )

        model.onDuty_bnd = Constraint(model.Idx_employee, rule=onDuty_rule)

        if np.sum(data.nonAvail) > 0:
            model.nonAvail_c = ConstraintList()
            for Idx_employee in model.Idx_employee:
                if sum(data.nonAvail[Idx_employee]) > 0:
                    lhs = sum(
                        model.onDuty[Idx_employee, i] * data.nonAvail[Idx_employee, i]
                        for i in model.Idx_hour
                    )
                    model.nonAvail_c.add(lhs == 0)

        model.Idx_noGoPref = np.array(
            [
                idx
                for idx, i in enumerate(
                    np.sum(data.noGoPreference, axis=1)
                    .astype(bool)
                    .astype(int)
                    .tolist()
                )
                if i > 0
            ]
        )

        def noGoPreference_rule(model, Idx_noGoPref):
            return (
                sum(
                    data.noGoPreference[Idx_noGoPref, i] * model.x[Idx_noGoPref, i]
                    for i in model.Idx_shift
                )
                == 0
            )

        if sum(model.Idx_noGoPref) > 0:
            model.noGoPref_c = Constraint(model.Idx_noGoPref, rule=noGoPreference_rule)

        def demandHour_rule(model, Idx_hour):
            return (
                data.Demand.minDemand[Idx_hour],
                sum(
                    model.onDuty[i, Idx_hour] * data.hourlyOutput[i]
                    for i in model.Idx_employee
                ),
                data.Demand.maxDemand[Idx_hour],
            )

        model.demandHour_c = Constraint(model.Idx_hour, rule=demandHour_rule)
        model.demandTotal_c = Constraint(
            expr=sum(
                model.onDuty[i, j] * data.hourlyOutput[i]
                for i in model.Idx_employee
                for j in model.Idx_hour
            )
            >= sum(data.Demand.demand)
        )

        model.costpp = Var(
            model.Idx_employee, bounds=(0, max(data.cost) * sum(data.Demand.maxDemand))
        )

        def costpp_rule(model, Idx_employee):
            # data.cost: hourly cost depending on standard/overtime definition
            # data.hourlyWage: employee-specific hourly cost
            return (
                sum(
                    model.onDuty[Idx_employee, i]
                    * data.cost[i]
                    * data.hourlyWage[Idx_employee]
                    for i in model.Idx_hour
                )
                == model.costpp[Idx_employee]
            )

        model.costpp_c = Constraint(model.Idx_employee, rule=costpp_rule)

        model.prefpp = Var(
            model.Idx_employee,
            bounds=(0, np.max(data.preference) * sum(data.Demand.maxDemand)),
        )

        def prefpp_rule(model, Idx_employee):
            return (
                sum(
                    model.x[Idx_employee, i] * data.preference[Idx_employee, i]
                    for i in model.Idx_shift
                )
                == model.prefpp[Idx_employee]
            )

        model.prefpp_c = Constraint(model.Idx_employee, rule=prefpp_rule)

        def obj_rule(model):
            return sum(
                data.regDict["weightPreference"] * model.prefpp[i]
                - data.regDict["weightCost"] * model.costpp[i]
                for i in model.Idx_employee
            )

        model.obj = Objective(rule=obj_rule, sense=maximize)

        return model

    def runOptimizer(self, data):
        if self.solverChoice == "cplex":
            opt = SolverFactory(self.solverChoice, executable=self.solverpath_exe)
            #        opt.options["threads"] = 4
            opt.options["timelimit"] = 30
            opt.options["mipgap"] = 0.02

        elif self.solverChoice == "glpk":
            opt = SolverFactory(self.solverChoice)
            opt.options["tmlim"] = 30
            opt.options["mipgap"] = 0.02
        else:
            opt = SolverFactory(self.solverChoice)

        self.model.preprocess()
        if self.solverChoice == "gdpopt":
            opt.solve(
                self.model,
                time_limit=30,
                mip_solver=self.mip,
                nlp_solver=self.nlp,
                bound_tolerance=1e-8,
                iterlim=30,
                tee=False,
            )
        else:
            opt.solve(self.model, tee=True)
        x = list(self.model.x.get_values().values())
        self.result = x

        return x


def workerProcess(dataInput):
    filename = "record_" + str(dataInput[0]) + ".txt"
    data = dataInput[1]
    data.hourlyOutput = dataInput[2]
    print(
        "process {} start time: {} hourly output: {}".format(
            dataInput[0], datetime.now(), data.hourlyOutput
        )
    )
    model = ShiftModel(modelData=data, solverChoice="gdpopt", mip="glpk", nlp="ipopt")

    if Specs.SAVELOG:
        log = utLog.openLog(filename=filename)

    x = model.runOptimizer(data)

    if Specs.SAVELOG:
        for k in [
            "minWorkHour",
            "maxWorkHour",
            "minShiftsPerWeek",
            "maxShiftsPerWeek",
            "minShiftContinuous",
            "minNightShiftContinuous",
        ]:
            log.write(k + ":{}\n".format(data.regDict[k]))
        utLog.closeLog(log, endMsg="ending solver")

    data.checkResult(x, filename=filename)

    if Specs.SAVELOG:
        log = utLog.openLog(filename=filename, titleMsg="printing model params")
        log.write("model.contNightShifts: \n")
        try:
            v = []
            for i in model.model.contNightShifts:
                v.append(model.model.contNightShifts[i].value)
            v = np.array(v, dtype=int).reshape(len(data.employeeNames), -1).astype(str)
            for i in v:
                log.write(" ".join(i.tolist()) + "\n")
        except AttributeError:
            pass
        log.write("\n##### finished printing model params #####\n\n")
        log.write("\n##### printing output: #####\n\n")
        log.write("shift assignment:\n")
        utLog.printDf(log, data.shiftAssignment)
        log.write("schedule:\n")
        utLog.printDf(log, data.schedule.iloc[:, 4:])
        utLog.closeLog(log, endMsg="finished printing output")
    result = np.hstack([dataInput[2], data.rating])
    print(
        "process {} end time: {} hourly output: {}".format(
            dataInput[0], datetime.now(), data.hourlyOutput
        )
    )
    return result


def runAssignment():
    data = DataProblem()
    h = utCom().createCombinations(data.hourlyOutput)
    idx = range(len(h))
    dataInput = list(zip(idx, [data] * len(h), h))

    cpus = cpu_count()
    timeout = Specs.THREAD_TIMEOUT
    result = list()
    with ProcessPool(max_workers=cpus) as pool:
        future = pool.map(workerProcess, dataInput, timeout=timeout)
        iterator = future.result()
        while True:
            try:
                res = next(iterator)
                result.append(res)
            except TimeoutError as error:
                print("function took longer than %d seconds" % error.args[1])
            except StopIteration:
                break
            except:  # noqa E722
                pass

        print("finished.")
    result_all = np.vstack(result)

    cr = c.ResultOutputColumns
    coln = ["hourlyOutput" + n for n in data.employeeNames]
    coln = coln + cr.all_output_cols
    if len(coln) == result_all.shape[1]:
        resultTbl = pd.DataFrame(result_all, columns=coln)
    else:
        resultTbl = pd.DataFrame(result_all)
    data.writeCsv(resultTbl, pathOut, of.RESULT_ALL_CSV)

    for maxIdx in cr.max_rule_idx_cols:
        maxRule = np.max(resultTbl[maxIdx])
        resultTbl = resultTbl.loc[resultTbl[maxIdx] == maxRule, :]
    for minIdx in cr.min_rule_idx_cols:
        minRule = np.min(resultTbl[minIdx])
        resultTbl = resultTbl.loc[resultTbl[minIdx] == minRule, :]
    resultTbl = resultTbl.iloc[0, : len(data.employeeNames)]

    data.hourlyOutput = resultTbl
    model = ShiftModel(modelData=data, solverChoice="gdpopt", mip="glpk", nlp="ipopt")
    x = model.runOptimizer(data)
    data.checkResult(x, filename=of.FINAL_RECORD_TXT, savelogs=Specs.SAVELOG)
    data.writeCsv(pd.DataFrame(data.schedule), pathOut, of.SCHEDULE_CSV)
    data.writeCsv(pd.DataFrame(data.shiftAssignment), pathOut, of.SHIFT_ASSIGN_CSV)


# %%
if __name__ == "__main__":
    runAssignment()
