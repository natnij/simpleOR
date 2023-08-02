"""
@author: natnij
"""
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from multiprocessing import cpu_count
from datetime import datetime
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    Binary,
    Reals,
    maximize,
    SolverFactory,
)
import pandas as pd
import numpy as np
from simpleor.config.config_jobshopScheduling import (
    Specs,
    LOGS_DIR as pathLog,
    RESOURCES_OUTPUT_DIR as pathOut,
    OutputFiles as of,
)
from simpleor.supports.data_loader_jobshopScheduling import DataProblem
from simpleor.supports.column_names import ColumnNames_jobshop as c
from simpleor.utils.common_utils import CommonUtils as utCom


class JobshopModel:
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
        model.Idx_task = range(data.resourceTbl.shape[0])
        model.Idx_resource = range(len(data.resources))
        model.Idx_resource_nonzero = data.idx_resource_nonzero
        model.Idx_resourceTbl_nonzero = list(
            zip(
                data.resourceTbl_sparse.nonzero()[0],
                data.resourceTbl_sparse.nonzero()[1],
            )
        )
        greaterThanOne = data.resourceTbl_sparse > 1
        model.Idx_resourceTbl_greaterThanOne = list(
            zip(greaterThanOne.nonzero()[0], greaterThanOne.nonzero()[1])
        )
        model.Idx_timeslot = range(data.planHorizon)
        model.Idx_precedence = range(data.precedenceRulesPos.shape[0])
        model.x = Var(model.Idx_task, model.Idx_timeslot, within=Binary, initialize=1)

        # calculate xblock
        model.xblock0 = Var(
            model.Idx_precedence, model.Idx_timeslot, within=Binary, initialize=0
        )

        def calBlock0_rule(model, Idx_precedence, Idx_timeslot):
            return (
                sum(
                    data.precedenceRulesPos[Idx_precedence, i]
                    * model.x[i, Idx_timeslot]
                    for i in model.Idx_task
                )
                == model.xblock0[Idx_precedence, Idx_timeslot]
            )

        model.xblock0_c = Constraint(
            model.Idx_precedence, model.Idx_timeslot, rule=calBlock0_rule
        )

        model.xblock = Var(
            model.Idx_precedence, model.Idx_timeslot, within=Binary, initialize=0
        )

        def calBlock_rule(model, Idx_precedence, Idx_timeslot):
            return (
                sum(
                    data.durationTbl[int(data.precedenceDuration[Idx_precedence])][
                        Idx_timeslot, j
                    ]
                    * model.xblock0[Idx_precedence, j]
                    for j in model.Idx_timeslot
                )
                == model.xblock[Idx_precedence, Idx_timeslot]
            )

        model.xblock_c = Constraint(
            model.Idx_precedence, model.Idx_timeslot, rule=calBlock_rule
        )

        # enforce precedence with no overlap
        model.precedenceVar = Var(
            model.Idx_precedence, model.Idx_timeslot, within=Reals, initialize=0
        )

        def calPrecedence_rule(model, Idx_precedence, Idx_timeslot):
            return (
                model.xblock[Idx_precedence, Idx_timeslot]
                + sum(
                    data.precedenceRulesNeg[Idx_precedence, i]
                    * model.x[i, Idx_timeslot]
                    * data.precedenceDuration[Idx_precedence]
                    for i in model.Idx_task
                )
                == model.precedenceVar[Idx_precedence, Idx_timeslot]
            )

        model.precedenceVar_c = Constraint(
            model.Idx_precedence, model.Idx_timeslot, rule=calPrecedence_rule
        )

        model.posPrecedenceVar = Var(
            model.Idx_precedence,
            model.Idx_timeslot,
            bounds=(-data.planHorizon, data.planHorizon),
            initialize=0,
        )

        def calPos_rule(model, Idx_precedence, Idx_timeslot):
            helper = np.zeros(data.planHorizon, dtype=int)
            helper[0:Idx_timeslot] = 1
            return (
                sum(
                    model.precedenceVar[Idx_precedence, i] * helper[i]
                    for i in model.Idx_timeslot
                )
                == model.posPrecedenceVar[Idx_precedence, Idx_timeslot]
            )

        model.posPrecedenceVar_c = Constraint(
            model.Idx_precedence, model.Idx_timeslot, rule=calPos_rule
        )

        def calPos2_rule(model, Idx_precedence, Idx_timeslot):
            return model.posPrecedenceVar[Idx_precedence, Idx_timeslot] >= 0

        model.posPrecedenceVar2_c = Constraint(
            model.Idx_precedence, model.Idx_timeslot, rule=calPos2_rule
        )

        # make sure every task has one and only one start time.
        def taskStart_rule(model, Idx_task):
            return sum(model.x[Idx_task, j] for j in model.Idx_timeslot) == 1

        model.taskStart_c = Constraint(model.Idx_task, rule=taskStart_rule)

        # make sure every task-resource combination has the correct duration planned
        def taskDuration_rule(
            model, Idx_task_greaterThanOne, Idx_resource_greaterThanOne
        ):
            duration = (
                int(
                    data.resourceTbl[
                        Idx_task_greaterThanOne, Idx_resource_greaterThanOne
                    ]
                )
                - 1
            )
            return (
                sum(
                    model.x[Idx_task_greaterThanOne, i]
                    for i in range(data.planHorizon - duration, data.planHorizon)
                )
                == 0
            )

        model.taskDuration_c = Constraint(
            model.Idx_resourceTbl_greaterThanOne, rule=taskDuration_rule
        )

        model.resourceUsed = Var(
            model.Idx_resource, model.Idx_timeslot, within=Reals, initialize=0
        )

        def calResourceUsed_rule(model, Idx_resource_nonzero, Idx_timeslot):
            task_nonzero = [
                taskId
                for taskId, resourceId in model.Idx_resourceTbl_nonzero
                if resourceId == Idx_resource_nonzero
            ]
            return (
                sum(
                    sum(
                        data.durationTbl[
                            int(data.resourceTbl[i, Idx_resource_nonzero])
                        ][Idx_timeslot, j]
                        * model.x[i, j]
                        for j in model.Idx_timeslot
                    )
                    for i in task_nonzero
                )
                == model.resourceUsed[Idx_resource_nonzero, Idx_timeslot]
            )

        model.resourceUsed_c = Constraint(
            model.Idx_resource_nonzero, model.Idx_timeslot, rule=calResourceUsed_rule
        )

        def resourceCapa_rule(model, Idx_resource_nonzero, Idx_timeslot):
            return (
                model.resourceUsed[Idx_resource_nonzero, Idx_timeslot]
                <= data.resourceCapa[Idx_resource_nonzero]
            )

        model.resourceCapa_c = Constraint(
            model.Idx_resource_nonzero, model.Idx_timeslot, rule=resourceCapa_rule
        )

        model.costpp = Var(
            model.Idx_resource,
            bounds=(0, max(data.resourceCost) * data.planHorizon),
            initialize=max(data.resourceCost) * data.planHorizon,
        )

        def costpp_rule(model, Idx_resource):
            return (
                sum(
                    model.resourceUsed[Idx_resource, i]
                    * data.resourceCost[Idx_resource]
                    for i in model.Idx_timeslot
                )
                == model.costpp[Idx_resource]
            )

        model.costpp_c = Constraint(model.Idx_resource, rule=costpp_rule)

        model.taskPrio = Var(model.Idx_task, bounds=(0, max(data.prios)), initialize=0)

        def taskPrio_rule(model, Idx_task):
            return (
                sum(
                    model.x[Idx_task, i] * data.prios[Idx_task] * data.timeslotWeight[i]
                    for i in model.Idx_timeslot
                )
                == model.taskPrio[Idx_task]
            )

        model.taskPrio_c = Constraint(model.Idx_task, rule=taskPrio_rule)

        def obj_rule(model):
            return (
                sum(model.taskPrio[i] for i in model.Idx_task)
                / len(model.Idx_task)
                * data.jobPrioImportance
                - sum(model.costpp[j] for j in model.Idx_resource)
                / len(model.Idx_resource)
                * data.costImportance
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
                mip_solver=self.mip,  # mip_solver_args={'tmlim':30},
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


def getBlocks(data, x):
    xArray = x.copy()
    maxDuration = np.max(data.resourceTbl, axis=1)
    for row in range(xArray.shape[0]):
        idx = list(xArray[row]).index(1)
        xArray[row, idx : idx + maxDuration[row]] = 1  # noqa E203
    return xArray


def workerProcess(dataInput):
    processIdx, planHorizon = dataInput
    print(
        "process {} start time: {} planHorizon: {}".format(
            processIdx, datetime.now(), planHorizon
        )
    )
    data = DataProblem(planHorizon=planHorizon)
    model = JobshopModel(modelData=data, solverChoice="gdpopt", mip="glpk", nlp="ipopt")
    x = model.runOptimizer(data)

    xStart = np.array(x, dtype=int).reshape(data.resourceTbl.shape[0], data.planHorizon)
    xStartDf = pd.DataFrame(xStart, index=data.taskNames)
    data.xStart = xStartDf
    xBlock = getBlocks(data, xStart)
    xBlockDf = pd.DataFrame(xBlock, index=data.taskNames)
    data.xBlock = xBlockDf
    resourceUsed = np.array(
        list(model.model.resourceUsed.get_values().values()), dtype=int
    ).reshape(len(data.resources), -1)
    data.resourceUsed = pd.DataFrame(resourceUsed)
    if Specs.SAVELOG:
        model.model.pprint(
            utCom().patchFilePath([pathLog, "log_" + str(planHorizon) + ".txt"])
        )

    for i, resource in enumerate(data.resources):
        resource.used = resourceUsed[i]

    for project in data.projects:
        for job in project.jobs:
            tmp = [job.name in x for x in data.taskNames]
            tmpStart = xStartDf[tmp]
            tmpBlock = xBlockDf[tmp]
            job.start = tmpStart.copy()
            job.block = tmpBlock.copy()

    earliest = list(np.sum(xBlock, axis=0) > 0).index(True)
    latest = (
        xBlock.shape[1] - 1 - list(reversed(np.sum(xBlock, axis=0) > 0)).index(True)
    )
    totalDuration = latest - earliest
    taskPrio = sum(model.model.taskPrio.get_values().values())
    costpp = sum(model.model.costpp.get_values().values())
    objective = model.model.obj()
    totalTask = np.sum(xStart)
    totalResource = np.sum(resourceUsed)

    print(
        "process {} end time: {} planHorizon: {}".format(
            processIdx, datetime.now(), planHorizon
        )
    )
    return (
        data,
        totalDuration,
        objective,
        taskPrio,
        costpp,
        totalTask,
        totalResource,
        planHorizon,
    )


def runJobshopScheduling(start=None, end=None, step=None):
    if start is None:
        start = Specs.PLAN_HORIZON_START
        end = Specs.PLAN_HORIZON_END
        step = Specs.PLAN_HORIZON_STEP
    if end < start:
        end = start
    if step < 1:
        step = 1
    cs = c.ScoreOutputColumns
    planHorizonRange = [start, end]
    planHorizon = list(
        np.arange(
            start=planHorizonRange[0],
            stop=planHorizonRange[1] + step,
            step=step,
            dtype=int,
        )
    )

    cpus = cpu_count()
    timeout = Specs.THREAD_TIMEOUT
    result = list()
    with ProcessPool(max_workers=cpus) as pool:
        future = pool.map(workerProcess, enumerate(planHorizon), timeout=timeout)
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

    data = [x[0] for x in result]
    score = pd.DataFrame([], columns=cs.allCols)
    score[cs.planHorizon] = [x[7] for x in result]
    score[cs.totalDuration] = [x[1] for x in result]
    score[cs.objective] = [x[2] for x in result]
    score[cs.taskPrios] = [x[3] for x in result]
    score[cs.costPerPerson] = [x[4] for x in result]
    score[cs.totalTasks] = [x[5] for x in result]
    score[cs.totalResources] = [x[6] for x in result]
    score = score.sort_values(
        [cs.totalDuration, cs.planHorizon], ascending=[True, True]
    )
    ut = utCom().patchFilePath
    score.to_csv(ut([pathOut, of.SCORE_CSV]))
    winner = data[score.index[0]]
    winner.xStart.to_csv(ut([pathOut, of.START_TIME_CSV]))
    winner.xBlock.to_csv(ut([pathOut, of.DURATION_CSV]))
    winner.resourceUsed.to_csv(ut([pathOut, of.RESOURCE_USED_CSV]))


# %%
if __name__ == "__main__":
    runJobshopScheduling()
