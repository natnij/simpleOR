"""
@author: natnij
"""
import numpy as np
import pandas as pd
from simpleor.config.config_assignment import (
    RESOURCES_INPUT_DIR as pathIn,
    InputFiles as f,
    Specs,
)
from simpleor.supports.data_loader import DataLoader
from simpleor.supports.column_names import (
    OriginalColumnNames_assignment as o,
    ColumnNames_assignment as c,
)
from simpleor.utils.log_utils import LogUtilities as utLog


class DataProblem(DataLoader):
    def __init__(
        self,
        demandFile=f.DEMAND_CSV,
        shiftFile=f.SHIFT_CSV,
        prefFile=f.PREFERENCE_CSV,
        availFile=f.AVAILABILITY_CSV,
        regFile=f.REGULATION_CSV,
        costFile=None,
    ):
        DataLoader.__init__(self)
        od = o.OriginalDemandColumns
        oreg = o.OriginalRegulationColumns
        cd = c.DemandColumns
        creg = c.RegulationColumns
        op = o.OriginalPreferenceColumns
        cp = c.PreferenceColumns
        oa = o.OriginalAvailColumns
        ca = c.AvailColumns
        oc = o.OriginalCostColumns

        self.similarShiftHours = Specs.SIMILAR_SHIFT_HOURS
        self.defaultPref = 1

        self.regulation = self.loadCsv(pathIn, regFile, idx=oreg.origKey, dtype=None)
        self.regulation = self.cleanseData(self.regulation, oreg, creg, creg.filters)
        self.regDict = dict(
            self.regulation.loc[~self.regulation.value.isnull(), creg.value]
        )

        self.demand = self.loadCsv(pathIn, demandFile, dtype=None)
        self.demand = self.cleanseData(self.demand, od, cd, cd.filters)
        self.timeTbl = self.demand[
            [cd.isHoliday, cd.weekNr, cd.dayOfWeek, cd.hourIdx, cd.hourOfDay]
        ]
        self.schedule = self.timeTbl.copy()  # placeholder for result
        self.Demand = Demand(
            demand=np.array(self.demand[cd.demand]),
            maxDemand=np.array(self.demand[cd.maxCapa]),
            minDemand=np.array(self.demand[cd.minCapa]),
        )

        self.baseShift = self.loadCsv(pathIn, shiftFile, dtype=None)
        self.baseNightShift = self.baseShift.loc[
            (self.baseShift.hours > self.regDict["nightShiftDefinitionStart"])
            | (self.baseShift.hours <= self.regDict["nightShiftDefinitionEnd"]),
            :,
        ]
        self.baseShift = self.baseShift.iloc[:, 4:]
        self.baseNightShift = (
            np.array(np.sum(self.baseNightShift.iloc[:, 4:], axis=0))
            .astype(bool)
            .astype(int)
        )
        self.allNightShifts = np.array(
            self.baseNightShift.tolist() * int(len(self.timeTbl) / len(self.baseShift))
        )
        self.minShiftLen = np.min(np.sum(self.baseShift, axis=0))
        self.maxShiftLen = np.max(np.sum(self.baseShift, axis=0))

        self.shift, self.shiftTypeNames = self.extendShiftFromBase(self.baseShift)
        self.noGoShiftGroups = (
            self.createNoGoShifts()
        )  # at least 12 hours between shifts
        self.noGoNightShifts = (
            self.createNoGoNightShifts()
        )  # continuous night shifts <= 5
        # at least 2 continuous similar shifts, shifts with more
        # than 8 overlapping hours are considered similar
        self.continuousShifts = self.createContinuousShifts(
            considerSimilarShifts=True, similarShiftHours=self.similarShiftHours
        )
        self.continuousNightShifts = self.createContinuousShifts(
            selectedBaseShifts=self.baseNightShift,
            minShiftContinuous=self.regDict["minNightShiftContinuous"],
            considerSimilarShifts=True,
            similarShiftHours=self.similarShiftHours,
        )

        self.originalPreference = self.loadCsv(pathIn, prefFile, dtype=None)
        self.originalPreference = self.cleanseData(
            self.originalPreference, op, cp, cp.filters
        )
        self.priority = np.array(self.originalPreference[cp.employeePrio])
        self.hourlyOutput = np.array(self.originalPreference[cp.hourlyOutput])
        self.hourlyWage = np.array(self.originalPreference[cp.hourlyWage])
        self.hourlyWage = self.hourlyWage / np.max(
            self.hourlyWage
        )  # all variables used in the objective are standardized
        self.preference = np.array(
            self.originalPreference.apply(
                lambda row: self.updatePreference(row, cp.employeePrio), axis=1
            )
        )
        self.preference = self.preference / np.sum(self.preference)
        self.noGoPreference = np.where(self.preference == 0, 1, 0)
        self.shiftAssignment = None  # placeholder for result

        self.availability = self.loadCsv(pathIn, availFile, dtype=None)
        self.availability = self.cleanseData(self.availability, oa, ca, ca.filters)
        self.availability = self.availability.loc[
            :, ~self.availability.columns.isin(self.timeTbl.columns)
        ]
        self.nonAvail = 1 - np.array(self.availability).transpose()
        self.employeeNames = self.availability.columns.tolist()

        if costFile is not None:
            originalCost = self.loadCsv(pathIn, costFile, dtype=None)
            self.cost = np.array(originalCost[oc.origCost])
        else:
            self.cost = self.createCostFromReg()
        self.originalCost = self.cost.copy()
        self.cost = self.cost / np.sum(self.cost)

        self.Employees = self.availability.apply(
            lambda col: self.createEmployees(col, cp), axis=0
        ).tolist()
        self.shiftTypes = self.shift.apply(
            lambda col: self.createShiftTypes(col), axis=0
        ).tolist()

        self.checkFeasibility(cd)

    def checkFeasibility(self, cd):
        supply = 0
        for p in self.Employees:
            supply = (
                supply
                + min([sum(p.availability), self.regDict["maxWorkHour"]])
                * p.hourlyOutput
            )
        demand = sum(self.Demand.demand)
        if supply < demand:
            ratio = supply / demand
            print(
                "###### overall availability is smaller than demand, "
                + "reducing demand to {}. #######".format(ratio)
            )
            self.Demand.demand = self.Demand.demand * ratio
            self.Demand.minDemand = np.min(
                [self.Demand.minDemand, self.Demand.demand], axis=0
            )

    def createCostFromReg(self):
        baseCost = np.array(
            [self.regDict["standardShiftCostPerPersonPerHour"]] * len(self.timeTbl)
        )

        def we(x):
            return int(
                x >= self.regDict["additionalWeekendDayStart"]
                and x <= self.regDict["additionalWeekendDayEnd"]
            )

        weekDayCost = (
            np.array([we(x) for x in self.timeTbl.days])
            * self.regDict["additionalWeekendShiftCost"]
        )

        def night(x):
            return int(
                x > self.regDict["additionalNightShiftPaymentStart"]
                or x <= self.regDict["additionalNightShiftPaymentEnd"]
            )

        nightCost = (
            np.array([night(x) for x in self.timeTbl.hourPerDay])
            * self.regDict["additionalNightShiftCost"]
        )

        publicHolidayCost = (
            np.array(self.timeTbl.publicHoliday)
            * self.regDict["additionalPublicHolidayCost"]
        )

        return baseCost + weekDayCost + nightCost + publicHolidayCost

    def updatePreference(self, row, prioCol):
        tmp = np.array(row[-len(self.shiftTypeNames) :])  # noqa E203
        prio = row[prioCol]
        return pd.Series(tmp * prio)

    def extendShiftFromBase(self, baseShift):
        multiplier = int(len(self.timeTbl) / len(baseShift))
        coln = baseShift.columns.tolist()
        colnames = list()
        shiftTbl = None
        for i in range(multiplier):
            colnames.extend([x + str(i + 1) for x in coln])
            upperFill = np.zeros((i * len(baseShift), baseShift.shape[1]), dtype=int)
            lowerFill = np.zeros(
                ((multiplier - i - 1) * len(baseShift), baseShift.shape[1]), dtype=int
            )
            tbl = np.vstack((upperFill, np.array(baseShift, dtype=int), lowerFill))
            if shiftTbl is None:
                shiftTbl = tbl
            else:
                shiftTbl = np.hstack((shiftTbl, tbl))
        shiftTbl = pd.DataFrame(shiftTbl, columns=colnames)

        # deal with shifts at midnight:
        weirdBaseShifts = np.array(baseShift.iloc[0, :]) * np.array(
            baseShift.iloc[-1, :]
        )
        weirdBaseShiftHoursPart1 = (
            np.sum(baseShift.iloc[:12, :], axis=0) * weirdBaseShifts
        )
        weirdBaseShiftHoursPart2 = (
            np.sum(baseShift.iloc[12:, :], axis=0) * weirdBaseShifts
        )
        weirdShifts = np.hstack([weirdBaseShifts] * multiplier)
        for i in range(len(weirdShifts)):
            if weirdShifts[i] == 1:
                if i < len(weirdBaseShifts):
                    shiftTbl.iloc[:, i] = np.hstack(
                        (
                            np.ones(weirdBaseShiftHoursPart1[i], dtype=int),
                            np.zeros(
                                len(shiftTbl)
                                - weirdBaseShiftHoursPart1[i]
                                - weirdBaseShiftHoursPart2[i],
                                dtype=int,
                            ),
                            np.ones(weirdBaseShiftHoursPart2[i], dtype=int),
                        )
                    )
                else:
                    tmp = shiftTbl.iloc[:, i].tolist()
                    firstOne = tmp.index(1)
                    baseIdx = int(np.mod(i, len(weirdBaseShifts)))
                    part1Len = weirdBaseShiftHoursPart1[baseIdx]
                    part2Len = weirdBaseShiftHoursPart2[baseIdx]
                    tmp[firstOne - part2Len : firstOne] = [1] * part2Len  # noqa E203
                    tmp[firstOne + part1Len :] = [0] * len(  # noqa E203
                        tmp[firstOne + part1Len :]  # noqa E203
                    )
                    shiftTbl.iloc[:, i] = tmp
        return shiftTbl, colnames

    def createEmployees(self, col, cp):
        name = col.name
        pref = self.originalPreference.loc[
            self.originalPreference[cp.employeeName] == name,
        ]
        prio = int(pref[cp.employeePrio])
        hourlyOutput = int(pref[cp.hourlyOutput])
        hourlyWage = int(pref[cp.hourlyWage])
        if len(pref) == 0:
            pref = np.array([self.defaultPref] * (self.preference.shape[1]))
        else:
            pref = np.array(pref.iloc[0, -self.preference.shape[1] :])  # noqa E203
        employee = Employee(
            name=name,
            availability=np.array(col),
            preference=pref,
            priority=prio,
            hourlyOutput=hourlyOutput,
            timeTbl=self.timeTbl.copy(),
            shiftTypeNames=self.shiftTypeNames.copy(),
            hourlyWage=hourlyWage,
        )
        return employee

    def createShiftTypes(self, col):
        name = col.name
        shift = ShiftType(name=name, shiftTime=np.array(col))
        return shift

    def reformatTable(self, conditionTbl, defaultValue, defaultLen):
        # deduplicate
        conditionTbl = np.unique(conditionTbl, axis=0)
        # delete zero rows
        conditionTbl = conditionTbl[~np.all(conditionTbl == 0, axis=1)]
        # return default values in default shapes
        if conditionTbl.shape[0] == 0:
            conditionTbl = np.array([defaultValue] * defaultLen, dtype=int).reshape(
                1, -1
            )
        return conditionTbl

    def createNoGoShifts(self):
        """overlapping shifts within 12 hours are not permitted"""
        interval = int(self.regDict["minHourBetweenShift"])
        noGoShiftGroups = self.shift.copy()
        noGoShiftGroups = pd.concat(
            [noGoShiftGroups, noGoShiftGroups.iloc[0:interval, :]], axis=0
        )
        conditions = None
        pointer = 0
        while pointer + interval < len(noGoShiftGroups):
            newLine = (
                np.array(
                    np.sum(
                        noGoShiftGroups.iloc[
                            pointer : pointer + interval, :  # noqa E203
                        ],
                        axis=0,
                    )
                )
                .astype(bool)
                .astype(int)
            )
            if conditions is None:
                conditions = newLine
            else:
                conditions = np.vstack((conditions, newLine))
            pointer = pointer + 1  # int(self.minShiftLen / 2)

        return self.reformatTable(
            conditions, defaultValue=0, defaultLen=self.shift.shape[1]
        )

    def createNoGoNightShifts(self):
        noGoNightShifts = self.baseNightShift.tolist()
        multiplier = int(len(self.timeTbl) / len(self.baseShift))

        maxlen = np.min(
            (multiplier, (int(self.regDict["maxNightShiftContinuous"]) + 1))
        )
        zerolen = multiplier - maxlen
        conditions = np.array(noGoNightShifts * maxlen, dtype=int)
        if zerolen == 0:
            return self.reformatTable(
                conditions, defaultValue=0, defaultLen=self.shift.shape[1]
            )

        conditions = np.hstack(
            (
                conditions,
                np.zeros((zerolen * len(noGoNightShifts)), dtype=int),
                conditions,
            )
        )
        conditionTbl = None
        pointer = 0
        for i in range(multiplier):
            if conditionTbl is None:
                conditionTbl = conditions[
                    pointer : pointer + len(noGoNightShifts) * multiplier  # noqa E203
                ]
            else:
                conditionTbl = np.vstack(
                    (
                        conditionTbl,
                        conditions[
                            pointer : pointer  # noqa E203
                            + len(noGoNightShifts) * multiplier
                        ],
                    )
                )
            pointer = pointer + len(noGoNightShifts)
        return self.reformatTable(
            conditionTbl, defaultValue=0, defaultLen=self.shift.shape[1]
        )

    def createContinuousShifts(
        self,
        baseShift=None,
        selectedBaseShifts=None,
        minShiftContinuous=None,
        considerSimilarShifts=True,
        similarShiftHours=Specs.SIMILAR_SHIFT_HOURS,
    ):
        if baseShift is None:
            baseShift = self.baseShift.copy()
        multiplier = int(len(self.timeTbl) / len(baseShift))
        if selectedBaseShifts is None:
            selectedBaseShifts = np.array([1] * baseShift.shape[1])
        if minShiftContinuous is None:
            minShiftContinuous = self.regDict["minShiftContinuous"]
        minShiftContinuous = int(minShiftContinuous)
        if minShiftContinuous == 0:
            return self.reformatTable(
                np.zeros([1, 1]),
                defaultValue=0,
                defaultLen=baseShift.shape[1] * multiplier,
            )

        conditions = None
        if considerSimilarShifts:
            shiftTbl = pd.concat(
                [baseShift, baseShift.iloc[0:similarShiftHours, :]], axis=0
            )
            pointer = 0
            while pointer + similarShiftHours < len(shiftTbl):
                newLine = (
                    np.sum(
                        shiftTbl.iloc[
                            pointer : pointer + similarShiftHours, :  # noqa E203
                        ],
                        axis=0,
                    )
                    >= similarShiftHours
                ).astype(int)
                if np.sum(newLine) < 2:
                    pass
                elif conditions is None:
                    conditions = newLine
                else:
                    conditions = np.vstack((conditions, newLine))
                pointer = pointer + 1
            conditions = np.unique(conditions, axis=0)

        if conditions is None:
            conditions = np.eye(baseShift.shape[1], dtype=int) * selectedBaseShifts
        else:
            conditions = np.vstack(
                (conditions, np.eye(baseShift.shape[1], dtype=int) * selectedBaseShifts)
            )
        conditions = conditions[~np.all(conditions == 0, axis=1)]

        zeroFill = np.zeros_like(conditions)
        zeroFill = np.hstack([zeroFill] * (multiplier - minShiftContinuous))
        conditionTbl = np.hstack([conditions] * minShiftContinuous)
        conditionTbl = np.hstack((conditionTbl, zeroFill, conditionTbl, zeroFill))

        pointer = 0
        conditionTbl0 = None
        for i in range(multiplier):
            if conditionTbl0 is None:
                conditionTbl0 = conditionTbl[
                    :, pointer : pointer + baseShift.shape[1] * multiplier  # noqa E203
                ]
            else:
                conditionTbl0 = np.vstack(
                    (
                        conditionTbl0,
                        conditionTbl[
                            :,
                            pointer : pointer  # noqa E203
                            + baseShift.shape[1] * multiplier,
                        ],
                    )
                )
            pointer = pointer + baseShift.shape[1]
        return self.reformatTable(
            conditionTbl0, defaultValue=0, defaultLen=baseShift.shape[1] * multiplier
        )

    def findShiftType(self, shiftTypeName):
        for shift in self.shiftTypes:
            if shift.name == shiftTypeName:
                return shift
        return None

    def findEmployee(self, employeeName):
        for employee in self.Employees:
            if employee.name == employeeName:
                return employee
        return None

    def updateShiftAndEmployee(self, employee, shift):
        check = employee.addShift(shift)
        if check:
            shift.employeesOnshift.append(employee)
            return (employee.name, shift.name, "success")
        else:
            return (employee.name, shift.name, "failure")

    def checkResult(self, result, filename, savelogs=Specs.SAVELOG):
        self.shiftAssignment = pd.DataFrame(
            np.array(result, dtype=int).reshape(len(self.employeeNames), -1),
            columns=self.shiftTypeNames,
            index=self.employeeNames,
        )
        self.updateResult = [
            self.updateShiftAndEmployee(
                self.findEmployee(
                    self.employeeNames[int(idx / len(self.shiftTypeNames))]
                ),
                self.findShiftType(
                    self.shiftTypeNames[np.mod(idx, len(self.shiftTypeNames))]
                ),
            )
            for idx, i in enumerate(result)
            if i == 1
        ]
        self.Demand.calDemandFulfillment(self.Employees)
        if savelogs:
            log = utLog.openLog(filename=filename, titleMsg="start recording")
            log.write("demand fulfilled: {}\n".format(self.Demand.totalFulfillment))
            log.write("totalDemand: {}\n".format(sum(self.Demand.demand)))
            log.write(
                "maxDemand: {}\nminDemand: {}\n".format(
                    max(self.Demand.maxDemand), max(self.Demand.minDemand)
                )
            )
            log.write("totalSupply: {}\n".format(sum(self.Demand.supply)))
            log.write(
                "hourlyOutput: {}\n".format(
                    " ".join(self.hourlyOutput.astype(str).tolist())
                )
            )

        self.rating = None
        self.schedule = self.timeTbl.copy()
        for p in self.Employees:
            p.calCost(self.originalCost)
            p.calBenefit()
            p.calNightShift(
                self.noGoNightShifts,
                self.regDict["maxNightShiftContinuous"],
                self.continuousNightShifts,
                self.regDict["minNightShiftContinuous"],
            )
            p.calContinuousShifts(
                self.continuousShifts, self.regDict["minShiftContinuous"]
            )
            p.calBtwShifts(self.regDict["minHourBetweenShift"])
            self.schedule[p.name] = p.timeTbl.onDuty

            rating = [
                sum(p.timeTbl.onDuty),
                len(p.onShifts),
                p.totalCost,
                p.totalPref,
                p.checkNoGoPref(),
                p.minHourBtwRules,
                p.maxNightShiftRules_all & p.minNightShiftRules_all,
                p.minContShiftRules_all,
            ]
            if self.rating is None:
                self.rating = np.array(rating)
            else:
                self.rating = self.rating + np.array(rating)
            if savelogs:
                info = (
                    "{}:workhour:{},nrShifts:{},cost:{},pref:{},noGoPref:{},"
                    + "minHourBtwShift:{},nightShift:{},contShift:{}.\n"
                )
                log.write(info.format(p.name, *rating))
        self.rating = self.rating.astype(int)
        if savelogs:
            utLog.closeLog(log, endMsg="end recording")


class Employee:
    def __init__(
        self,
        name,
        availability,
        preference,
        priority,
        hourlyOutput,
        timeTbl,
        shiftTypeNames,
        hourlyWage,
    ):
        self.name = name
        self.availability = availability
        self.shiftTypeNames = shiftTypeNames
        self.preferenceWithPrio = preference * priority
        self.originalPreference = preference
        self.priority = priority
        self.hourlyOutput = hourlyOutput
        self.hourlyWage = hourlyWage
        self.onShifts = list()
        self.onShiftTbl = np.zeros_like(preference)
        self.timeTbl = timeTbl
        self.ct = c.TimeTblColumns
        self.timeTbl[self.ct.onduty] = 0

    def addShift(self, shiftType):
        req = shiftType.shiftTime
        check_avail = np.alltrue(self.availability & req == req)
        check_onduty = np.alltrue(
            (1 - np.array(self.timeTbl[self.ct.onduty])) & req == req
        )
        if check_avail & check_onduty:
            self.onShifts.append(shiftType)
            self.onShiftTbl[self.shiftTypeNames.index(shiftType.name)] = 1
            self.timeTbl[self.ct.onduty] = self.timeTbl[self.ct.onduty] + req
            return True
        else:
            return False

    def calCost(self, cost):
        self.totalCost = np.sum(
            np.array(self.timeTbl[self.ct.onduty]) * self.hourlyWage * cost
        )

    def calBenefit(self):
        self.totalPref = np.sum(np.array(self.onShiftTbl) * self.originalPreference)

    def calNightShift(
        self, maxContNightshifts, maxNights, minContNightshifts, minNights
    ):
        myshifts = self.onShiftTbl.copy().reshape(-1, 1)
        assert maxContNightshifts.shape[1] == myshifts.shape[0]
        self.maxNightShiftRules = np.matmul(
            maxContNightshifts.astype(float), myshifts.astype(float)
        )
        self.maxNightShiftRules_all = np.alltrue(self.maxNightShiftRules <= maxNights)

        assert minContNightshifts.shape[1] == myshifts.shape[0]
        self.minNightShiftRules = np.matmul(
            minContNightshifts.astype(float), myshifts.astype(float)
        )
        self.minNightShiftRules_all = (
            np.max(self.minNightShiftRules >= minNights)
            or np.sum(self.minNightShiftRules) == 0
        )

    def calBtwShifts(self, mustMinHourBtw):
        myHours = np.array(self.timeTbl[self.ct.onduty])
        minHourBtw = len(myHours) - 1
        maxHourBtw = 0
        firstOne = len(myHours)
        lastOne = -1
        count = 0
        for idx in range(len(myHours)):
            if firstOne < len(myHours):
                if myHours[idx] == 0:
                    count = count + 1
                if myHours[idx] == 1 and myHours[idx - 1] == 0:
                    minHourBtw = min([minHourBtw, count])
                    maxHourBtw = max([maxHourBtw, count])
                    count = 0
            if myHours[idx] == 1:
                firstOne = min([firstOne, idx])
                lastOne = max([lastOne, idx])

        crossWeek = (firstOne if firstOne < len(myHours) else 0) + (
            len(myHours) - 1 - lastOne if lastOne >= 0 else 0
        )
        if crossWeek > 0:
            minHourBtw = min([minHourBtw, crossWeek])
            maxHourBtw = max([maxHourBtw, crossWeek])
        self.minHourBtwRules = minHourBtw >= mustMinHourBtw
        self.minHourBtw = minHourBtw
        self.maxHourBtw = maxHourBtw

    def calContinuousShifts(self, minContshifts, minDays):
        myshifts = self.onShiftTbl.copy().reshape(-1, 1)
        assert minContshifts.shape[1] == myshifts.shape[0]
        self.minContShiftRules = np.matmul(
            minContshifts.astype(float), myshifts.astype(float)
        )
        self.minContShiftRules_all = (np.max(self.minContShiftRules) >= minDays) | (
            np.sum(myshifts) == 0
        )

    def checkNoGoPref(self):
        return sum(self.onShiftTbl * np.where(self.originalPreference == 0, 1, 0)) == 0


class ShiftType:
    def __init__(self, name, shiftTime):
        self.name = name
        self.shiftTime = shiftTime
        self.employeesOnshift = list()


class Demand:
    def __init__(self, demand, maxDemand, minDemand):
        self.demand = demand
        self.originalDemand = demand
        self.maxDemand = maxDemand
        self.minDemand = minDemand
        self.supply = np.zeros_like(demand)
        self.hourlyFulfillment = np.zeros_like(demand)
        self.totalFulfillment = False

    def calDemandFulfillment(self, employees):
        for employee in employees:
            self.supply = (
                self.supply
                + np.array(employee.timeTbl[[employee.ct.onduty]])
                * employee.hourlyOutput
            )
        self.minDemandFulfilled = self.supply >= self.minDemand
        self.maxDemandFulfilled = self.supply <= self.maxDemand
        self.hourlyFulfillment = np.sum(
            (self.minDemandFulfilled & self.maxDemandFulfilled)
        ) == len(self.demand)
        self.totalFulfillment = sum(self.supply) >= sum(self.originalDemand)
