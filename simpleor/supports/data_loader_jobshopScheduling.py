"""
@author: natnij
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from simpleor.config.config_jobshopScheduling import (
    RESOURCES_INPUT_DIR as pathIn,
    InputFiles as f,
    Specs,
)
from simpleor.supports.data_loader import DataLoader
from simpleor.supports.column_names import (
    OriginalColumnNames_jobshop as o,
    ColumnNames_jobshop as c,
)


class DataProblem(DataLoader):
    def __init__(
        self,
        routingFile=f.ROUTING_CSV,
        resourceFile=f.RESOURCE_CSV,
        costFile=f.COST_CSV,
        hierarchyFile=f.HIERARCHY_CSV,
        planHorizon=Specs.PLAN_HORIZON_START,
        planStrategy=Specs.PLAN_STRATEGY,
    ):
        DataLoader.__init__(self)

        oroute = o.OriginalRoutingColumns
        croute = c.RoutingColumns
        ores = o.OriginalTaskResourceReqColumns
        ocost = o.OriginalCostColumns
        ccost = c.CostColumns
        ohier = o.OriginalHierarchyColumns
        chier = c.HierarchyColumns

        self.routing = self.loadCsv(pathIn, routingFile, dtype=None)
        self.routing = self.cleanseData(self.routing, oroute, croute, croute.filters)

        self.baseResourceTbl = self.loadCsv(
            pathIn, resourceFile, idx=ores.origTasks, dtype=None
        )
        self.baseCost = self.loadCsv(pathIn, costFile, dtype=None)
        self.baseCost = self.cleanseData(self.baseCost, ocost, ccost, ccost.filters)
        coln = self.baseResourceTbl.columns.tolist()
        self.baseCost.index = self.baseCost[ccost.resourceIdx].apply(
            lambda x: coln.index(x)
        )
        self.baseCost.sort_index(inplace=True)

        self.resourceCapa = np.array(self.baseCost[ccost.resourceCapa])
        self.resourceCost = np.array(self.baseCost[ccost.unitCost]) / max(
            self.baseCost[ccost.unitCost]
        )

        self.hierarchy = self.loadCsv(pathIn, hierarchyFile, dtype=None)
        self.hierarchy = self.cleanseData(self.hierarchy, ohier, chier, chier.filters)
        self.originalPlanHorizon = planHorizon
        self.planHorizon = planHorizon
        self.checkFeasibility(
            multiplier=Specs.MULTIPLIER, croute=croute, chier=chier
        )  # bigger multiplier means more buffer in planHorizon

        self.planStrategy = planStrategy  # 'earliest', 'latest', 'leastCost'
        if planStrategy == "earliest":
            self.timeslotWeight = np.arange(self.planHorizon, 0, -1)
        elif planStrategy == "latest":
            self.timeslotWeight = np.arange(1, self.planHorizon, 1)
        else:
            self.timeslotWeight = np.ones(self.planHorizon)
        self.timeslotWeight = self.timeslotWeight / max(self.timeslotWeight)

        self.projects = [
            self.createProject(str(x))
            for x in list(set(self.hierarchy[chier.projectIdx]))
        ]
        self.hierarchy.apply(
            lambda row: self.createJobs(row, chier=chier, croute=croute), axis=1
        )
        self.resourceTbl, self.prios = self.combineResourceTbls()
        self.idx_resource_nonzero = np.sum(self.resourceTbl, axis=0).nonzero()[0]
        self.resourceTbl_sparse = csr_matrix(self.resourceTbl)
        self.resources = (
            pd.DataFrame(self.resourceTbl, columns=self.baseResourceTbl.columns)
            .apply(lambda col: self.createResource(col, ccost), axis=0)
            .tolist()
        )
        # precedenceDuration is for precedence constraints only
        (
            self.precedenceRulesPos,
            self.precedenceRulesNeg,
            self.precedenceDuration,
            self.taskNames,
        ) = self.combinePrecedenceTbls()
        self.durationTbl = self.createResourceRequirement()
        self.jobPrioImportance = 0.9
        self.costImportance = 0.1

    def checkFeasibility(self, multiplier, croute, chier):
        tmp0 = self.routing.loc[:, [croute.jobIdx, croute.taskIdx]]
        tmp1 = self.routing.loc[:, [croute.jobIdx, croute.postTaskIdx]]
        tmp1.columns = [croute.jobIdx, croute.taskIdx]
        tmp = pd.concat([tmp0, tmp1], axis=0)
        tmp.drop_duplicates(inplace=True)
        tmpHier = (
            self.hierarchy.loc[:, [chier.jobIdx, chier.count]]
            .groupby(chier.jobIdx, as_index=False)
            .sum()
        )
        tmp = tmp.merge(tmpHier, on=chier.jobIdx, how="left")
        tmp = tmp.groupby(croute.taskIdx).sum()
        tmp = tmp.to_dict()[chier.count]
        tmpRes = np.array(self.baseResourceTbl)
        for i in range(tmpRes.shape[0]):
            tmpRes[i] = tmpRes[i] * tmp[self.baseResourceTbl.index[i]]
        tmpRes = np.sum(tmpRes, axis=0)
        result = tmpRes * multiplier <= self.resourceCapa * self.planHorizon
        if not np.alltrue(result):
            extension = int(
                np.ceil(
                    np.max(
                        (tmpRes * multiplier - self.resourceCapa * self.planHorizon)
                        / self.resourceCapa
                    )
                )
            )
            print(
                "resource shortage or planHorizon too close. "
                + "Extending planHorzion to {}.".format(self.planHorizon + extension)
            )
            self.planHorizon = self.planHorizon + extension

    def createProject(self, projectName):
        return Project(projectName)

    def createJobs(self, row, chier, croute):
        projectName = row[chier.projectIdx]
        project = self.findProject(projectName)
        if project is None:
            project = self.createProject(projectName)
            self.projects.append(project)
        jobType = row[chier.jobIdx]
        nrOfSites = row[chier.count]
        prio = row[chier.jobTypePrio]
        taskDependencies = self.routing.loc[
            self.routing[croute.jobIdx] == jobType, [croute.taskIdx, croute.postTaskIdx]
        ]
        intervals = np.array(
            self.routing.loc[self.routing[croute.jobIdx] == jobType, croute.interval],
            dtype=int,
        )
        taskTypes = np.unique(np.array(taskDependencies))
        resourceReq = self.baseResourceTbl.loc[
            self.baseResourceTbl.index.isin(taskTypes), :
        ]
        job = Job(
            projectName=projectName,
            jobType=jobType,
            jobPrio=prio,
            nrOfSites=nrOfSites,
            taskDependencies=taskDependencies,
            intervals=intervals,
            resourceReq=resourceReq,
            planHorizon=self.planHorizon,
            croute=croute,
        )
        project.addJob(job)

    def createResource(self, col, ccost):
        resourceName = col.name
        resourceReq = col.tolist()
        resourceCost = int(
            self.baseCost.loc[
                self.baseCost.resource_idx == resourceName, ccost.unitCost
            ]
        )
        resourceCapa = int(
            self.baseCost.loc[
                self.baseCost.resource_idx == resourceName, ccost.resourceCapa
            ]
        )
        return Resource(
            name=resourceName,
            resourceReq=resourceReq,
            resourceCost=resourceCost,
            resourceCapa=resourceCapa,
            planHorizon=self.planHorizon,
        )

    def createResourceRequirement(self):
        durations = np.unique(
            np.hstack(
                (
                    np.array(self.baseResourceTbl, dtype=int).reshape(-1),
                    self.precedenceDuration,
                )
            )
        )
        durationTbl = {}
        for duration in durations:
            if duration == 0:
                continue
            tbl = np.zeros([self.planHorizon, self.planHorizon], dtype=int)
            for i in range(self.planHorizon):
                tbl[i : np.min([self.planHorizon, i + duration]), i] = 1  # noqa E203
            durationTbl[duration] = tbl
        return durationTbl

    def combineResourceTbls(self):
        resourceTbl = None
        prios = []
        for project in self.projects:
            for job in project.jobs:
                if resourceTbl is None:
                    resourceTbl = job.jobResourceTbl.copy()
                else:
                    resourceTbl = np.vstack((resourceTbl, job.jobResourceTbl))
                prios.extend(job.jobPrio)
        prios = np.array(prios) / max(prios)
        return resourceTbl, prios

    def combinePrecedenceTbls(self):
        tblPos = None
        tblNeg = None
        duration = None
        taskNames = []
        for project in self.projects:
            for job in project.jobs:
                if tblPos is None:
                    tblPos = job.precedenceTblPos.copy()
                    tblNeg = job.precedenceTblNeg.copy()
                    duration = job.precedenceDuration
                else:
                    tblPos = pd.concat(
                        [tblPos, job.precedenceTblPos], axis=0, sort=False
                    )
                    tblNeg = pd.concat(
                        [tblNeg, job.precedenceTblNeg], axis=0, sort=False
                    )
                    duration = np.hstack((duration, job.precedenceDuration))
                taskNames = taskNames + job.precedenceTblPos.columns.tolist()
        tblPos = tblPos.loc[:, taskNames]
        tblNeg = tblNeg.loc[:, taskNames]
        tblPos.fillna(0, inplace=True)
        tblNeg.fillna(0, inplace=True)
        tblPos = np.array(tblPos, dtype=int)
        tblNeg = np.array(tblNeg, dtype=int)
        return tblPos, tblNeg, duration, taskNames

    def findProject(self, projectName):
        for project in self.projects:
            if project.name == projectName:
                return project
        return None

    def findJob(self, projectName, jobType):
        project = self.findProject(projectName)
        if project is None:
            return None
        jobName = str(projectName) + "_" + str(jobType)
        for job in project.jobs:
            if job.name == jobName:
                return job
        return None

    def findResource(self, resourceName):
        for resource in self.resources:
            if resource.name == resourceName:
                return resource
        return None


class Project:
    def __init__(self, name):
        self.name = name
        self.jobs = list()

    def addJob(self, job):
        self.jobs.append(job)


class Job:
    def __init__(
        self,
        projectName,
        jobType,
        jobPrio,
        nrOfSites,
        taskDependencies,
        intervals,
        resourceReq,
        planHorizon,
        croute,
    ):
        self.name = str(projectName) + "_" + str(jobType)
        self.project = projectName
        self.jobType = jobType
        self.nrOfSites = nrOfSites
        self.taskDependencies = taskDependencies
        self.intervals = intervals
        self.resourceReq = resourceReq
        self.maxDuration = np.max(resourceReq, axis=1)
        self.timeslot_idx = list(range(planHorizon))
        self.taskNames, self.jobResourceTbl = self.createJobResourceTbl()
        (
            self.precedenceTblPos,
            self.precedenceTblNeg,
            self.precedenceDuration,
        ) = self.createTaskDependency(croute)
        self.jobPrio = np.ones_like(self.taskNames, dtype=int) * jobPrio

    def createJobResourceTbl(self):
        taskTypes = [self.name + "_" + x for x in self.resourceReq.index.tolist()]
        taskNames = []
        for i in range(self.nrOfSites):
            taskNames.extend([x + "_site" + str(i) for x in taskTypes])
        jobResourceTbl = pd.concat([self.resourceReq] * self.nrOfSites, axis=0)
        return taskNames, np.array(jobResourceTbl)

    def createTaskDependency(self, croute):
        taskTypes = self.resourceReq.index.tolist()
        dep = self.taskDependencies.copy()
        dep = dep.loc[
            dep[croute.taskIdx].isin(taskTypes)
            & dep[croute.postTaskIdx].isin(taskTypes),
            :,
        ]
        posIdx = dep[croute.taskIdx].apply(lambda x: taskTypes.index(x)).tolist()
        negIdx = dep[croute.postTaskIdx].apply(lambda x: taskTypes.index(x)).tolist()
        baseTblPos = np.zeros([len(posIdx), len(taskTypes)], dtype=int)
        baseTblNeg = np.zeros([len(posIdx), len(taskTypes)], dtype=int)
        baseDuration = []
        for i in range(len(posIdx)):
            if posIdx[i] != negIdx[i]:
                baseTblPos[i, posIdx[i]] = 1
                baseTblNeg[i, negIdx[i]] = -1
                baseDuration.append(self.maxDuration[posIdx[i]] + self.intervals[i])
        precedenceTblPos = None
        precedenceTblNeg = None
        precedenceDuration = None
        for j in range(self.nrOfSites):
            tblPos = np.hstack(
                [np.zeros_like(baseTblPos)] * j
                + [baseTblPos]
                + [np.zeros_like(baseTblPos)] * (self.nrOfSites - 1 - j)
            )
            tblNeg = np.hstack(
                [np.zeros_like(baseTblNeg)] * j
                + [baseTblNeg]
                + [np.zeros_like(baseTblNeg)] * (self.nrOfSites - 1 - j)
            )
            if precedenceTblPos is None:
                precedenceTblPos = tblPos
                precedenceTblNeg = tblNeg
                precedenceDuration = np.array(baseDuration, dtype=int)
            else:
                precedenceTblPos = np.vstack((precedenceTblPos, tblPos))
                precedenceTblNeg = np.vstack((precedenceTblNeg, tblNeg))
                precedenceDuration = np.hstack((precedenceDuration, baseDuration))

        return (
            pd.DataFrame(precedenceTblPos, columns=self.taskNames),
            pd.DataFrame(precedenceTblNeg, columns=self.taskNames),
            precedenceDuration,
        )


class Resource:
    def __init__(self, name, resourceReq, resourceCost, resourceCapa, planHorizon):
        self.name = name
        self.resourceReq = resourceReq
        self.resourceCost = resourceCost
        self.resourceCapa = resourceCapa
        self.timeslot_idx = list(range(planHorizon))
