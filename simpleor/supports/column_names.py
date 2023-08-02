"""
@author: natnij
"""


class OriginalColumnNames_assignment:
    class OriginalDemandColumns:
        origIsHoliday = "publicHoliday"
        origWeekNr = "weeks"
        origDayOfWeek = "days"
        origHourIdx = "hours"
        origHourOfDay = "hourPerDay"
        origDemand = "demand"
        origMinCapa = "min"
        origMaxCapa = "max"

    class OriginalRegulationColumns:
        origKey = "name"
        origValue = "value"
        origUnit = "unit"
        origDescr = "description"

    class OriginalPreferenceColumns:
        origEmployeeName = "employees"
        origHourlyWage = "employeeHourlyWage"
        origHourlyOutput = "employeeHourlyOutput"
        origEmployeePrio = "employeePriority"

    class OriginalAvailColumns:
        origIsHoliday = "publicHoliday"
        origWeekNr = "weeks"
        origDayOfWeek = "days"
        origHourIdx = "hours"
        origHourOfDay = "hourPerDay"

    class OriginalCostColumns:
        origCost = "cost"


class ColumnNames_assignment:
    class DemandColumns:
        isHoliday = "publicHoliday"
        weekNr = "weeks"
        dayOfWeek = "days"
        hourIdx = "hours"
        hourOfDay = "hourPerDay"
        demand = "demand"
        minCapa = "min"
        maxCapa = "max"

        filters = {}

    class RegulationColumns:
        key = "name"
        value = "value"
        unit = "unit"
        descr = "description"

        filters = {value: {"not": [None, ""]}}

    class PreferenceColumns:
        employeeName = "employees"
        hourlyWage = "employeeHourlyWage"
        hourlyOutput = "employeeHourlyOutput"
        employeePrio = "employeePriority"

        filters = {}

    class AvailColumns:
        isHoliday = "publicHoliday"
        weekNr = "weeks"
        dayOfWeek = "days"
        hourIdx = "hours"
        hourOfDay = "hourPerDay"

        filters = {}

    class TimeTblColumns:
        onduty = "onDuty"

    class ResultOutputColumns:
        workhour = "workhour"
        nrShifts = "nrShifts"
        cost = "cost"
        preference = "pref"
        no_go_pref_fulfilled = "noGoPrefTrue"
        min_hour_btw_shifts_fulfilled = "minHourBtwShiftTrue"
        night_shift_fullfilled = "nightShiftTrue"
        continuous_shift_fullfilled = "contShiftTrue"

        all_output_cols = [
            workhour,
            nrShifts,
            cost,
            preference,
            no_go_pref_fulfilled,
            min_hour_btw_shifts_fulfilled,
            night_shift_fullfilled,
            continuous_shift_fullfilled,
        ]
        max_rule_idx_cols = [
            min_hour_btw_shifts_fulfilled,
            night_shift_fullfilled,
            continuous_shift_fullfilled,
            no_go_pref_fulfilled,
        ]
        min_rule_idx_cols = [cost]


class OriginalColumnNames_jobshop:
    class OriginalRoutingColumns:
        origJobIdx = "job_idx"
        origTaskIdx = "task_idx"
        origPostTaskIdx = "post_task_idx"
        origInterval = "interval"

    class OriginalTaskResourceReqColumns:
        origTasks = "tasks"

    class OriginalCostColumns:
        origResourceIdx = "resource_idx"
        origUnitCost = "unitCost"
        origResourceCapa = "resourceCapa"

    class OriginalHierarchyColumns:
        origProjectIdx = "project_idx"
        origJobIdx = "job_idx"
        origCount = "count"
        origJobTypePrio = "jobTypePriority"


class ColumnNames_jobshop:
    class RoutingColumns:
        jobIdx = "job_idx"
        taskIdx = "task_idx"
        postTaskIdx = "post_task_idx"
        interval = "interval"

        filters = {}

    class CostColumns:
        resourceIdx = "resource_idx"
        unitCost = "unitCost"
        resourceCapa = "resourceCapa"

        filters = {}

    class HierarchyColumns:
        projectIdx = "project_idx"
        jobIdx = "job_idx"
        count = "count"
        jobTypePrio = "jobTypePriority"

        filters = {}

    class ScoreOutputColumns:
        planHorizon = "planHorizon"
        totalDuration = "totalDurations"
        objective = "objective"
        taskPrios = "taskPrios"
        costPerPerson = "costpps"
        totalTasks = "totalTasks"
        totalResources = "totalResources"

        allCols = [
            planHorizon,
            totalDuration,
            objective,
            taskPrios,
            costPerPerson,
            totalTasks,
            totalResources,
        ]
