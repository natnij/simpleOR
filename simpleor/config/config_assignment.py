"""
@author: natnij
"""
import os
import logging

#
# logging
#
LOG_LEVEL = logging.DEBUG
ENABLE_FILE_HANDLER = False

# O
# dir
#
ROOT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
)
RESOURCES_DIR = os.path.join(ROOT_DIR, "resources")
RESOURCES_INPUT_DIR = os.path.join(RESOURCES_DIR, "input", "assignment")
RESOURCES_OUTPUT_DIR = os.path.join(RESOURCES_DIR, "output", "assignment")
RESOURCES_OUTPUT_FIGURE_DIR = os.path.join(RESOURCES_DIR, "figure", "assignment")
LOGS_DIR = os.path.join(ROOT_DIR, "logs", "assignment")


class InputFiles:
    AVAILABILITY_CSV = "availability.csv"  # 需要排程的日期以及对应员工availability
    DEMAND_CSV = "demand.csv"  # 工作量需求
    PREFERENCE_CSV = "preference.csv"  # 员工对班次的喜好优先级
    REGULATION_CSV = "regulation.csv"  # 行政法规或排班要求
    SHIFT_CSV = "shift.csv"  # 公司可用班次定义


class OutputFiles:
    SCHEDULE_CSV = "schedule.csv"
    SHIFT_ASSIGN_CSV = "shiftAssignment.csv"
    FINAL_RECORD_TXT = "record_final.txt"
    RESULT_ALL_CSV = "result_all.csv"


class Specs:
    SAVELOG = True
    SIMILAR_SHIFT_HOURS = 8
    THREAD_TIMEOUT = 60
