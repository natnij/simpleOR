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

#
# dir
#
ROOT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
)
LOGS_DIR = os.path.join(ROOT_DIR, "logs", "jobshopScheduling")
RESOURCES_DIR = os.path.join(ROOT_DIR, "resources")
RESOURCES_INPUT_DIR = os.path.join(RESOURCES_DIR, "input", "jobshopScheduling")
RESOURCES_OUTPUT_DIR = os.path.join(RESOURCES_DIR, "output", "jobshopScheduling")
RESOURCES_OUTPUT_FIGURE_DIR = os.path.join(RESOURCES_DIR, "figure", "jobshopScheduling")


class InputFiles:
    COST_CSV = "cost.csv"  # 资源成本
    HIERARCHY_CSV = "hierarchy.csv"  # 项目和任务的从属关系
    RESOURCE_CSV = "resource.csv"  # 资源信息
    ROUTING_CSV = "routing.csv"  # 任务的先后依赖关系


class OutputFiles:
    START_TIME_CSV = "startTime.csv"
    SCORE_CSV = "score.csv"
    DURATION_CSV = "actualDuration.csv"
    RESOURCE_USED_CSV = "resourceUsed.csv"


class Specs:
    PLAN_HORIZON_START = 30  # START / END / STEP: 测试给定不同计划时间窗得出的不同优化结果
    PLAN_HORIZON_END = 60
    PLAN_HORIZON_STEP = 10
    # 'earliest'：任务尽量早排，'latest'：任务尽量晚排，'leastCost'：任务安排尽量降低总成本
    PLAN_STRATEGY = "earliest"
    MULTIPLIER = 2  # 优化前整体可行性分析时设置的宽松resource上限倍数：
    # planHorizon内的可用资源总量在优化前需要达到需求资源总量的muliplier倍，提高优化效率
    # 如果不能达到，可以通过增加planHorizon
    THREAD_TIMEOUT = 120
    SAVELOG = True
