"""
@author: natnij
"""
import os
import numpy as np
from datetime import datetime
from simpleor.config.config_assignment import LOGS_DIR as pathLog
from simpleor.utils.common_utils import CommonUtils as utcom


class LogUtilities:
    def openLog(filename="record.txt", titleMsg="starting"):
        path = utcom().patchFilePath([pathLog, filename])
        if os.path.exists(path):
            append_write = "a"
        else:
            append_write = "w"
        log = open(path, append_write)
        log.write("\n##### {}: {} #####\n\n".format(titleMsg, datetime.now()))
        return log

    def closeLog(log, endMsg="ending"):
        log.write("\n##### {}: {} #####\n\n".format(endMsg, datetime.now()))
        log.close()

    def printDf(log, df):
        log.write(" ".join(df.columns.tolist()) + "\n")
        a = np.array(df, dtype=int).astype(str)
        for row in a:
            log.write(" ".join(row.tolist()) + "\n")
