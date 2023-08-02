"""
@author: natnij
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, date
from itertools import product


class CommonUtils:
    def patchFilePath(self, listOfStr):
        return os.path.join(*listOfStr)

    def pdDFtoArray(self, fullTs):
        """
        check input and convert pandas dataframes to numpy arrays.
        @param fullTs: univariate timeseries in either pandas dataframe format or
            numpy array format.
        @return: always a 1d array.
        """
        if isinstance(fullTs, pd.DataFrame):
            tsArray = np.array(fullTs.iloc[:, 0])
        else:
            tsArray = np.array(fullTs)
        return tsArray

    def parseDate(self, ser, timeFormat="%Y%m%d %H:%M:%S", timeLen=19, toDate=False):
        """
        parse series into date.
        @param ser: pandas data series with date in format
        @param timeFormat: corresponding input datetime format
        @param timeLen: corresponding input date string length
        @param toDate: if True then return only year-month-day.
        @return: new list with date format. All nan's replaced with 9999-12-01 00:00
        """

        def dtConverter(x):
            if isinstance(x, (datetime, date)):
                try:
                    x = x.to_pydatetime()
                except:  # noqa E722
                    pass
                if toDate:
                    return x.date()
                else:
                    return x
            else:
                x = str(x)

            if len(x) == timeLen:
                xFormat = timeFormat
            elif len(x) == 19:
                xFormat = "%Y-%m-%d %H:%M:%S"
            elif len(x) == 16:
                xFormat = "%Y-%m-%d %H:%M"
            elif len(x) == 10:
                xFormat = "%Y-%m-%d"
            elif len(x) == 8:
                xFormat = "%Y%m%d"
            elif len(x) == 6:
                xFormat = "%Y%m"
            else:
                return date(1970, 1, 1)

            y = datetime.strptime(x, xFormat)
            if toDate:
                return y.date()
            else:
                return y

        return list(map(dtConverter, ser))

    def addCompoundKey(self, _df, _key):
        """
        turn list of key columns into one compound key with '_' as connector
        @param _df: table with original columns as named in _key
        @param _key: column names of keys
        @return: df with additional compound key column, new compound key as a string
        """
        key = "+".join(_key)
        newIndex = pd.DataFrame(
            _df.loc[:, _key].apply(lambda row: "+".join(row), axis=1), columns=[key]
        )
        df = pd.concat([newIndex, _df], axis=1)
        return df, key

    def convertToInt(self, ls):
        if isinstance(ls, list):
            if isinstance(ls[0], tuple):
                return [(int(np.ceil(float(x))), int(np.ceil(float(y)))) for x, y in ls]
            else:
                return [int(np.ceil(float(x))) for x in ls]
        else:
            return int(np.ceil(float(ls)))

    def createCombinations(self, inputlist):
        t = []
        for i in range(len(inputlist)):
            t.append(list(range(1, inputlist[i] + 1)))
        h = list(product(*t))
        return [np.array(x) for x in h]
