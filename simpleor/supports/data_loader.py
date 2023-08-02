"""
@author: natnij
"""
import pandas as pd
import numpy as np
import pickle
from simpleor.utils.common_utils import CommonUtils as utcom


class DataLoader:
    def __init__(self):
        pass

    def loadExcel(self, filePath, fileName, sheet):
        path = utcom().patchFilePath([filePath, fileName])
        data = pd.read_excel(path, sheet_name=sheet, na_values=["-"])
        data.fillna(0, inplace=True)
        return data

    def loadCsv(self, filePath, fileName, idx=None, dtype=object):
        path = utcom().patchFilePath([filePath, fileName])
        try:
            data = pd.read_csv(path, dtype=dtype, index_col=idx)
        except UnicodeDecodeError:
            data = pd.read_csv(path, dtype=dtype, index_col=idx, encoding="GBK")
        data.fillna(0, inplace=True)
        return data

    def writeCsv(self, data, filePath, fileName, separator=",", outputIdx=False):
        path = utcom().patchFilePath([filePath, fileName])
        data.to_csv(path, index=outputIdx, sep=separator)

    def loadPkl(self, filePath, fileName):
        path = utcom().patchFilePath([filePath, fileName])
        infile = open(path, "rb")
        data = pickle.load(infile)
        infile.close()
        return data

    def writePkl(self, data, filePath, fileName):
        path = utcom().patchFilePath([filePath, fileName])
        outfile = open(path, "wb")
        pickle.dump(data, outfile)
        outfile.close()

    def concatData(self, originalDf, newDf, groupkey, naFill=0):
        """utility函数，数据拼接+异常处理"""
        if originalDf.shape[0] == 0:
            originalDf = newDf.copy()
            originalDf.fillna(naFill, inplace=True)
        else:
            commonCols = set(originalDf.columns) & set(newDf.columns)
            newDf = newDf.loc[:, newDf.columns.isin(commonCols)]
            cols = originalDf.columns
            originalDf = pd.concat([originalDf, newDf], axis=0, sort=True)
            originalDf = originalDf.loc[:, cols]
            originalDf.fillna(naFill, inplace=True)
            try:
                originalDf = originalDf.groupby(groupkey, as_index=False).sum()
            except:  # noqa E722
                pass
        return originalDf

    def applyFilter(self, newData, filters):
        """uitility函数，数据筛选"""
        for col in newData.columns:
            if col in filters.keys():
                for k in filters[col].keys():
                    if k == "any":
                        newData = newData.loc[newData[col].isin(filters[col][k]), :]
                    if k == "not":
                        newData = newData.loc[~newData[col].isin(filters[col][k]), :]

        return newData

    def replaceColNames(self, colnames, OriginalColumnNames, ColumnNames):
        """uitility函数，替换字段名称"""
        coln = colnames.tolist().copy()
        matchCol = self.alignColNames(
            [x for x in list(OriginalColumnNames.__dict__) if x.startswith("orig")],
            list(ColumnNames.__dict__),
        )
        for k, v in matchCol.items():
            try:
                coln[coln.index(eval("OriginalColumnNames." + k))] = eval(
                    "ColumnNames." + v
                )
            except ValueError:
                continue
        return coln

    def alignColNames(self, oldColn, newColn):
        """uitility函数，对齐字段名称"""
        matchCol = dict()
        newColn = sorted(newColn, key=len, reverse=True)
        for oldc in oldColn:
            newc = [x for x in newColn if x.lower() in oldc.lower()]
            if len(newc) > 0:
                matchCol[oldc] = newc[0]
        return matchCol

    def getValueByKey(self, dictionary, key, returnOriginal=False):
        """uitility函数，根据键值获取value + 异常处理"""
        try:
            return dictionary[int(key)]
        except:  #noqa E722
            try:
                return dictionary[str(key)]
            except:  #noqa E722
                if returnOriginal:
                    return key
                return None

    def createPivot(self, data, idx, cols, func="size", v=None, fill=0):
        """uitility函数，行变列 + 异常处理"""
        if data.shape[0] == 0:
            return pd.DataFrame()
        for col in cols:
            if not isinstance(data[col].tolist()[0], str):
                try:
                    data[col] = data[col].astype(pd.Int32Dtype())
                except:  #noqa E722
                    pass
                data[col] = data[col].astype(str)
        df = data.pivot_table(
            values=v, index=idx, columns=cols, aggfunc=func, fill_value=fill
        )
        df.sort_index(axis=0, ascending=True, inplace=True)
        df.reset_index(inplace=True)
        return df

    def getCompleteTimeseries(self, data, tsCol, step=1, fill=0, allTimeIdx=None):
        timeIdx = data[tsCol].tolist()
        if allTimeIdx is None:
            allTimeIdx = list(
                np.arange(min(timeIdx), max(timeIdx) + 1, step=step, dtype=float)
            )
        addIdx = [x for x in allTimeIdx if x not in timeIdx]
        if len(addIdx) > 0:
            df = pd.DataFrame([], columns=data.columns)
            df[tsCol] = addIdx
            data = pd.concat([data, df], axis=0)
        data.fillna(fill, inplace=True)
        return data

    def cleanseData(self, df, odict, cdict, filters):
        df.columns = self.replaceColNames(df.columns, odict, cdict)
        df = self.applyFilter(df, filters)
        return df
