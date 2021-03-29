# -*- coding: UTF-8 -*-

import numpy as np
import math
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def RMSE(values_1, values_2, baseNum) -> float:
    """均方根误差"""
    if(len(values_1)==0 or len(values_2)==0 or len(values_1)!=len(values_2)):
        raise BaseException("数组无值或长度不一致")
    values1 = np.array(values_1)
    values2 = np.array(values_2)
    a = np.power((values1-values2)/baseNum, 2).sum() / len(values1)
    a = math.sqrt(a)
    return a

def MAE(values_1, values_2, baseNum) -> float:
    """平均绝对误差"""
    if (len(values_1) == 0 or len(values_2) == 0 or len(values_1) != len(values_2)):
        raise BaseException("数组无值或长度不一致")
    values1 = np.array(values_1)
    values2 = np.array(values_2)
    a = np.abs((values1 - values2) / baseNum).sum() / len(values1)
    return a

def CORR(values_1, values_2) -> float:
    """相关性系数"""
    if (len(values_1) == 0 or len(values_2) == 0 or len(values_1) != len(values_2)):
        raise BaseException("数组无值或长度不一致")
    values1 = np.array(values_1)
    values2 = np.array(values_2)
    avg1 = values1.mean()
    avg2 = values2.mean()
    fenzi = ((values1-avg1)*(values2-avg2)).sum()/len(values1)
    a = np.power(values1-avg1, 2).sum()
    b = np.power(values2-avg2, 2).sum()
    fenmu = math.sqrt(a/len(values1)*b/len(values2))
    if fenmu == 0:
        return 0
    else:
        return fenzi/fenmu
    
def HMA(P, FP) -> float:
    """调和平均数"""
    Ps = np.array(P)
    FPs = np.array(FP)
    Ps[Ps < 0] = 0
    FPs[FPs < 0] = 0
    
    d = abs(Ps - FPs).sum()
    a = abs(Ps / (Ps+FPs) - 0.5)
    c = abs(Ps - FPs) / d
    result = 1 - 2*(a*c).sum()
    return result

def PD(values_1, values_2) -> float:
    """偏差率"""
    if (len(values_1) == 0 or len(values_2) == 0 or len(values_1) != len(values_2)):
        raise BaseException("数组无值或长度不一致")
    values1 = np.array(values_1)
    values2 = np.array(values_2)
    a = np.average(np.min([np.abs(values1 - values2) / values2, [1]*len(values1)], 0))
    return a

def CalcCorr_byDate(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str) -> dict:
    """计算相关性"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")
    result = dict()
    for key, group in dateGroup:
        result[key] = CORR(group[titleP], group[titleFP])
    return result
    
def CalcMAE_byDate_exclude3(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str, baseNum:float) -> dict:
    """计算平均绝对误差（排除实发小于装机3%的数据）"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    # dataSet[titleFP] = dataSet[titleFP].apply(lambda p : p if p>=0 else 0)
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")
    result = dict()
    for key, group in dateGroup:
        group = group[group[titleP]>=baseNum*0.03]
        if len(group)==0:
            continue
        result[key] = 1-MAE(group[titleP], group[titleFP], baseNum)
    return result

def CalcRMSE_byDate(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str, baseNum:float) -> dict:
    """计算均方根误差"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    # dataSet[titleFP] = dataSet[titleFP].apply(lambda p : p if p>=0 else 0)
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")
    result = dict()
    for key, group in dateGroup:
        if len(group) == 1:
            continue
        result[key] = 1-RMSE(group[titleP], group[titleFP], baseNum)
    return result

def CalcKouDian_RMSE_byDate(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str, baseNum:float, threshold:float) -> dict:
    """计算均方根扣电百分百"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    # dataSet[titleFP] = dataSet[titleFP].apply(lambda p : p if p>=0 else 0)
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")
    result = dict()
    for key, group in dateGroup:
        if len(group) == 1:
            continue
        acc = 1-RMSE(group[titleP], group[titleFP], baseNum)
        result[key] = threshold - acc if acc < threshold else 0
    return result

def CalcKouDian_MAE_byDate(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str, baseNum:float, threshold:float) -> dict:   #P is gt, FP is predict, baseNum is cap, threshold is 0.85 for wind (MAE), 0.8 (RMSE)
    """计算平均绝对误差扣电百分百"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    # dataSet[titleFP] = dataSet[titleFP].apply(lambda p : p if p>=0 else 0)
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")
    result = dict()
    for key, group in dateGroup:
        group = group[group[titleP] >= baseNum * 0.03]
        if len(group) <= 1:
            continue
        acc = 1-MAE(group[titleP], group[titleFP], baseNum)
        result[key] = threshold - acc if acc < threshold else 0
    return result

def CalcTheroyMAE_byDate(data:pd.DataFrame, titleTime:str, titleP:str, title_Radia:str, title_Example:str, baseNum:float) -> dict:
    """计算理论功率精度（排除实发小于0）"""
    dataSet = data[[titleTime, titleP, title_Radia, title_Example]].copy()
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")
    result = dict()
    for key, group in dateGroup:
        group[titleP] = group[titleP].apply(lambda a : 0 if a<0 else a)
        
        sumP = group[titleP].sum()
        sumRadia = group[title_Radia].sum()
        sumExample = group[title_Example].sum()
        
        if sumP <= 0:
            continue
        
        accRadia = 1 - math.fabs(sumP - sumRadia)/sumP if math.fabs(sumP - sumRadia) <= sumP else 0
        accExample = 1 - math.fabs(sumP - sumExample)/sumP if math.fabs(sumP - sumExample) <= sumP else 0
        scoreRadia = 0 if accRadia >= 0.97 else ((0.97-accRadia)*100*0.05/10*baseNum if accRadia > 0 else 0.97*100*0.05/10*baseNum)
        scoreExample = 0 if accExample >= 0.97 else ((0.97-accExample)*100*0.05/10*baseNum if accExample > 0 else 0.97*100*0.05/10*baseNum)
    
        result[key] = (accRadia, accExample, scoreRadia, scoreExample)
    return result

def CalcHMA_byDate_exclude3(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str, baseNum:float) -> dict:
    """计算调和平均数准确率（排除实发小于装机3%的数据）"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    # dataSet[titleFP] = dataSet[titleFP].apply(lambda p : p if p>=0 else 0)
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")
    result = dict()
    for key, group in dateGroup:
        group = group[(group[titleP]>baseNum*0.03) | (group[titleFP]>baseNum*0.03)]
        if len(group)==0:
            continue
        result[key] = HMA(group[titleP], group[titleFP])
    return result

def CalcDIP_byDate_backup(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str, baseNum:float, threshold=0.25) -> dict:
    """计算偏差积分电量（按指定达标线计算）"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")
    result = dict()
    for key, group in dateGroup:
        Ps = np.array(group[titleP])
        FPs = np.array(group[titleFP])
        Ps[Ps < 0] = 0
        FPs[FPs < 0] = 0
        DS = abs((Ps - FPs) / FPs)
        DS[(FPs==0) & (Ps<=baseNum*0.03)] = 0
        DS[(FPs==0) & (Ps>baseNum*0.03)] = 1
        DS[(Ps==0) & (FPs<=baseNum*0.03)] = 0
        DS[(Ps==0) & (FPs>baseNum*0.03)] = 1
        if True not in (DS>threshold):
            result[key] = 0
        else:
            DP = FPs[DS>threshold]*(DS[DS>threshold]-threshold)
            result[key] = DP.sum()
    return result

def CalcDIP_byDate(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str, baseNum:float, threshold=0.25) -> dict:
    """计算偏差积分电量（按指定达标线计算），处理预测为 0 时结果为 0 的情况"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")
    result = dict()
    for key, group in dateGroup:
        Ps = np.array(group[titleP])
        FPs = np.array(group[titleFP])
        Ps[Ps < 0] = 0
        FPs[FPs < 0] = 0
        DS = abs((Ps - FPs) / FPs)
        DS[(FPs==0) & (Ps<=baseNum*0.03)] = 0
        DS[(FPs==0) & (Ps>baseNum*0.03)] = 1
        DS[(Ps==0) & (FPs<=baseNum*0.03)] = 0
        DS[(Ps==0) & (FPs>baseNum*0.03)] = 1
        overload = (DS > threshold)
        if True not in (overload):
            result[key] = 0
        else:
            group = group[overload]
            DP = group.apply(lambda row : row[titleP]-row[titleFP]*1.2 if row[titleP]>=row[titleFP] else row[titleFP]*0.8-row[titleP], axis=1)
            result[key] = DP.sum()
    return result

def CalcDIP_byDate_3(data: pd.DataFrame, titleTime: str, titleP: str, titleFP: str, baseNum: float, threshold=0.25) -> dict:
    """计算偏差积分电量（按指定达标线计算）"""
    dataSet = data[[titleTime, titleP, titleFP]].copy()
    dataSet["Date"] = dataSet[titleTime].apply(lambda t: t.date())
    dateGroup = dataSet.groupby("Date")
    result = dict()
    for key, group in dateGroup:
        group[titleP] = group[titleP].apply(lambda a: a if a >= 0 else 0)
        group[titleFP] = group[titleFP].apply(lambda a: a if a >= 0 else 0)
        Ps = np.array(group[titleP])
        FPs = np.array(group[titleFP])
        DS = abs((Ps - FPs) / FPs)
        DS[(FPs == 0) & (Ps <= baseNum * 0.03)] = 0
        DS[(FPs == 0) & (Ps > baseNum * 0.03)] = 1
        DS[(Ps == 0) & (FPs <= baseNum * 0.03)] = 0
        DS[(Ps == 0) & (FPs > baseNum * 0.03)] = 1
        overload = (DS > threshold)
        overload_1 = (DS > threshold) & ~ (((FPs == 0) & (Ps > baseNum * 0.03)) | ((Ps == 0) & (FPs > baseNum * 0.03)))
        overload_2 = (DS > threshold) & (((FPs == 0) & (Ps > baseNum * 0.03)) | ((Ps == 0) & (FPs > baseNum * 0.03)))
        if True not in (overload):
            result[key] = 0
        else:
            group_1 = group[overload_1]
            group_2 = group[overload_2]
            DP_1 = group_1.apply(lambda row: row["P"] - row["F_P"] * 1.2 if row["P"] >= row["F_P"] else row["F_P"] * 0.8 - row["P"],axis=1)
            DP_2 = group_2.apply(lambda row: row["P"] - row["F_P"] * 1.0 if row["P"] >= row["F_P"] else row["F_P"] * 1.0 - row["P"],axis=1)
            result[key] = DP_1.sum() if len(DP_1)>0 else 0 + DP_2.sum() if len(DP_2)>0 else 0
    return result

def CalcPPR_byDate(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str, baseNum:float, threshold=0.8) -> dict:
    """计算合格率"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")
    result = dict()
    for key, group in dateGroup:
        if len(group)==1:
            continue
            
        Ps = np.array(group[titleP])
        FPs = np.array(group[titleFP])
        Ps[Ps < 0] = 0
        FPs[FPs < 0] = 0

        DS = abs((Ps - FPs) / baseNum)
        pp = DS[DS<=(1-threshold)]
        ppRatio = len(pp) / len(group)
        result[key] = ppRatio
    return result

def CalcRMSE_byMonth(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str, baseNum:float) -> dict:
    """按整月计算均方根精度"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    # dataSet[titleFP] = dataSet[titleFP].apply(lambda p : p if p>=0 else 0)
    dataSet["Month"] = dataSet[titleTime].apply(lambda t : t.strftime("%Y-%m"))
    dateGroup = dataSet.groupby("Month")
    result = dict()
    for key, group in dateGroup:
        if len(group) == 1:
            continue
        result[key] = 1-RMSE(group[titleP], group[titleFP], baseNum)
    return result

def CalcCORR_byDate(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str) -> dict:
    """计算日相关性"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")
    result = dict()
    for key, group in dateGroup:
        result[key] = CORR(group[titleP], group[titleFP])
    return result

def CalcDIP_Powertrading(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str, baseNum:float) -> (dict, pd.DataFrame):
    """计算偏差积分电量（甘肃电力交易）"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")
    result = dict()
    dipFrame=pd.DataFrame()
    for key, group in dateGroup:
        Ps = np.array(group[titleP])
        FPs = np.array(group[titleFP])
        Ps[Ps < 0] = 0
        FPs[FPs < 0] = 0
        
        # FPs[(FPs*0.02 < 2)] = 2
        # FPs[(FPs*0.02 >= 2)&(FPs*0.05 < 5)] = 5
        # FPs[(FPs*0.05 >= 5)&(FPs*0.1 < 10)] = 10
        # FPs[(FPs*0.1 >= 10)&(FPs*0.2 < 20)] = 20
        
        # FPs[(FPs*0.02 < 2)] = 2
        # FPs[FPs*0.05 < 5] = 5
        # FPs[FPs*0.1 < 10] = 10
        # FPs[FPs*0.2 < 20] = 20
        
        dips = Ps - FPs
        
        coff = np.zeros(len(dips))
        
        coff[(-0.1*FPs<=dips)&(dips<=0.02*FPs)] = 0.0
        coff[((-0.2*FPs<=dips)&(dips<-0.1*FPs))|((0.02*FPs<dips)&(dips<=0.05*FPs))] = 1.0
        coff[(dips<-0.2*FPs)|(0.05*FPs<dips)] = 3.0
        
        dips_after = np.abs(dips*coff)
        
        result[key] = dips_after.sum()

        group["DIP"] = np.abs(dips)
        group["DIP_After"] = dips_after
        dipFrame = dipFrame.append(group.copy(),ignore_index=True)
        
    return result, dipFrame

def CalcDIP_Powertrading2(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str, baseNum:float) -> (dict, pd.DataFrame):
    """计算预测总考核（甘肃电力交易），把预测当作计划值，所有点均考核"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")

    piancha_down = 0.15
    piancha_up = 0.05
    
    result = dict()
    dipFrame=pd.DataFrame()
    for key, group in dateGroup:
    
        group["Error"] = (group[titleP]/group[titleFP]).replace(np.inf,1) -1
        group["ErrorPower"] = group.apply(lambda x: x[titleFP]*(x["Error"] - piancha_up) if x["Error"] >= piancha_up \
            else (x[titleFP] * (x["Error"]+piancha_down) if x["Error"] <= 0 - piancha_down else 0), axis=1)
        group.loc[((group[titleFP] * piancha_up < 5) & (group["Error"] > piancha_up)), "ErrorPower"] = 0
        group.loc[((group[titleFP] * piancha_down < 5) & (group["Error"] < 0 - piancha_down)), "ErrorPower"] = 0

        group["ErrorPower"] = abs(group["ErrorPower"])

        
        result[key] = group["ErrorPower"].sum()
        
        dipFrame = dipFrame.append(group.copy(),ignore_index=True)
    
    return result, dipFrame

def CalcDIP_Powertrading4(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str, baseNum:float) -> (dict, pd.DataFrame):
    """计算预测总考核（甘肃电力交易），把预测当作计划值，所有点均考核"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dateGroup = dataSet.groupby("Date")
    
    piancha_down = 0.15
    piancha_up = 0.05
    
    result = dict()
    dipFrame=pd.DataFrame()
    for key, group in dateGroup:
        
        group["Error"] = (group[titleP]/group[titleFP]).replace(np.inf,1) -1
        group["ErrorPower"] = group.apply(lambda x: x[titleFP]*(x["Error"] - piancha_up) if x["Error"] >= piancha_up \
            else (x[titleFP] * (x["Error"]+piancha_down) if x["Error"] <= 0 - piancha_down else 0), axis=1)
        index_up = group[((group[titleFP] * piancha_up < 5) & (group["Error"] > piancha_up))].index.tolist()
        group.loc[index_up, "ErrorPower"] = group.loc[index_up, titleP] - group.loc[index_up, titleFP] - 5
        group.loc[index_up, "ErrorPower"] = group.loc[index_up, "ErrorPower"].apply(lambda x: 0 if x<0 else x)
        index_down = group[((group[titleFP] * piancha_down < 5) & (group["Error"] < 0 - piancha_down))].index.tolist()
        group.loc[index_down, "ErrorPower"] = group.loc[index_down, titleP] - group.loc[index_down, titleFP] + 5
        group.loc[index_down, "ErrorPower"] = group.loc[index_down, "ErrorPower"].apply(lambda x: 0 if x>0 else x)
        
        group["ErrorPower"] = abs(group["ErrorPower"])
        
        
        result[key] = group["ErrorPower"].sum()
        
        dipFrame = dipFrame.append(group.copy(),ignore_index=True)
    
    return result, dipFrame


def CalcPD_byDate_exclude3(data:pd.DataFrame, titleTime:str, titleP:str, titleFP:str, baseNum:float) -> dict:
    """偏差率    |预测-实发|/实发"""
    dataSet = data[[titleTime,titleP,titleFP]].copy()
    dataSet["Date"] = dataSet[titleTime].apply(lambda t : t.date())
    dataSet = dataSet[dataSet[titleP]>=baseNum*0.03]
    result = dict()
    for key, group in dataSet.groupby("Date"):
        result[key] = PD(group[titleFP], group[titleP])
    return result



#if __name__=="__main__":
#    import numpy as np
#    import pandas as pd
#    from com.common.utils.FileUtils import ReadXLSX, ReadCSV
#    
#    df = ReadCSV(r"D:\数据整理\气象数据\QHGDTNMH.csv", timecolumns=["Time"], timeformats=["%Y-%m-%d %H:%M:%S"])
#    # df = ReadCSV(r"D:\FTP\1.csv", timecolumns=["Time"], timeformats=["%Y-%m-%d %H:%M:%S"])
#    df = df[["Time", "P", "F_P"]]
#    df.dropna(axis=0, how="any", inplace=True)
#    accFP = CalcDIP_byDate(df, "Time", "P", "F_P", 49.5, 0.25)
#    
#    for key in accFP:
#        print(key, accFP.get(key))
