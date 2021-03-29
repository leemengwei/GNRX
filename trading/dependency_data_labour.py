import pandas as pd
from IPython import embed
import sys,os
import glob
import datetime
import numpy as np
import config
import copy
import dependency_misc

def get_duration(files):
    start = '2020'+files[0].split('/')[-1][:4]
    end = '2020'+files[-1].split('/')[-1][:4]
    end = (datetime.datetime.strptime(end, '%Y%m%d') + datetime.timedelta(1)).strftime('%Y%m%d') 
    time_index = pd.date_range(start, end, freq='15min')
    return time_index[1:]   # all data start with 00:15 end with 00:00

def check_in_advance(code, types_of_data):
    lens = []
    for type_of_data in types_of_data:
        files = glob.glob("../data/%s/%s/*.xlsx"%(code, type_of_data))[::-1]
        files.sort()
        lens.append(len(files))
    if np.mean(lens)!=lens[0]:
        print(lens)
        print('Check folder length inconsistent, continue?')
        input()

def step1_rearrange_all_in_one(req_json):
    '''
    Loop read all files in each 0X-XXXX/dates-xxx.csv and generate a put-together.csv named 01-05(for convinience of later mannual copy-paste)
    Mannual copy-paste to all and copy 07 data into it, save it as all.csv for later use.
    '''
    code = req_json.get('farm_code')
 
    types_of_data = ["01-统调负荷预测及实测","02-东送计划预测及实测","03-1-全网风电出力预测及实测","03-2-全网光伏出力预测及实测","03-新能源出力预测及实测","04-正负备用容量","06-日分解及出清结果"]   # "05-检修信息",
    check_in_advance(code, types_of_data)
    all_in_one = []
    for type_of_data in types_of_data:
        files = glob.glob("../data/%s/%s/*.xlsx"%(code, type_of_data))[::-1]
        files.sort()
        duration = get_duration(files)
        datas = pd.DataFrame() 
        for i in files: 
            print(type_of_data, i)
            data = pd.read_excel(i) 
            datas = pd.concat([datas, data]) 
        try:
            datas.index = duration
        except:
            embed()
        token = type_of_data.split('-')[-1][:4]
        datas.columns = '%s-'%token + data.columns
        datas.to_csv('../data/%s/%s/put_together.csv'%(code, type_of_data)) 
        all_in_one.append(datas)

    all_in_one = pd.concat(all_in_one, axis=1)
    all_in_one.to_csv('../data/%s/01-05.csv'%code)
    return

def step2_read_all_in_one_data(req_json):
    '''
    After run step1, mannual paste 07 to csv, paste trade clean data, paste short-term data all in one file. Read it then. 
    This file should be directly given from C in practice.
    '''
    data = pd.read_csv(req_json['supply_path'], index_col=0, parse_dates=True)
    for col in data.columns: 
        if '-序号' in col or '-时刻' in col or '-发电单元名称' in col or '日分解及-日期' in col:  #these are duplicate columns
            data = data.drop(col, axis=1) 
    data = data.astype(float)
    data = data[dependency_misc.data_rename_dict]
    data = data.rename(columns = dependency_misc.data_rename_dict)
    return data

def mengxi_raw_data_prepare(req_json):
    # rearrange 01-06:
    #step1_rearrange_all_in_one(req_json)
    # should then mannual copy-paste 07 & [并网,辐照] then continue:
    data = step2_read_all_in_one_data(req_json)
    return data

def more_shift_features(data, mode):
    # TODO: features like daytime?
    # make shift over data:
    tmp = copy.deepcopy(data)
    for shift_step in range(1, config.feature_shift_steps+1):
        print('Shifting feature', shift_step)
        shifted_data = tmp.shift(shift_step)
        shifted_data.columns = tmp.columns+'_shift_%s'%shift_step
        data = pd.merge(data, shifted_data, left_on=data.index, right_on=shifted_data.index, left_index=True, right_index=True)
    # make shift over column names: (for later use)
    input_features = config.predictive_model_dicts[mode]['in']
    output_features = config.predictive_model_dicts[mode]['out']
    shift_names = []
    for shift_step in range(1, config.feature_shift_steps+1):
        for name in input_features:
            shift_names.append('%s_shift_%s'%(name, shift_step))
    input_features = input_features + shift_names
    return data, input_features, output_features

def split_dataset(dataset):
    train_date = dataset.resample('D').mean().index[:-config.val_days].strftime('%Y%m%d').to_list()
    val_date = dataset.resample('D').mean().index[-config.val_days:].strftime('%Y%m%d').to_list()
    train_dataset = pd.DataFrame(columns=dataset.columns)
    val_dataset = pd.DataFrame(columns=dataset.columns)
    for i in train_date: 
        train_dataset = train_dataset.append(dataset[i])
    for i in val_date: 
        val_dataset = val_dataset.append(dataset[i])
    return train_dataset, val_dataset

#if __name__ == '__main__':
#    print('start')
#    step1_rearrange_all_in_one()
#    step2_read_mannual_copy_paste_data()
#    print('done')
#    print(step1_rearrange_all_in_one.__doc__)

