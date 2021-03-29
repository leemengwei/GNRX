import pandas as pd
#from IPython import embed
import config
import numpy as np
import scipy
import scipy.interpolate as spi
import sys, os
import sklearn
import warnings
warnings.filterwarnings('ignore')


class Recorder():
    def __init__(self):
        self.data = {}
    def get_mean(self, name):
        try:
            tmp = np.nanmean(self.data[name])
        except:
            return 'Nop'
        return tmp
    def get_keys(self):
        return self.data.keys()
    def get_all_data(self):
        return self.data
    def get_data(self, name):
        return np.array(self.data[name])
    def add_one(self, name, value):
        if name not in self.data.keys():
            self.data[name] = [value]
        else:
            self.data[name].append(value)

def evaluation(predicted, label, cap):
    threshold = config.low_value_region_threshold
    #eval matrices:
    MSE = sklearn.metrics.mean_squared_error(label, predicted)
    RMSE_PEER_CAP = np.sqrt(MSE)
    MAE = sklearn.metrics.mean_absolute_error(label, predicted)
    R2 = sklearn.metrics.r2_score(label, predicted)
    PEARSON = np.corrcoef(predicted, label)[0, 1]
    RELATIVE_ERROR = np.abs((predicted[np.where(label>cap*threshold)[0]] - label[np.where(label>cap*threshold)[0]])/label[np.where(label>cap*threshold)[0]]).mean()
    RELATIVE_ERROR_misc = np.abs((predicted[np.where(label<cap*threshold)[0]] - label[np.where(label<cap*threshold)[0]])/label[np.where(label<cap*threshold)[0]]).mean()
    RELATIVE_ERROR_all = np.abs((predicted - label)/label).mean()
    return RMSE_PEER_CAP, MAE, R2, PEARSON, RELATIVE_ERROR, RELATIVE_ERROR_misc, RELATIVE_ERROR_all

def basic_data_clean(data_raw, cap, use_by_wind=True):
    epsilon = 1e-7
    data_raw = data_raw.loc[data_raw.loc[:, 'power_true'].dropna().index]
    data_raw = data_raw.fillna(0)
    data_raw = data_raw[data_raw.power_true.between(-0.5, 5000)]   #power within 5000mwa
    data_raw = data_raw[data_raw.power_true.between(np.percentile(data_raw.power_true, 0.1), np.percentile(data_raw.power_true, 99.9)+epsilon)]
    data_raw = data_raw[data_raw.obs_true.between(0, 100 if use_by_wind else 2500)]  #wind<100, radiation<2500
    data_raw = data_raw[data_raw.obs_true.between(np.percentile(data_raw.obs_true, 0.1)-epsilon, np.percentile(data_raw.obs_true, 99.9)+epsilon)]

    #replacing \\N
    data_raw.loc[:, 'power_true'] = data_raw.loc[:, 'power_true'].where(data_raw.loc[:, 'power_true']!='\\N', 0).values.astype(float).reshape(-1)  #There're some \\N in data, replace them
    data_raw.loc[:, 'obs_true'] = data_raw.loc[:, 'obs_true'].where(data_raw.loc[:, 'obs_true']!='\\N', 0).values.astype(float).reshape(-1)

    #cleaning power, out of range and dead
    data_raw = data_raw[data_raw.power_true.between(-1, cap*1.2)]  #only keep those power -1 < where < 1.2*cap 
    where_undead_power = np.append(data_raw.power_true.values[:-1] - data_raw.power_true.values[1:] != 0, False)
    data_cleaned = data_raw[where_undead_power]  #clean power that dead
    return data_cleaned

def restore_restricted_data(data_cleaned, cap, drop=True, use_by_wind=True):
    data_cleaned = data_cleaned.loc[:, ['obs_true', 'power_true']].copy()

    #cleanning meteor, out of range and dead
    data_cleaned = data_cleaned[data_cleaned.obs_true.between(0, 100 if use_by_wind else 2500)]  #speed must not exceeding 100 (they're errors!), max radiantion must not greater than 2500
    where_undead_meteor = np.append(data_cleaned.obs_true.values[:-1] - data_cleaned.obs_true.values[1:] != 0, False)
    data_cleaned = data_cleaned[where_undead_meteor]  #clean meteor that dead

    #restoring:   
    number_of_seg = max(int(len(data_cleaned)/400), 20)
    segs = np.linspace(min(data_cleaned.obs_true), max(data_cleaned.obs_true), number_of_seg)
    restored_data = pd.DataFrame()
    last_should_have_power = -9999
    for idx in range(len(segs)-1):
        low = segs[idx]
        high = segs[idx+1]
        power_delta = data_cleaned.power_true.mean()/20
        data_seg = data_cleaned[data_cleaned.obs_true.between(low, high)]
        if len(data_seg) < 10:continue
        should_have_power = np.percentile(data_seg.power_true, 85)
        if low>np.mean(segs):
            if should_have_power < last_should_have_power:
                should_have_power = last_should_have_power
        last_should_have_power = should_have_power

        below_points_index = data_seg.power_true.between(-np.inf, should_have_power-data_cleaned.power_true.mean())
        above_points_index = data_seg.power_true.between(should_have_power+data_cleaned.power_true.mean(), np.inf)
        if drop:
            data_seg = data_seg[~below_points_index]
            data_seg = data_seg[~above_points_index]
        else:
            data_seg.power_true[below_points_index] = np.clip(np.random.normal(should_have_power, power_delta, len(below_points_index)), 0, cap)
            data_seg.power_true[above_points_index] = np.clip(np.random.normal(should_have_power, power_delta, len(above_points_index)), 0, cap)
        restored_data = restored_data.append(data_seg)
    restored_data = restored_data.sort_index()
    return restored_data


#clean for power is complicated.    
def gt_data_clean_for_power_restriction(df, cap):  
    #NOTE: data cleaning must be done on its original time scale, e.g 15mins, rather than after something like resample(D)
    meteor_name = 'obs_true'
    power_name = 'power_true'
    df = df.loc[:, [power_name, meteor_name]].dropna()
    where_undead = np.append(df[meteor_name][:-1].values - df[meteor_name][1:].values != 0, False)
    df = df[where_undead]
    where_undead = np.append(df[power_name][:-1].values - df[power_name][1:].values != 0, False)
    df = df[where_undead]

    #First keep a specific part where power <0, and clip it.
    df_power_lt_zero_part_and_speed_lt_cut_in_part = df[((df.loc[:, power_name]<0).values+0 + (df.loc[:, meteor_name] < config._cut_in_wind_speed).values+0)==2].copy()
    df_power_lt_zero_part_and_speed_lt_cut_in_part.loc[:, power_name] = np.clip(df_power_lt_zero_part_and_speed_lt_cut_in_part.loc[:, power_name], -2, np.inf)
    
    df = df[df.loc[:, meteor_name]> config._cut_in_wind_speed]   #Where gt-wind less than 2, just discard them 
    df = df[df.loc[:, power_name]!=0]  #Where gt-power equals 0, just discard them.
    df_returned = df.copy()
 
    #stagewise polyfit cleaning model.
    wind_speed_intervals = np.linspace(min(df.loc[:, meteor_name]), max(df.loc[:, meteor_name]), config._num_of_interval)
    data_interval = pd.DataFrame(data={'wind_speed_intervals': wind_speed_intervals, 'power_interval': 0})
    last_power = -9999
    for index, i in enumerate(range(len(wind_speed_intervals)-1)):
        lower_speed = wind_speed_intervals[i]
        upper_speed = wind_speed_intervals[i+1]
        df_in_speed_range = df[df.loc[:, meteor_name].between(lower_speed, upper_speed)]
        if len(df_in_speed_range) == 0:
            data_interval.loc[i, 'power_interval'] = last_power
        else:
            power_interval = np.percentile(df_in_speed_range.loc[:, power_name], config.percentile)
            if power_interval > last_power:    #my rule: larger wind speed should correspond to larger power
                data_interval.loc[i, 'power_interval'] = power_interval
                last_power = power_interval
            else:
                if lower_speed > max(df.loc[:, meteor_name])/2:  #just a work around: when wind speed is greater than some value, my rule must be satisfied.
                    data_interval.loc[i, 'power_interval'] = last_power
                else:
                    data_interval.loc[i, 'power_interval'] = power_interval
                    last_power = power_interval
    data_interval = data_interval.iloc[:-1, :]
    model_cleaning = scipy.interpolate.interp1d(data_interval.wind_speed_intervals, data_interval.power_interval, kind='slinear', fill_value='extrapolate')

    #Discard where power below %percentile%, they are seen as 'limited'
    df_returned['threshold_curve'] = model_cleaning(df_returned.loc[:, meteor_name])
    df_returned = df_returned[df_returned.loc[:,power_name]>df_returned.loc[:,'threshold_curve']] 
    #df_returned = df_returned.append(pd.DataFrame(data={power_name:np.random.normal(cap,0.3,999), meteor_name: 22.5+100*np.random.rand(999)}))
    return df_returned

def gt_data_clean_simple(df, col_name):     #clean for meteor (only meteor) is relatively simple.   #NOTE: data cleaning must be done on its original time scale, e.g 15mins, rather than after something like resample(D)
    df = df.loc[:, ['power_true', 'obs_true']]
    df = df.dropna(subset=[col_name])
    df = df[df.loc[:, col_name].between(np.percentile(df.loc[:, col_name], 1), np.percentile(df.loc[:, col_name], 99))]
    where_undead = np.append(df['power_true'][:-1].values - df['power_true'][1:].values != 0, False)
    df = df[where_undead]
    return df

def read_15min_gt_and_clean_and_convert_to_24h(code, how): 
    true_df = pd.read_csv(os.path.join(config.PROJECT_NAME, 'data', 'true', code+'.csv'), index_col=0, parse_dates=True)  #15-mins instantaneous
    if how == 'handle_restriction':
        true_df = gt_data_clean_for_power_restriction(true_df)
    elif how == 'meteor_simple':
        true_df = gt_data_clean_simple(true_df, col_name='obs_true')
    elif how == 'power_simple':
        true_df = gt_data_clean_simple(true_df, col_name='power_true')
    else:
        print("Error", how)
        sys.exit()
    true_df_24 = true_df.resample('D').mean().dropna()
    return true_df_24, true_df

def read_6hour_meteor_and_convert_to_24h(filename):
    this_meteors = pd.read_csv(open(filename), index_col=0, parse_dates=True)   #6-hourly instantaneous by WuYuan  --> sum()/4 to get daily average
    this_meteors_24h = this_meteors.resample('D').mean().dropna()
    return this_meteors_24h, this_meteors

def generate_distribution(input_series, cap, workable_hours, ratio=0.95):
    if not isinstance(input_series, pd.DataFrame):
        input_series = pd.DataFrame(input_series)
    distribution = pd.DataFrame(data={'100': input_series.values.reshape(-1), \
		'80_lower': input_series.values.reshape(-1)*[np.random.randint(70,90) for i in range(len(input_series))]/100, \
		'80_upper': input_series.values.reshape(-1)*[np.random.randint(110,130) for i in range(len(input_series))]/100, \
		'60_lower': input_series.values.reshape(-1)*[np.random.randint(50,70) for i in range(len(input_series))]/100, \
		'60_upper': input_series.values.reshape(-1)*[np.random.randint(130,150) for i in range(len(input_series))]/100})
    #postprocess:
    output_distribution = np.clip(distribution, 0, cap*workable_hours*ratio)
    output_distribution.index = input_series.index
    return output_distribution 




