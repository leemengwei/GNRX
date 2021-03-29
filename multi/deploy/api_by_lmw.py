#2020-11-17 by leemengwei
'''
WangDeMin use multiseg model to do stagewise linear polyfit of power.
'''
import glob
import sys, os
import pandas as pd
import time
import numpy as np
import datetime
import copy
import warnings
warnings.filterwarnings('ignore')
#from IPython import embed

class PrintHandler():
    def __init__(self, to_file=''):
        self.original_stdout = sys.stdout
        if to_file != '':
            self.to_file = to_file
        else:
            self.to_file = os.devnull
    def open(self):
        local_stdout = self.original_stdout
        return local_stdout
    def close(self): 
        local_stdout = open(self.to_file, 'w')
        return local_stdout

class ModelWeightedSum():
    def __init__(self, json):
        self.name = 'model weighted sum'
        self.num_of_cols_in_with_power_file = 9   # -1 for time index column
        self.power_column_in_sources_with_power = -3
        self.wind_column_in_sources_with_power = 0
        self.wind_column_in_sources_without_power = 0
        self.humidity_column_in_sources_with_power = -5
        self.humidity_column_in_sources_without_power = -2
        self.json = json
        self.jobs = self.json['plants']
        self.output_power = None
        self.logger = []
        self.ec_data = None
        self.forecast_time = datetime.datetime.strptime(self.json['fcstdt'][:10], '%Y-%m-%d')   #time when action triggered, WPD from this day
        self.prediction_start = self.forecast_time + datetime.timedelta(days=1)   # prediction starts at next day 
        self.template_columns = ['##id', 'Timestamp', 'Power', 'Windspeed', 'WindDir', 'Temperature', 'Humidity', 'Pressure']
        
    def parse_weights(self, weights_file):
        #read in weight config file:
        self.weights_content = pd.read_csv(weights_file, index_col=0)['weights'].dropna()
        # Misc configuration:
        self.weights_content = self.weights_content.astype(float)
        self.lowest_power_threshold = self.weights_content.loc['misc_lowest_power_threshold']
        self.set_lowest_to = self.weights_content.loc['misc_set_lowest_to']
        self.from_which_to_get_all_others = self.weights_content.loc['misc_from_which_to_get_all_others']
        self.rand_k = 1
        self.day_2_factor_for_day_5 = self.weights_content.loc['day_2_factor_for_day_5']
        self.day_3_factor_for_day_5 = self.weights_content.loc['day_3_factor_for_day_5']
        self.day_4_factor_for_day_5 = self.weights_content.loc['day_4_factor_for_day_5']
        self.day_2_factor_for_day_6 = self.weights_content.loc['day_2_factor_for_day_6'] 
        self.day_4_factor_for_day_6 = self.weights_content.loc['day_4_factor_for_day_6'] 
        self.day_5_factor_for_day_6 = self.weights_content.loc['day_5_factor_for_day_6'] 
        self.day_3_factor_for_day_7 = self.weights_content.loc['day_3_factor_for_day_7']  
        self.day_4_factor_for_day_7 = self.weights_content.loc['day_4_factor_for_day_7']  
        self.day_5_factor_for_day_7 = self.weights_content.loc['day_5_factor_for_day_7']  
        self.day_6_factor_for_day_7 = self.weights_content.loc['day_6_factor_for_day_7']  
        self.day_5_factor_for_day_8 = self.weights_content.loc['day_5_factor_for_day_8']  
        self.day_6_factor_for_day_8 = self.weights_content.loc['day_6_factor_for_day_8']  
        self.day_7_factor_for_day_8 = self.weights_content.loc['day_7_factor_for_day_8']  
        self.day_5_factor_for_day_9 = self.weights_content.loc['day_5_factor_for_day_9']  
        self.day_7_factor_for_day_9 = self.weights_content.loc['day_7_factor_for_day_9']  
        self.day_8_factor_for_day_9 = self.weights_content.loc['day_8_factor_for_day_9']  
        self.day_6_factor_for_day_10 = self.weights_content.loc['day_6_factor_for_day_10']
        self.day_7_factor_for_day_10 = self.weights_content.loc['day_7_factor_for_day_10']
        self.day_8_factor_for_day_10 = self.weights_content.loc['day_8_factor_for_day_10']
        self.day_9_factor_for_day_10 = self.weights_content.loc['day_9_factor_for_day_10']
        self.day_1_factor_for_day_11 = self.weights_content.loc['day_1_factor_for_day_11']
        self.day_3_factor_for_day_11 = self.weights_content.loc['day_3_factor_for_day_11']
        self.day_4_factor_for_day_11 = self.weights_content.loc['day_4_factor_for_day_11']

        power_keep_index = [] 
        speed_keep_index = [] 
        for i in self.weights_content.index:
            if i.startswith('power_'): 
                power_keep_index.append(i)
        for i in self.weights_content.index:
            if i.startswith('speed_'): 
                speed_keep_index.append(i)
        self.power_weights = self.weights_content.loc[power_keep_index]
        self.speed_weights = self.weights_content.loc[speed_keep_index]

    def prepare_a_job(self, idx):
        self.job = self.jobs[idx]
        self.station_code = self.job['code']
        self.prediction_end = self.prediction_start + datetime.timedelta(self.job['daynum'])
        self.prediction_period = pd.date_range(self.prediction_start, self.prediction_end, freq='15min', closed='right')
        self.parse_weights(weights_file = self.job['modeldir']+"/weights.csv")
        self.cap = self.job['runcap']

    def read_in_sources_data(self):
        # read in data of meteor of ALL sources you can (decide later which to use)
        self.platform_prediction_of_different_sources = {}
        self.platform_meteor_of_different_sources = {}
        self.platform_power_of_different_sources = {}
        self.powers = {}
        self.windspeeds = {}
        self.meteors = {}
        self.bakqxlist = []
        self.use_bak_flag = 0
        for this_qixiang in self.job['qxlist']:
            qx_id = this_qixiang['qxid']
            qx_code = this_qixiang['qxcode']
            qx_dir = this_qixiang['qxdir']
            qx_files = glob.glob("%s/*.WPD"%(qx_dir))
            if len(qx_files)==0:
                info = "*Warning, no WPD file found in %s"%qx_dir
                self.handle_exception(info)
                continue
            # read powers
            self.powers['power_%s'%str(qx_id)], self.meteors[str(qx_id)], useBakQx, bakqx = self._read_WPD(qx_dir, qx_code, read_its_power=True)
            # and read windspeeds
            self.windspeeds['speed_%s'%str(qx_id)], self.meteors[str(qx_id)], useBakQx, bakqx = self._read_WPD(qx_dir, qx_code, read_its_power=False)
            if useBakQx == None:continue
            if useBakQx:
                bakqx_dict = {'qxID': qx_id, 'mainQx': '', 'useBakQx':bakqx}
                self.bakqxlist.append(bakqx_dict)
                self.use_bak_flag = 1
        self.powers = pd.DataFrame(self.powers)
        self.windspeeds = pd.DataFrame(self.windspeeds)
        # drop columns with all nan
        self.powers = self.powers.drop(columns = self.powers.columns[np.where(self.powers.count() == 0)[0]].tolist())
        self.windspeeds = self.windspeeds.drop(columns = self.windspeeds.columns[np.where(self.windspeeds.count() == 0)[0]].tolist())

    def append_bias_columns(self):
        # append additional column for bias:
        self.powers['power_bias'] = np.tile(1, self.powers.shape[0])
        self.windspeeds['speed_bias'] = np.tile(1, self.windspeeds.shape[0])
        self.powers = self.powers.fillna(self.powers.mean()) 
        self.windspeeds = self.windspeeds.fillna(self.windspeeds.mean()) 

    def extend_over_time(self):
        # get all others like humidity, wind direction etc, extend them.
        self.ec_data = self.meteors['%s'%int(self.from_which_to_get_all_others)] 
        # extend them over 11-day:
        self.powers = self._method_for_extension(self.powers)
        self.ec_data = self._method_for_extension(self.ec_data)

    def _method_for_extension(self, data):
        #follow doc description to extend missing days:
        day_1 = pd.date_range(data.index[0] + datetime.timedelta(0), data.index[0] + datetime.timedelta(1), freq='15min', closed='left')
        day_2 = pd.date_range(data.index[0] + datetime.timedelta(1), data.index[0] + datetime.timedelta(2), freq='15min', closed='left')
        day_3 = pd.date_range(data.index[0] + datetime.timedelta(2), data.index[0] + datetime.timedelta(3), freq='15min', closed='left')
        day_4 = pd.date_range(data.index[0] + datetime.timedelta(3), data.index[0] + datetime.timedelta(4), freq='15min', closed='left')
        day_5 = pd.date_range(data.index[0] + datetime.timedelta(4), data.index[0] + datetime.timedelta(5), freq='15min', closed='left')
        day_6 = pd.date_range(data.index[0] + datetime.timedelta(5), data.index[0] + datetime.timedelta(6), freq='15min', closed='left')
        day_7 = pd.date_range(data.index[0] + datetime.timedelta(6), data.index[0] + datetime.timedelta(7), freq='15min', closed='left')
        day_8 = pd.date_range(data.index[0] + datetime.timedelta(7), data.index[0] + datetime.timedelta(8), freq='15min', closed='left')
        day_9 = pd.date_range(data.index[0] + datetime.timedelta(8), data.index[0] + datetime.timedelta(9), freq='15min', closed='left')
        day_10 = pd.date_range(data.index[0] + datetime.timedelta(9), data.index[0] + datetime.timedelta(10), freq='15min', closed='left')
        day_11 = pd.date_range(data.index[0] + datetime.timedelta(10), data.index[0] + datetime.timedelta(11), freq='15min', closed='left')

        # drop surplus col and row
        #fix missing:  (follow instructions of doc. Also, a workaround for day 3/4 for possible 3day GFS)
        if not set(day_3) < set(data.index):
            data = data.reindex(list(data.index)+list(day_3))
            data.loc[day_3] = self.day_2_factor_for_day_5 * data.loc[day_2].values + self.day_3_factor_for_day_5 * data.loc[day_3].values + self.day_4_factor_for_day_5 * data.loc[day_1].values
        if not set(day_4) < set(data.index):
            data = data.reindex(list(data.index)+list(day_4))
            data.loc[day_4] = self.day_2_factor_for_day_5 * data.loc[day_2].values + self.day_3_factor_for_day_5 * data.loc[day_3].values + self.day_4_factor_for_day_5 * data.loc[day_1].values
        if not set(day_5) < set(data.index):
            data = data.reindex(list(data.index)+list(day_5))
            data.loc[day_5] = self.day_2_factor_for_day_5 * data.loc[day_2].values + self.day_3_factor_for_day_5 * data.loc[day_3].values + self.day_4_factor_for_day_5 * data.loc[day_4].values
        if not set(day_6) < set(data.index):
            data = data.reindex(list(data.index)+list(day_6))
            data.loc[day_6] = (self.day_2_factor_for_day_6 * data.loc[day_2].values + self.day_4_factor_for_day_6 * data.loc[day_4].values + self.day_5_factor_for_day_6 * data.loc[day_5].values) * self.rand_k
        if not set(day_7) < set(data.index):
            data = data.reindex(list(data.index)+list(day_7))
            data.loc[day_7] = (self.day_3_factor_for_day_7 * data.loc[day_3].values + self.day_4_factor_for_day_7 * data.loc[day_4].values + self.day_5_factor_for_day_7 * data.loc[day_5].values + self.day_6_factor_for_day_7 * data.loc[day_6].values) * self.rand_k
        if not set(day_8) < set(data.index):
            data = data.reindex(list(data.index)+list(day_8))
            data.loc[day_8] = (self.day_5_factor_for_day_8 * data.loc[day_5].values + self.day_6_factor_for_day_8 * data.loc[day_6].values + self.day_7_factor_for_day_8 * data.loc[day_7].values) * self.rand_k
        if not set(day_9) < set(data.index):
            data = data.reindex(list(data.index)+list(day_9))
            data.loc[day_9] = (self.day_5_factor_for_day_9 * data.loc[day_5].values + self.day_7_factor_for_day_9 * data.loc[day_7].values + self.day_8_factor_for_day_9 * data.loc[day_8].values) * self.rand_k
        if not set(day_10) < set(data.index):
            data = data.reindex(list(data.index)+list(day_10))
            data.loc[day_10] = (self.day_6_factor_for_day_10 * data.loc[day_6].values + self.day_7_factor_for_day_10 * data.loc[day_7].values + self.day_8_factor_for_day_10 * data.loc[day_8].values + self.day_9_factor_for_day_10 * data.loc[day_9].values) * self.rand_k
        if not set(day_11) < set(data.index):
            data = data.reindex(list(data.index)+list(day_11))
            data.loc[day_11] = self.day_1_factor_for_day_11 * data.loc[day_1].values + self.day_3_factor_for_day_11 * data.loc[day_3].values + self.day_4_factor_for_day_11 * data.loc[day_4].values
        return data
         
    def step1_get_multi_seg_power(self):
        # read in windspeed for multi_seg, and  get its power later after time extension
        self.multi_seg_curve = pd.read_csv(self.job['modeldir']+'/curve.csv', index_col=0)
        self.speed_weights_valid = self._make_weights_valid(self.speed_weights, self.windspeeds)
        # weithed_speed = weights * values + bias(implicit):
        self.weighted_speed = (self.speed_weights_valid.T * self.windspeeds)
        wind_for_multi_segmented_model = (self.speed_weights_valid.T * self.windspeeds).sum(axis=1)
        # after extension, get multi_seg's power:
        self.powers['power_multi_seg'] = pd.Series(np.interp(wind_for_multi_segmented_model, self.multi_seg_curve.values[:,0], self.multi_seg_curve.values[:,1]), index=wind_for_multi_segmented_model.index)

    def _make_weights_valid(self, weights_config, source_data):
        #compute share weights_config which are in configuratin file yet not in source data
        not_valid_options = list(set(weights_config.index) - set(source_data.columns))
        weights_to_share = weights_config.loc[not_valid_options].sum()
        weights_valid = (weights_config.reindex(source_data.columns)).fillna(0)  
        pure_weights_valid = weights_valid[set(weights_valid.index)-{'speed_bias', 'power_bias'}]  #Note do not share on bias
        pure_weights_valid += pure_weights_valid/pure_weights_valid.sum() * weights_to_share
        weights_valid = pure_weights_valid.append(weights_valid[set(weights_valid.index)-set(pure_weights_valid.index)]) #add back
        if len(not_valid_options)>0:
            info = "*Warning, %s are not valid weight for sources and will not be used! Meanwhile adjusting ratio to:\n %s"%(not_valid_options, weights_valid)
            self.handle_exception(info)
        return weights_valid

    def step2_get_composed_power(self):
        self.power_weights_valid = self._make_weights_valid(self.power_weights, self.powers)
        # combined_power = power_weights * values + bias (implicit):
        self.weighted_power = self.power_weights_valid.T * self.powers
        self.output_power = self.weighted_power.sum(axis=1)
        # subsitution, for day after day-7, just use extended ec:
        self.output_power[self.output_power.index > self.output_power.index[0] + datetime.timedelta(7)] = self.powers.loc[self.output_power.index > self.output_power.index[0] + datetime.timedelta(7), :].mean(axis=1)
        #print('\nFORWARD', self.weighted_power)
        #print('\nFORWARD', self.output_power)

    def postprocess_for_limitations_and_expansions(self):
        #handle power limitation:
        limit_start = datetime.datetime.strptime(self.job['limitBegin'], '%Y-%m-%d %H:%M:%S')
        limit_end = datetime.datetime.strptime(self.job['limitEnd'], '%Y-%m-%d %H:%M:%S')
        expand_start = datetime.datetime.strptime(self.job['limitBegin'], '%Y-%m-%d %H:%M:%S')
        expand_end = datetime.datetime.strptime(self.job['limitEnd'], '%Y-%m-%d %H:%M:%S')
        limit_period = (self.output_power.index > limit_start) & (self.output_power.index < limit_end)
        self.output_power[limit_period] = self.output_power[limit_period] * float(self.job['limitFactor'])
        #handle power expansion:
        expansion_period = (self.output_power.index < limit_start) + (self.output_power.index > limit_end)
        self.output_power[expansion_period] = self.output_power[expansion_period] * float(self.job['expandFactor'])
        #handle threshold:
        self.output_power[np.where(self.output_power < self.lowest_power_threshold)[0]] = self.set_lowest_to
        self.output_power = np.clip(self.output_power, 0, self.cap)
        #handle prediction period:
        self.output_power = pd.DataFrame(self.output_power).reindex(self.prediction_period)
        if len(sys.argv)>1:
            try:
                import matplotlib.pyplot as plt
                for i in self.powers:
                    plt.plot(self.powers.loc[:,i], label=i)
                    pass
                plt.plot(self.output_power, c='r', label='composed')
                plt.legend()
                plt.show()
            except:
                pass
        #print('\nPOST', self.output_power)

    def save_output_file(self):
        output_power_data = pd.DataFrame(columns = self.template_columns)
        output_power_data['Timestamp'] = self.output_power.index
        output_power_data = output_power_data.set_index('Timestamp')
        output_power_data['##id'] = range(1, len(self.output_power)+1)
        output_power_data.loc[self.output_power.index, 'Power'] = self.output_power.values
        with_power = True if self.ec_data.shape[1]==self.num_of_cols_in_with_power_file else False   # Confirmed by SongQian: wpd with(out) power:col=9(6)
        if not with_power:  # when meteor data is from sources without_power
            output_power_data.loc[self.output_power.index, ['Windspeed', 'WindDir', 'Temperature', 'Pressure']] = self.ec_data.reindex(self.prediction_period).values[:, :-2]
            output_power_data.loc[self.output_power.index, 'Humidity'] = self.ec_data.reindex(self.prediction_period).values[:, self.humidity_column_in_sources_without_power]
        else:  
            output_power_data.loc[self.output_power.index, ['Windspeed', 'WindDir', 'Temperature', 'Pressure']] = self.ec_data.reindex(self.prediction_period).values[:, :-5]   #corresponding column in 9\10\12
            output_power_data.loc[self.output_power.index, 'Humidity'] = self.ec_data.reindex(self.prediction_period).values[:, self.humidity_column_in_sources_with_power]
        timestamps = pd.Series(self.output_power.index - datetime.timedelta(hours=8)).apply(lambda i: int(i.timestamp()))   #python timestamp bug, -6h
        timestamps.index = output_power_data.index   #only valid way to give data later
        output_power_data['Timestamp'] = timestamps
        output_power_data = np.round(output_power_data.fillna(0)[self.template_columns], 2)

        #save to wanted format (delimiter = ' '*4)
        output_power_data.to_csv(self.job['outfcstfile'], index=None, sep=' ')
        output_power_data.to_csv('log_lmw/backend_%s_%s.csv'%(self.station_code, time.time()), index=None, sep=' ')
        tmp = open(self.job['outfcstfile'], 'r').readlines()
        for idx,_ in enumerate(tmp):
            tmp[idx] = _.replace(' ', ' '*4)
        tmp = ''.join(tmp)
        with open(self.job['outfcstfile'], 'w') as f:
            f.write(tmp)

    def _read_WPD(self, qx_dir, qx_code, read_its_power):
        # step1, determine filename first
        def this_is_not_met_file_dir():
            try:
                int(glob.glob(qx_dir+'/*.WPD').pop().split('/')[-1].split('.')[0][-10:])
                return True
            except:
                return False
        if this_is_not_met_file_dir():
            source_file = qx_dir +'/'+ qx_code + self.forecast_time.strftime('%Y%m%d') + '06.WPD'
            if source_file not in glob.glob(qx_dir+'/*.WPD'):
                info = '*Warning, will use meteor from 18 instead of 06'
                self.handle_exception(info)
                source_file = qx_dir +'/'+ qx_code + self.forecast_time.strftime('%Y%m%d') + '18.WPD'
                useBakQx = True
                bak_filename = source_file
                main_qx = qx_dir +'/'+ qx_code + self.forecast_time.strftime('%Y%m%d') + '06.WPD'
            else:
                useBakQx = False
                bak_filename = ''
                main_qx = source_file
        else:   # this is met file dir
            source_file = qx_dir +'/'+ qx_code + self.forecast_time.strftime('%Y%m%d') + '.WPD'
            useBakQx = False
            bak_filename = ''
            main_qx = source_file
        # step2, read corresponding file then
        print("Will read platform source file from %s, read power: %s"%(source_file, read_its_power))
        try:
            data = pd.read_csv(source_file, skiprows=2, header=None, sep=r'[ ][ ]+', engine='python', index_col=0, parse_dates=True)
            if len(data)%96!=0:
                data = data.drop(data.index[-(len(data)%96):])
        except Exception as e:
            print(e)
            self.handle_exception(e)
            return None, None, None, None
        with_power = True if data.shape[1]==self.num_of_cols_in_with_power_file else False   # Confirmed by SongQian: wpd with(out) power:col=10(7)
        if with_power:
            if read_its_power:
                platform_col_data = data.iloc[:, self.power_column_in_sources_with_power]   # power column in sources
            else:
                platform_col_data = data.iloc[:, self.wind_column_in_sources_with_power]
        else:
            if read_its_power:
                platform_col_data = np.nan
            else:
                platform_col_data = data.iloc[:, self.wind_column_in_sources_without_power]     # wind column in De_Min
        return platform_col_data, data, useBakQx, bak_filename

    def handle_exception(self, info):
        self.logger.append(info)
        print(info)

def main(json):
    if not os.path.exists('log_lmw/'):
        os.mkdir('log_lmw')
    # Save input json log:
    with open('log_lmw/lmw_request_%s.log'%time.time(), 'w') as f: 
        f.write(str(json) + str(time.time()))

    # Initiate Class:
    model = ModelWeightedSum(json)
    print_handler = PrintHandler('log_lmw/log_%s.log'%time.time())
    #sys.stdout = print_handler.close()
    #run over stations one-by-one:
    for job_idx in range(len(model.jobs)):
        print("\n\nStation job_idx %s"%job_idx)
        model.prepare_a_job(job_idx) 
        model.read_in_sources_data() 
        model.append_bias_columns()
        model.step1_get_multi_seg_power()
        model.extend_over_time()
        model.step2_get_composed_power() 
        model.postprocess_for_limitations_and_expansions()
        model.save_output_file()

        json['plants'][job_idx]['status'] = True
        json['plants'][job_idx]['errdes'] = '\n'.join(model.logger)
        json['plants'][job_idx]['useBakQx'] = model.use_bak_flag
        json['plants'][job_idx]['bakqxlist'] = model.bakqxlist

    # Save output json log:
    with open('log_lmw/lmw_response_%s.log'%time.time(), 'w') as f: 
        f.write(str(json) + str(time.time())) 

    sys.stdout = print_handler.open()
    return json






