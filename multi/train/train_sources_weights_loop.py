# python train_sources_weights_loop.py -A 

import config
import datetime
import pandas as pd
import numpy as np
import os,sys
import matplotlib.pyplot as plt
import shutil
import argparse
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

import warnings
import grasp_from_zhikong_web
import dependency_wind_bases

#from IPython import embed
#import tqdm
#from sko.GA import GA
#import torch
warnings.filterwarnings('ignore')


def evaluate(cap:float, real:pd.Series, predict:pd.Series, method:str='MSE') -> np.array:
    if method == 'MSE':
        error = np.nanmean((real - predict)**2)
    elif method == 'MAE':
        error = np.nanmean(np.abs(real - predict))
    else:
        import AccuracyFormula   #ZouQianKun's lib 
        data = pd.DataFrame({'time':real.index, 'gt':real, 'predict':predict})
        #For NorthWest region
        if method == 'PianChaKaoHe': 
            error = AccuracyFormula.CalcDIP_byDate(data, 'time', 'gt', 'predict', cap, 0.25)
        #For other regions
        elif method == 'KouDian_RMSE':   
            error = AccuracyFormula.CalcKouDian_RMSE_byDate(data, 'time', 'gt', 'predict', cap, 0.8)   #80% for wind RMSE
        elif method == 'KouDian_MAE':  
            error = AccuracyFormula.CalcKouDian_MAE_byDate(data, 'time', 'gt', 'predict', cap, 0.85)   #85% for wind MAE
        else:
            raise NotImplementedError
        error = pd.DataFrame.from_dict(error, orient='index', columns=['koudian'])
        error = error.values.sum()
    return error

def save_log(Y_test:pd.Series, optimized_combination_test:pd.Series, business_power_test:pd.Series, filename:str='log') -> np.array:
    out = pd.DataFrame({'real': Y_test, 'combined': optimized_combination_test, 'business': business_power_test})
    out.to_csv(os.path.join('..', 'output', '%s.csv'%filename))

def batchGradientDescent(x, y, theta, alpha = 0.1, maxIterations=10000):
    m = x.shape[0]
    alpha = alpha/m
    for i in range(0, maxIterations):
        y_pred = np.dot(x, theta)
        ERROR_loss = 1/m * np.sum(np.abs(y_pred - y))
        #ERROR_gradient:
        mask = (y-y_pred).copy()
        mask[y-y_pred>0] = 1
        mask[y-y_pred<=0] = -1
        #theta = theta - alpha * gradient
        theta = theta + alpha * 1/m * mask.dot(x)
        print('epoch', i, ERROR_loss)
    return np.array(theta)

def obj_func(W):
    error = np.nanmean(np.abs((W*X_train.values).sum(axis=1) - Y_train.values))
    #error = np.nanmean(np.abs((W*X_train.values).sum(axis=1) - Y_train.values)**2)
    return error

def save_output(station_name, meteor_powers, w):
    #save output:
    if not os.path.exists(os.path.join('..', 'output', station_name)):
        os.mkdir(os.path.join('..', 'output', station_name))
    meteor_weights = get_ready_output(meteor_powers.columns, w)
    meteor_weights.to_csv(os.path.join('..', 'output', station_name, 'weights.csv'))
    shutil.copy(os.path.join('..', 'data', 'model_curve_data', '%s.csv'%station_name), os.path.join('..', 'output', station_name, 'curve.csv'))
    return meteor_weights

def get_ready_output(column_names, w):
    col_names = [] 
    for i in column_names: 
        col_names.append(i.strip('pow_'))
    meteor_weights = pd.DataFrame(w.reshape(1, -1), columns=col_names)
    aux = { \
    'day_2_factor_for_day_5': 0.333333333,
    'day_3_factor_for_day_5': 0.333333333,
    'day_4_factor_for_day_5': 0.333333333,
    'day_2_factor_for_day_6': 0.45,
    'day_4_factor_for_day_6': 0.35,
    'day_5_factor_for_day_6': 0.2,
    'day_3_factor_for_day_7': 0.4,
    'day_4_factor_for_day_7': 0.05,
    'day_5_factor_for_day_7': 0.2,
    'day_6_factor_for_day_7': 0.35,
    'day_5_factor_for_day_8': 0.333333333,
    'day_6_factor_for_day_8': 0.333333333,
    'day_7_factor_for_day_8': 0.333333333,
    'day_5_factor_for_day_9': 0.45,
    'day_7_factor_for_day_9': 0.35,
    'day_8_factor_for_day_9': 0.2,
    'day_6_factor_for_day_10': 0.4,
    'day_7_factor_for_day_10': 0.05,
    'day_8_factor_for_day_10': 0.2,
    'day_9_factor_for_day_10': 0.35,
    'day_1_factor_for_day_11': 0.4,
    'day_3_factor_for_day_11': 0.3,
    'day_4_factor_for_day_11': 0.3,
    'day_lowest_power_threshold': 5,
    'day_set_lowest_to': 0
    }
    for name in aux.keys():
        meteor_weights[name] = aux[name]
    meteor_weights = meteor_weights.T
    meteor_weights.columns=['weights']
    meteor_weights['source_name'] = meteor_weights.index
    meteor_weights.index = meteor_weights['source_name']
    meteor_weights = meteor_weights.drop('source_name', axis=1)
    return meteor_weights

def get_steady_meteor_powers(meteor_powers):
    meteor_powers = meteor_powers.fillna(0)
    singular_column_names = list(meteor_powers.columns[meteor_powers.values.mean(axis=0)==0])
    if len(singular_column_names) > 0:
        meteor_powers = meteor_powers.drop(singular_column_names, axis=1)
        print("Notice: %s drop for calculation steadyness"%singular_column_names)
    return meteor_powers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_length', '-TRAIN', type=int, default=config.train_length)
    parser.add_argument('--VISUALIZATION', "-V", action='store_true', default=False)
    parser.add_argument('--ANIMATION', "-A", action='store_true', default=False)
    parser.add_argument('--use_spd', "-SPD", action='store_true', default=False)
    parser.add_argument('--test_length', '-TEST', type=int, default=config.test_length)
    parser.add_argument('--shift_months', '-S', type=int, default=config.shift_months)
    parser.add_argument('--data_gap_day', '-G', type=int, default=0)
    parser.add_argument('--loop_days', '-L', type=int, default=config.loop_days)
    parser.add_argument('--method', '-M', type=str, default='ode')
    parser.add_argument('--filename', '-F', type=str, default=config.filename)
    args = parser.parse_args()
    args.test_length = int(args.test_length - 1)
    print(args)
    shift_now = datetime.datetime.today() - datetime.timedelta(args.shift_months*31)
    station_names = pd.read_csv(args.filename, header=None).iloc[:,:]
    start_date_grasp_date = shift_now - datetime.timedelta(args.train_length+args.test_length+args.loop_days+args.data_gap_day)
    start_date_grasp = start_date_grasp_date.strftime("%Y-%m-%d")
    end_date_grasp = shift_now.strftime("%Y-%m-%d")
    sources_to_use = pd.read_csv(os.path.join('..', 'data', 'sources_to_use.csv'), index_col=0)
    sources_to_use = sources_to_use[sources_to_use['use_or_not'] == 1]
   
    logs = pd.DataFrame(index = list(pd.Series(station_names.values.reshape(-1))), columns= ['ERROR_optimized_train', 'ERROR_optimized_test', 'ERROR_business_train', 'ERROR_business_test', 'improvement_train (%)', 'improvement_test (%)', 'remark'])
    for col in logs.columns: 
        for row in logs.index: 
            logs.loc[row, col] = []

    #STATION LOOP:
    overall_LSE_imp = []
    overall_ERROR_imp = []
    for station_name in station_names.iterrows():
        station_name = station_name[1].values[0]
        print("\n\nStation: %s"%station_name)
        print("grasp data %s~%s"%(start_date_grasp, end_date_grasp))
        #read zhikong data:
        raw_data_all, cap, plant_name, FarmType, longitude, latitude = grasp_from_zhikong_web.read_statistic_base(station_name, start_date_grasp, end_date_grasp, readweather=1)
        raw_data_all = raw_data_all.dropna(subset=['power_true'])
        cap = float(cap)
        if len(raw_data_all)==0:
            log = 'no gt'
            logs.loc[station_name, 'remark'] = log
            print(log)
            continue
        assert (FarmType == 0)   #0 for wind
        raw_data_all = raw_data_all.loc[np.append(True, (raw_data_all.power_true.values[1:] - raw_data_all.power_true.values[:-1]) != 0)]

        #get powers:
        real_power = np.clip(np.abs(raw_data_all['power_true']), -cap, cap)
        if 'fore_power' in raw_data_all.columns:
            business_power = np.clip(np.abs(raw_data_all['fore_power']), -cap, cap)
        else:
            log = 'no fore power'
            logs.loc[station_name, 'remark'] = log
            raw_data_all['fore_power'] = 0.1
            business_power = np.clip(np.abs(raw_data_all['fore_power']), -cap, cap)
            print(log)
        column_names_to_use = [] 
        for i in raw_data_all.columns: 
            if args.use_spd:
                use_feature = (i.startswith('pow_') or i.startswith('spd_'))
            else:
                use_feature = i.startswith('pow_')
            if use_feature and (i in sources_to_use.index):
                column_names_to_use.append(i)
        #get de min power:
        if 'de_min' in sources_to_use.index:
            #Use curve to give power prediction:
            DeMin_curve = pd.read_csv(os.path.join('..', 'data', 'model_curve_data', '%s.csv'%station_name), index_col=0)
            if 'spd_7' not in raw_data_all.columns:
                log = 'No spd_7, thus no de_min'
                logs.loc[station_name, 'remark'] = log
                print(log)
            else:
                column_names_to_use += ['de_min']
                DeMin_prediction = pd.Series(np.interp(raw_data_all.loc[raw_data_all.index, 'spd_7'], DeMin_curve.values[:,0], DeMin_curve.values[:,1]), index=raw_data_all.index)
                raw_data_all['de_min'] = DeMin_prediction
        meteor_powers = raw_data_all[column_names_to_use]
        if len(raw_data_all) == 0:
            log = 'no gt data' 
            logs.loc[station_name, 'remark'] = log
            print(log)
            continue
        elif meteor_powers.shape[1] == 0:
            log = 'no meteor_powers'
            logs.loc[station_name, 'remark'] = log
            print(log)
            continue
        else:
            #TIMESLICE LOOP:   #when loop over train and test
            error_recorder = dependency_wind_bases.Recorder()
            concat_optimized_test = pd.Series([], dtype=float)
            concat_business_test = pd.Series([], dtype=float)
            concat_real_test = pd.Series([], dtype=float)
            plt.ion()
            for i in list(range(args.loop_days)):
                print('Time slice', i)
                start_date_train = start_date_grasp_date + datetime.timedelta(i)
                end_date_train = start_date_train + datetime.timedelta(args.train_length)
                start_date_test = end_date_train + datetime.timedelta(int(args.data_gap_day+1))
                end_date_test = start_date_test + datetime.timedelta(args.test_length)
                start_date_train_str = start_date_train.strftime("%Y-%m-%d")
                end_date_train_str = end_date_train.strftime("%Y-%m-%d")
                start_date_test_str = start_date_test.strftime("%Y-%m-%d")
                end_date_test_str = end_date_test.strftime("%Y-%m-%d")
                print("Train from %s to %s"%(start_date_train_str, end_date_train_str))
                print("Test from %s to %s"%(start_date_test_str, end_date_test_str))
    
                meteor_powers_slice = meteor_powers.loc[start_date_train_str: end_date_test_str]
                real_power_slice = real_power[meteor_powers_slice.index]

                #split dataset:
                X = meteor_powers_slice
                X['bias'] = 1
                Y = real_power_slice
                X_train, X_test = X.loc[start_date_train_str:end_date_train_str], X.loc[start_date_test_str:end_date_test_str]
                Y_train, Y_test = Y.loc[start_date_train_str:end_date_train_str], Y.loc[start_date_test_str:end_date_test_str]
                #handle duplicates 
                X_train = get_steady_meteor_powers(X_train).dropna()
                X_test = X_test[X_train.columns].dropna()
                Y_train = Y_train.dropna()
                Y_test = Y_test.dropna()
    
                if len(set(X_train.columns) - {'de_min', 'bias'}) == 0:
                    log = 'source not enough'
                    logs.loc[station_name, 'remark'] = log
                    print(log)
                    continue
                if X_train.shape[0] < X_train.shape[1]:
                    print("shape of X 0<1")
                    continue
                if Y_test.shape[0] < 6:
                    print("len Y <6")
                    continue                
                business_power_train = business_power.loc[Y_train.index]
                business_power_test = business_power.loc[Y_test.index]

                #Choose methods:
                if args.method == 'ode':
                    #solve ODE equation:
                    try:
                        w = np.linalg.solve(np.dot(X_train.T.copy(), X_train), np.dot(X_train.T.copy(), Y_train)) 
                    except Exception as e:
                        log = '%s, \n %s, \t %s'%(e, X_train.describe(), X_test.describe())
                        #log = '%s'%e
                        logs.loc[station_name, 'remark'] = log
                        print(log)
                        continue
                elif args.method == 'gd':
                    init_w = [1/X_train.shape[1]]*(X_train.shape[1]-1)   # -1 for bias
                    w = batchGradientDescent(X_train, Y_train, init_w+[1], alpha=0.1)
                elif args.method == 'ga':   #ga
                    w = np.tile(0, (1,5))
                    n_dim = X_train.shape[1]
                    lb = [-3]*(n_dim-1);lb.append(-20)
                    ub = [3]*(n_dim-1);ub.append(20)
                    ga = GA(func=obj_func, n_dim=n_dim, size_pop=1000, max_iter=1000, lb=lb, ub=ub)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    #ga.to(device=deivce)
                    w, residuals = ga.run()
                elif args.method == 'poly':   #poly lr
                    w = np.tile(0, (1,5))
                    poly = PolynomialFeatures(degree=2)
                    poly_X_train = poly.fit_transform(X_train.iloc[:,:-1])
                    poly_X_test = poly.fit_transform(X_test.iloc[:,:-1])
                    regressor = LinearRegression()
                    regressor.fit(poly_X_train, Y_train)
                elif args.method == 'xgb':
                    w = np.tile(0, (1,5))
                    regressor = XGBRegressor(max_depth=4)
                    regressor.fit(X_train, Y_train)
                elif args.method == 'lasso':
                    w = np.tile(0, (1,5))
                    regressor = Lasso()
                    regressor.fit(X_train, Y_train)
                elif args.method == 'ridge':
                    w = np.tile(0, (1,5))
                    regressor = Ridge()
                    regressor.fit(X_train, Y_train)
                elif args.method == 'mlp':
                    w = np.tile(0, (1,5))
                    regressor = MLPRegressor()
                    regressor.fit(X_train, Y_train)
                else:
                    regressor = None
                #eval train:
                if args.method == 'ode' or args.method == 'ga' or args.method == 'gd':
                    optimized_combination_train = (w*X_train).sum(axis=1)
                    optimized_combination_test = (w*X_test).sum(axis=1)
                elif args.method == 'poly':
                    optimized_combination_train = regressor.predict(poly_X_train)
                    optimized_combination_test = regressor.predict(poly_X_test)
                    optimized_combination_train = pd.Series(optimized_combination_train, index=X_train.index)
                    optimized_combination_test = pd.Series(optimized_combination_test, index=X_test.index)
                else:
                    optimized_combination_train = regressor.predict(X_train)
                    optimized_combination_test = regressor.predict(X_test)
                    optimized_combination_train = pd.Series(optimized_combination_train, index=X_train.index)
                    optimized_combination_test = pd.Series(optimized_combination_test, index=X_test.index)
               
                
                #eval train:
                optimized_combination_train = np.clip(optimized_combination_train, 0, max(Y))
                ERROR_optimized_train = evaluate(cap, Y_train, optimized_combination_train, method=config.eval_metric)
                ERROR_business_train = evaluate(cap, Y_train, business_power_train, config.eval_metric)
                ERROR_improvement_train = (ERROR_business_train-ERROR_optimized_train)/ERROR_business_train*100
                #eval test:
                optimized_combination_test = np.clip(optimized_combination_test, 0, max(Y))
                ERROR_optimized_test = evaluate(cap, Y_test, optimized_combination_test, config.eval_metric)
                ERROR_business_test = evaluate(cap, Y_test, business_power_test, config.eval_metric)
                ERROR_improvement_test = (ERROR_business_test-ERROR_optimized_test)/ERROR_business_test*100
                #save externals:
                save_log(Y_test, optimized_combination_test, business_power_test, station_name)
                meteor_weights = save_output(station_name, X_train, w)
                print("Train Improvement from %s to %s, %s%%"%(ERROR_business_train, ERROR_optimized_train, ERROR_improvement_train))
                print("Test Improvement from %s to %s, %s%%"%(ERROR_business_test, ERROR_optimized_test, ERROR_improvement_test))
                #print('Weight:', meteor_weights)
                if args.ANIMATION:
                    plt.plot(meteor_powers_slice, 'blue', alpha=0.2, linewidth=3, label='sources')
                    plt.plot(Y_train, 'k', alpha=0.5)
                    plt.plot(optimized_combination_train, 'g', alpha=0.5)
                    plt.plot(business_power_train, 'r', alpha=0.5)
                    plt.plot(Y_test, 'k', label='real')
                    plt.plot(optimized_combination_test, 'g', label='optimized', linestyle='--')
                    plt.plot(business_power_test, 'r', label='business', linestyle=':')
                    plt.title('%s, %s, %s'%(station_name, ERROR_improvement_train, ERROR_improvement_test))
                    plt.legend()
                    plt.grid()
                    plt.draw()
                    plt.pause(0.1)
                    plt.clf()
 
                #Misc
                concat_optimized_test = concat_optimized_test.append(optimized_combination_test)
                concat_business_test = concat_business_test.append(business_power_test)
                concat_real_test = concat_real_test.append(Y_test)
                error_recorder.add_one('%s_ERROR_optimized_train'%station_name, ERROR_optimized_train)
                error_recorder.add_one('%s_ERROR_optimized_test'%station_name, ERROR_optimized_test)
                error_recorder.add_one('%s_ERROR_business_train'%station_name, ERROR_business_train)
                error_recorder.add_one('%s_ERROR_business_test'%station_name, ERROR_business_test)
                error_recorder.add_one('%s_improvement_train (%%)'%station_name, ERROR_improvement_train)
                error_recorder.add_one('%s_improvement_test (%%)'%station_name, ERROR_improvement_test)
            #TIME LOOP DONE.
            plt.close()
            plt.ioff()

            #Mean over redudant slices:
            if len(concat_optimized_test) == 0: 
                print("len concat test =0")
                continue

            #Concatenate all slices of timeloops
            optimized_combination_test = concat_optimized_test.resample('15min').mean().dropna()
            business_power_test = business_power.reindex(optimized_combination_test.index).dropna()
            common_index = optimized_combination_test.index &  business_power_test.index 
            optimized_combination_test = optimized_combination_test.loc[common_index]
            business_power_test = business_power_test.loc[common_index]
            real_power_test = real_power.loc[common_index]
            ERROR_opt = np.nanmean(np.abs(optimized_combination_test - real_power_test))
            LSE_opt = np.nanmean((optimized_combination_test - real_power_test)**2)
            ERROR_bus = np.nanmean(np.abs(business_power_test - real_power_test))
            LSE_bus = np.nanmean((business_power_test - real_power_test)**2)
            ERROR_imp = (ERROR_bus-ERROR_opt)/ERROR_bus*100
            LSE_imp = (LSE_bus-LSE_opt)/LSE_bus*100
            logs.loc[station_name, 'ERROR_optimized_train']   = error_recorder.get_mean('%s_ERROR_optimized_train'%station_name)
            logs.loc[station_name, 'ERROR_optimized_test']    = error_recorder.get_mean('%s_ERROR_optimized_test'%station_name)
            logs.loc[station_name, 'ERROR_business_train']    = error_recorder.get_mean('%s_ERROR_business_train'%station_name)
            logs.loc[station_name, 'ERROR_business_test']     = error_recorder.get_mean('%s_ERROR_business_test'%station_name)
            logs.loc[station_name, 'improvement_train (%)']   = error_recorder.get_mean('%s_improvement_train (%%)'%station_name)
            logs.loc[station_name, 'improvement_test (%)']    = error_recorder.get_mean('%s_improvement_test (%%)'%station_name)
            logs.loc[station_name, 'loop_optimized_output'] = ','.join(np.round(optimized_combination_test, 1).astype(str).to_list())
            logs.loc[station_name, 'loop_real_output'] = ','.join(np.round(real_power_test, 1).astype(str).to_list())
            logs.loc[station_name, 'loop_business_power_test'] = ','.join(np.round(business_power_test, 1).astype(str).to_list())
            logs.loc[station_name, 'loop_test ERROR (%)'] = ERROR_imp
            logs.loc[station_name, 'loop_test LSE (%)'] = LSE_imp
            print('loop given: ERROR:%s, LSE:%s'%(ERROR_imp, LSE_imp))

            #plots:
            fig = plt.figure(figsize=(18, 10))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.plot(meteor_powers.loc[Y_train.index], alpha=0.4, c='gray', label='sources')
            ax1.plot(real_power.loc[Y_train.index], label='real', c='k')
            ax1.plot(optimized_combination_train, label='optimized', c='g')
            ax1.plot(business_power_train, label='business', c='r')
            ax2.plot(meteor_powers.reindex(optimized_combination_test.index), alpha=0.4, c='gray', label='sources')
            ax2.plot(real_power_test, label='real', c='k')
            ax2.plot(optimized_combination_test, label='optimized', c='g', alpha=1)
            ax2.plot(business_power_test, label='business', c='r', alpha=0.8)
            ax1.legend()
            ax2.legend()
            ax1.set_title("%s \nTrain result for meteor sources, improvement(%s): %s%%"%(station_name, config.eval_metric, ERROR_improvement_train))
            ax2.set_title("%s \nTest result for meteor sources, improvement(%s): %s%%"%(station_name, config.eval_metric, ERROR_improvement_test))
            ax1.grid()
            ax2.grid()
            if args.VISUALIZATION:
                plt.show()    
            plt.savefig(os.path.join('..', 'png', '%s_%s_%s.png'%(station_name, args.train_length, args.test_length)))
            plt.close()
    #STATION LOOP DONE    


    #Statistics:
    #logs.to_csv(os.path.join('..', 'output', 'detials_%s_%s_%s_%s_%s_%s-%s_loop%s.csv'%(args.train_length, args.test_length, np.nanmean(overall_LSE_imp), args.method, args.use_spd, start_date_grasp, end_date_grasp, args.loop_days)))
    #print(logs)
    #print(logs.describe())

    print("Finish list", args.filename)


