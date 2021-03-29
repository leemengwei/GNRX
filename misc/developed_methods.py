#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import math
import warnings
import matplotlib.pyplot as plt  
from IPython import embed
import os,sys,copy
import datetime
import shutil
warnings.filterwarnings('ignore')
import dependency_wind_bases
import scipy
try:
    import statsmodels.formula.api as smf  #Quantile reg by lmw
    from statsmodels.tsa.arima_model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.api import Holt
    from sklearn.gaussian_process import GaussianProcessRegressor
    import pmdarima as pm
except:
    print("Import error, some method may not be supported")
    pass

PLOT = True
PLOT = False


def Holt_Linear(true_data, regress_from, pre_Monthseries):
    model = Holt(true_data[regress_from].dropna(), initial_level=true_data[regress_from].dropna()[0], initial_trend=true_data[regress_from].dropna()[1]-true_data[regress_from].dropna()[0], initialization_method='known')
    model = model.fit()
    #model = Holt(true_data[regress_from].dropna())
    #model = model.fit(initial_level=true_data[regress_from].dropna()[0], initial_trend=true_data[regress_from].dropna()[1]-true_data[regress_from].dropna()[0])  #mannual given value
    print(model.summary())
    pred = model.forecast(len(pre_Monthseries))  #can only m by m
    pred = pd.Series(pred, index=pre_Monthseries)
    _ = copy.deepcopy(pred)
    _.index = _.index.strftime('%Y-%m')
    out_dict = _.to_dict()
    return pred, out_dict

def Holt_Winters(true_data, regress_from, pre_Monthseries):
    _mode_ = 'add'  #'multiplicative'
    model = ExponentialSmoothing(true_data[regress_from].dropna(), seasonal_periods=12, trend=_mode_, seasonal=_mode_)
    try:
        model = model.fit()
    except:
        model = ExponentialSmoothing(true_data['power'].dropna(), seasonal_periods=12, trend=_mode_, seasonal=_mode_)
        model = model.fit()
    pred = model.forecast(len(pre_Monthseries))
    pred = pd.Series(pred, index=pre_Monthseries)
    _ = copy.deepcopy(pred)
    _.index = _.index.strftime('%Y-%m')
    out_dict = _.to_dict()
    return pred, out_dict

def counter_month(true_data, regress_from, pre_Monthseries):
    pred = []
    for i in pre_Monthseries:
        avg = np.nanmean(true_data[true_data.month==i.month][regress_from])
        pred.append(np.nanmean([avg]))
    out_dict = {}
    for idx,i in enumerate(pre_Monthseries):
       out_dict['%s-%s'%(i.year, i.month)] = pred[idx]
    pred = pd.Series(pred, index=pre_Monthseries)
    return pred, out_dict

def auto_arima(true_data, regress_from, pre_Monthseries, exogenous):
    true_data = true_data.drop('wind', axis=1)
    true_data = true_data.dropna(subset=[regress_from])
    exogenous.loc[:,:] = 0
    exogenous_test = exogenous.loc[pre_Monthseries]
    exogenous_train = exogenous.loc[true_data.index]
    model = pm.auto_arima(true_data[regress_from].dropna(), start_p=0, start_q=0, start_P=0, start_Q=0, max_p=5, max_q=5, max_P=5, max_Q=5, D=0, seasonal=True, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True, n_jobs=-1, m=12, information_criterion='bic', exogenous=exogenous_train[['random']])
    pred = model.predict(n_periods=len(pre_Monthseries), exogenous=exogenous_test[['random']])
    pred = pd.Series(pred, index=pre_Monthseries)
    _ = copy.deepcopy(pred)
    _.index = _.index.strftime('%Y-%m')
    out_dict = _.to_dict()
    return pred, out_dict

def guassian_model(cap, true_data, regress_from, pre_Monthseries): 
    guassian_pd = pd.DataFrame(columns=['level_mean', 'level_std', 'trend_mean', 'trend_std', 'prediction'], index = pre_Monthseries)
    for i in pre_Monthseries:
        # guassian of wind
        level = true_data[true_data.month==i.month][regress_from].values
        # guassian of trend
        trend = true_data.diff()[true_data.month==i.month][regress_from].values
        trend_2 = true_data.diff().diff()[true_data.month==i.month][regress_from].values
        guassian_pd.loc[i, 'level_mean'] = np.nanmean(level)
        guassian_pd.loc[i, 'level_std'] = np.nanstd(level) if not np.nanstd(level)==0 else 1
        guassian_pd.loc[i, 'trend_mean'] = np.nanmean(trend)
        guassian_pd.loc[i, 'trend_std'] = np.nanstd(trend) if not np.nanstd(trend)==0 else 1
        guassian_pd.loc[i, 'trend_mean_2'] = np.nanmean(trend_2)
        guassian_pd.loc[i, 'trend_std_2'] = np.nanstd(trend_2) if not np.nanstd(trend_2)==0 else 1
    #forecast:
    for idx,i in enumerate(pre_Monthseries):
        if idx==0:    # the first 'last month' is given by ground truth
            last_month = true_data[(i - datetime.timedelta(15)).strftime('%Y-%m')][regress_from]   # -15 will give last month only, since index is like 2020-12-01
            last_month_2 = true_data[(i - datetime.timedelta(30+15)).strftime('%Y-%m')][regress_from]
        else:         # later 'last months' are given by prediction
            last_month = guassian_pd.loc[(i - datetime.timedelta(15)).strftime('%Y-%m'), 'prediction']
            #last_month_2 = guassian_pd.loc[(i - datetime.timedelta(30+15)).strftime('%Y-%m'), 'prediction']
        _scope_ = 10000
        prob_given_by_level = scipy.stats.norm(guassian_pd.loc[i, 'level_mean'], guassian_pd.loc[i, 'level_std']).pdf(np.arange(-_scope_, _scope_))
        prob_given_by_trend = scipy.stats.norm(guassian_pd.loc[i, 'trend_mean']+last_month, guassian_pd.loc[i, 'trend_std']).pdf(np.arange(-_scope_, _scope_))
        #prob_given_by_trend_2 = scipy.stats.norm(guassian_pd.loc[i, 'trend_mean_2']+last_month_2, guassian_pd.loc[i, 'trend_std_2']).pdf(np.arange(-_scope_, _scope_))
        prob_all = prob_given_by_level + prob_given_by_trend
        guassian_pd.loc[i, 'prediction'] = np.argmax(prob_all) - _scope_    #from pdf index to its corresponding value
    pred = guassian_pd['prediction']
    _avg, _ = counter_month(true_data, regress_from, pre_Monthseries)
    pred = (pred+_avg)/2
    _ = copy.deepcopy(pred)
    _.index = _.index.strftime('%Y-%m')
    out_dict = _.to_dict()
    #plt.clf()
    #plt.plot(prob_given_by_level, label='level')
    #plt.plot(prob_given_by_trend, label='trend')
    #plt.plot(prob_all, label='all')
    #plt.legend()
    #plt.show()
    print(pred)
    if pred.any()<0:
        print('**Warning, prediction should not less than zero!', pred)
    return pred, out_dict

def holt_as_constraint(cap, true_data, regress_from, pre_Monthseries, predict_on_year=False):
    '''
    Method use by doc Xu converted from matlab, in which holt's equation is as constraint, double/triple param holts are implemented.
    Personally I reckon this method is of NO sense of original Holt, NOR physical significance, just a bit of LUCK. The only reason this method is refered to is, so far to the present, it yeilds better performance.
    Note: this method only predict one following value.
    Though this method can be run over time to generated a series of prediction iteratively, I'll just use this method once to get first prediction and just use average for others.
    '''
    if not predict_on_year:
        _avg, _ = counter_month(true_data, 'power', pre_Monthseries)
        _target = _avg.values[0]  # scalar
    else:
        _target = np.nanmean(true_data['power'])
    y = true_data.power.dropna().values
    def doublesmooth(y, alpha, beta):
        n = len(y)
        l = np.zeros(n)
        b = np.zeros(n)
        l[0] = y[0]
        b[0] = y[1]-y[0]
        for t in range(1,n):
            l[t] = alpha*y[t] + (1-alpha)*(l[t-1]+b[t-1])
            b[t] = beta*(l[t]-l[t-1]) + (1-beta)*b[t-1]
        return l, b   #level and trend
    def triplesmooth(y, alpha, beta, gamma, m):
        n = len(y)
        l = np.zeros(n)
        b = np.zeros(n)
        s = np.zeros(n)
        l[0] = y[0]
        b[0] = y[1]-y[0]
        for t in range(1,n):
            l[t] = alpha*(y[t]-s[t-m]) + (1-alpha)*(l[t-1]+b[t-1])
            b[t] = beta*(l[t]-l[t-1]) + (1-beta)*b[t-1]
            s[t] = gamma*(y[t]-l[t-1]-b[t-1]) + (1-gamma)*s[t-m]
        return l, b, s
    # grid given:
    l1 = np.linspace(0.1, 1, 51)[:-1]
    l2 = np.linspace(0.1, 1, 51)[:-1]
    l3 = np.linspace(0.1, 1, 11)[:-1]
    params = []
    double = True
    if double:
        f = np.zeros((len(l1), len(l2)))
        for ia, alpha in enumerate(l1):
            for ib, beta in enumerate(l2):
                lx, bx = doublesmooth(y, alpha, beta) #levels and trends
                f[ia, ib] = lx[-1] + bx[-1]
                params.append([l1, l2])
    else:
        m = 12
        f = np.zeros((len(l1), len(l2), len(l3)))
        for ia, alpha in enumerate(l1):
            for ib, beta in enumerate(l2):
                for ic, gamma in enumerate(l3):
                    lx, bx, gx = triplesmooth(y, alpha, beta, gamma, m)
                    f[ia, ib, ic] = lx[-1] + bx[-1] + gx[-m]
                    params.append([l1, l2, l3])
    # find the pair that closest to target (average of that month)
    diff = np.abs(f - _target)
    pred = f.reshape(-1)[diff.argmin()]
    only_first_month = pre_Monthseries[:1]
    pred = pd.Series(pred, index=only_first_month)
    _ = copy.deepcopy(pred)
    _.index = _.index.strftime('%Y-%m')
    out_dict = _.to_dict()
    return pred, out_dict 

def mixture(cap, true_data, regress_from, pre_Monthseries):
    base_pred, out_dict = counter_month(true_data, regress_from, pre_Monthseries)
    pred, out_dict = holt_as_constraint(cap, true_data, regress_from, pre_Monthseries)
    base_pred.update(pred)
    _ = copy.deepcopy(base_pred)
    _.index = _.index.strftime('%Y-%m')
    out_dict = _.to_dict()
    return base_pred, out_dict 

def guassian_process(cap, true_data, regress_from, pre_Monthseries):
    Y = true_data.power.dropna().values
    X = np.array(range(len(Y)))
    model = GaussianProcessRegressor()
    model.fit(X.reshape(-1,1),Y)
    y_pred, std_pred = model.predict(np.array(range(len(Y), len(Y)+len(pre_Monthseries))).reshape(-1,1), return_std=True)
    full_pred, full_std = model.predict(np.array(range(len(Y)+len(pre_Monthseries))).reshape(-1,1), return_std=True)
    #plt.close()
    #plt.scatter(X, Y)
    #plt.plot(np.array(range(len(Y)+len(pre_Monthseries))),full_pred)
    #plt.fill_between(np.array(range(len(Y)+len(pre_Monthseries))), full_pred-full_std, full_pred+full_std)
    #plt.show() 
    pred = pd.Series(y_pred, index=pre_Monthseries)
    _ = copy.deepcopy(pred)
    _.index = _.index.strftime('%Y-%m')
    out_dict = _.to_dict()
    #raise NotImplementedError
    return pred, out_dict
    
def year_prediction(cap, gt_file, pre_Monthseries, true_data, method, regress_from='power', exogenous=None):
    true_data = true_data.resample('Y').mean()*12
    pred, out_dict = holt_as_constraint(cap, true_data, regress_from, pre_Monthseries, predict_on_year=True)
    year_mean = pred.values[0]
    return year_mean

def new_method_prediction(cap, gt_file, pre_Monthseries, true_data, method, regress_from='power', exogenous=None):
    if 'arima' in method:
        pred, out_dict = auto_arima(true_data, regress_from, pre_Monthseries, exogenous=exogenous)
    if 'counter_month' in method:
        pred, out_dict = counter_month(true_data, regress_from, pre_Monthseries)
    if 'linear' in method:
        pred, out_dict = Holt_Linear(true_data, regress_from, pre_Monthseries)
    if 'winter' in method:
        pred, out_dict = Holt_Winters(true_data, regress_from, pre_Monthseries)
    if 'guassian' in method:
        pred, out_dict = guassian_model(cap, true_data, regress_from, pre_Monthseries)
    if 'constraint' in method:
        pred, out_dict = holt_as_constraint(cap, true_data, regress_from, pre_Monthseries)
    if 'mixture' in method:
        pred, out_dict = mixture(cap, true_data, regress_from, pre_Monthseries)
    if 'guassian_process' in method:
        pred, out_dict = guassian_process(cap, true_data, regress_from, pre_Monthseries)
    if PLOT and method != 'counter_month':
        fig = plt.figure(figsize=(18,14))
        ax1 = fig.add_subplot(211)
        ax1.scatter(true_data['wind'], true_data['power']/cap, label='real_power') 
        ax1.set_title('%s'%gt_file)
        ax1.grid()
        ax1.legend()
        ax2 = fig.add_subplot(212)
        ax2.plot(true_data.index, true_data['power'], label='real_power', c='k')
        ax2.plot(pred.index, pred.values, label='prediction', c='red')
        ax2.scatter(pred.index, pred.values, label='prediction', c='red', s=5)
        ax2.grid()
        ax2.legend()
        plt.draw()
        plt.pause(1)
        input()
        plt.close()
    return out_dict

