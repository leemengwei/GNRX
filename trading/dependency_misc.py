import numpy as np
import argparse
import config
from IPython import embed
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_req(mode):
    req_json = \
	{
		"farm_code": "202012_NMGYNSH",
		"cap": 20, 
		#"is_solar": 1, 
		#"ratio_cap": [1.0], 
		"farm_loss": 0.01, 

		#"nc_path":'', 
		#"new_weather_path": "",
		"supply_path": "../data/202012_NMGYNSH/all_in_one_202012_NMGYNSH.csv",
		#"resources_path": "../data/202012_NMGYNSH/", 

		#"days_left": 5,
		#"plan_left": 55,
		#"plan_get": 60, 
		#"midlong_left": 78, 
		"midlong_price": 55.7,
		#"butie_price": 0,
		"benchmark_price": 282.9,
		"yonghu_short": list(np.linspace(0,20,96)),
		"yonghu_short": [],

		"mode": mode,
		"wanted_day":'2020-12-20',

	#	"farm_code": "GSXHGHKH",
	#	"cap": 200, 
	#	"supply_path": "../data/GSXHGHKH/GSXHGHKH20210323.csv",
	#	"wanted_day":'2021-03-22',
	}
    return req_json

def form_rep(self):
    if self.req_json['mode'] == 'deploy':
        pd_wanted_day = self.calculated_data['use_pulp'][self.req_json['wanted_day']]
        rep_json = \
    	{
    	'pred_price': {'price_riqian': pd_wanted_day['riqianPriceClean_pred'].to_list(), 'price_shishi': pd_wanted_day['shishiPriceClean_pred'].to_list(), 'percentage_riqian': self.riqianPriceClean_confidence, 'percentage_shishi': self.shishiPriceClean_confidence, 'flag': 0, 'df_price_pred': None},
    	'pred_power': {'power_midlong': pd_wanted_day['midlongQuantityDecomposeAll'].to_list(), 'power_riqian': pd_wanted_day['riqianPowerClean_pred'].to_list(), 'power_shishi': pd_wanted_day['shishiPowerClean_pred'].to_list(), 'flag':0, 'df_AGCpower_pred': None, 'percentage_riqian': self.riqianPowerClean_confidence, 'percentage_shishi': self.shishiPowerClean_confidence},
    	'pred_xiandian': {'p96':[], 'xiandian':[], 'p_value': 0.0, 'flag':0},
    	'strategy_power':
    		{
    		'optimized': {'curve_shenbao': list(self.declarations['use_pulp']), 'curve_midlong':[], 'curve_riqian':[], 'piancha': self.additional_income['use_pulp'], 'earning': self.incomes['use_pulp'], 'price_riqian': pd_wanted_day['riqianPriceClean_pred'].to_list(), 'price_shishi': pd_wanted_day['shishiPriceClean_pred'].to_list(), 'flag':0},  
    		'manual': {'curve_shenbao': list(self.declarations['yonghu']), 'curve_midlong': [], 'piancha': self.additional_income['yonghu'], 'curve_riqian':[], 'earning': self.incomes['origin'], 'price_riqian': pd_wanted_day['riqianPriceClean_pred'].to_list(), 'price_shishi': pd_wanted_day['shishiPriceClean_pred'].to_list(), 'flag':0},
    		'origin': {'curve_shenbao': list(self.declarations['origin']), 'curve_midlong':[], 'curve_riqian':[], 'earning':100, 'piancha': self.additional_income['origin'], 'price_riqian': pd_wanted_day['riqianPriceClean_pred'].to_list(), 'price_shishi': pd_wanted_day['shishiPriceClean_pred'].to_list(), 'flag':0}
    		},
    	'strategy_price': {'stages_power':[0, 8*self.req_json['cap'], 20*self.req_json['cap']], 'stages_prices':[np.random.randint(1,10), np.random.randint(11, 25)]}
    	}
    else:
        rep_json = \
        {
        'status': [0, 0, 0],
        'infos': ''
        }
    return rep_json

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--val_length', '-VL', type=int, default = 3)
    parser.add_argument('--feature_shift_steps', '-FSS', type=int, default = 0)
    parser.add_argument('--VISUALIZATION', "-V", action='store_true', default=False)
    args = parser.parse_args()
    config.VISUALIZATION = args.VISUALIZATION
    config.feature_shift_steps = args.feature_shift_steps
    config.val_length = args.val_length
    return config

def plot_incomes(income_parts, info):
    plt.figure(figsize=(14,12))
    ax1 = plt.subplot(211)
    # matplotlib plot line:
    for income_part in income_parts:
        ax1.plot(income_parts[income_part], label="%s:%s"%(income_part, np.nansum(income_parts[income_part])))
    ax1.set_title('Income components of strategy: %s'%info)
    ax1.grid()
    ax1.legend()
    
    #ax2 = plt.subplot(212)
    ## pd plot pie:
    #income_pd = pd.DataFrame(income_parts).sum(axis=0)
    #width = 0.3
    #draw_on_factors_outter = ['income_future_market', 'income_spot_market', 'deviation_penalty']
    #draw_on_factors_inner = ['income_midlong_contract', 'income_base', 'income_riqian_market', 'income_shishi_market', 'income_share_market', 'deviation_penalty']
    #patches_outter, text_outter, pcts_outter = ax2.pie(income_pd[draw_on_factors_outter].values, radius=1, wedgeprops=dict(width=width, edgecolor='w'), autopct='%1.1f%%', pctdistance=1-width/2)
    #patches_inner, text_inner, pcts_inner = ax2.pie(income_pd[draw_on_factors_inner].values, radius=1-width, wedgeprops=dict(width=width, edgecolor='w'), autopct='%1.1f%%', pctdistance=1-width)
    #patches = patches_outter + patches_inner
    #texts = text_outter + text_inner
    #pcts = pcts_outter + pcts_inner
    #draw_on_factors = draw_on_factors_outter + draw_on_factors_inner
    ##embed()
    #ax2.set_title('Income porpotion of strategy: %s'%info)
    #ax2.legend(patches, draw_on_factors, loc='upper right', bbox_to_anchor=(1.8,1))

    # plot bar, pie can't work with negtives...
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(224)
    income_pd = pd.DataFrame(income_parts).sum(axis=0)/10000
    draw_on_factors_spot = ['income_riqian_market', 'income_shishi_market', 'income_share_market'] 
    draw_on_factors_future = ['income_midlong_contract', 'income_base']
    income_pd[draw_on_factors_spot].plot.bar(ax=ax2)
    income_pd[draw_on_factors_future].plot.bar(ax=ax3)
    ax2.set_title('Spot market: \n %s'%draw_on_factors_spot)
    ax3.set_title('Future market: \n %s'%draw_on_factors_future)
    ax2.set_ylabel('RMB(W)')
    ax3.set_ylabel('RMB(W)')
    ax2.tick_params(labelrotation=0) 
    ax3.tick_params(labelrotation=0) 
    ax2.grid()
    ax3.grid()
    #embed()

    plt.show()


'''
Data description: 
1, all pred data are given in D-1 day;
2, trading take place in D day;
3, gt data are given in D+1 day;
4, clean data are given in D+4 day.
'''
data_rename_dict = {
'统调负荷-预测值': 'powerNeed_pred',
'统调负荷-实测值': 'powerNeed_gt',
'东送计划-预测值': 'powerOut_pred',
'东送计划-实测值': 'powerOut_gt',
'全网风电-预测值': 'powerWind_pred', 
'全网风电-实测值': 'powerWind_gt', 
'全网光伏-预测值': 'powerSolar_pred', 
'全网光伏-实测值': 'powerSolar_gt', 
'新能源出-预测值': 'powerNew_pred', 
'新能源出-实测值': 'powerNew_gt', 
'正负备用-正备用': 'posBackup', 
'正负备用-负备用': 'negBackup', 

'日分解及-日分解电力': 'midlongPowerDecomposeStation',
'日分解及-日前出清电力': 'riqianPowerClean',
'日分解及-日前出清电价': 'riqianPriceClean',
'日分解及-实时出清电力': 'shishiPowerClean',
'日分解及-实时出清电价': 'shishiPriceClean',
#'日分解及-日内出清电力': 'rineiPowerClean', #abort 
#'日分解及-日内出清电价': 'rineiPriceClean', #abort

# from 07-JieSuanJieGuoChaXun
'全网中长期日分解电量（兆瓦时）': 'midlongQuantityDecomposeAll', 
'日前全网修正电量（兆瓦时）': 'riqianQuantityFixAll', 
#'日前修正电价（元/兆瓦时）': 'riqianPriceFix',  #redudant
'实时全网不平衡电量（兆瓦时）': 'shishiQuantityUnbalanceAll', 
#'实时不平衡电价（元/兆瓦时）': 'shishiPriceUnbalance',  #redudant

#'中长期日分解出力（兆瓦）': 'midlongPowerDecomposeStation', #duplicate
#'日前出清电力（兆瓦）': 'riqianPowerClean',  #duplicate
#'日前交易电价（元/兆瓦时）': 'riqianPriceClean', #duplicate
#'实时出清电力（兆瓦）': 'shishiPowerClean', #duplicate
#'实时交易电价（元/兆瓦时）': 'shishiPriceClean', #duplicate

'气象实测': 'obs_true',
'并网实测': 'power_true', 
'气象预测': 'fore',
'并网预测': 'fore_power',

# derived columns are:  (redudant, can be computed from above whenever wanted)
'日前交易电量（兆瓦时）': 'riqianQuantityClean', 
'日前交易电费（元）': 'riqianIncomeClean', 
'日前修正电量（兆瓦时）': 'riqianQuantityFixStation', 
'日前修正电费（元）': 'riqianIncomeFix', 
'实时交易电量（兆瓦时）': 'shishiQuantityClean', 
'实时交易电费（元）': 'shishiIncomeClean', 
'实时不平衡电量（兆瓦时）': 'shishiQuantityUnbalanceStation', 
'实时不平衡电费（元）': 'shishiIncomeUnbalance', 
'现货结算电量（兆瓦时）': 'spotQuantityClean', 
'现货结算电价（元/兆瓦时）': 'spotPriceClean', 
'现货结算电费（元）': 'spotIncomeClean', 
}


