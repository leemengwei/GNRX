import os,sys

# Before all you should know:
'''
Data description: 
1, all pred data are given in D-1 day;
2, trading takes place in D day;
3, gt data are given in D+1 day;
4, cleaning data (closure of trading) are given in D+4 day.
'''

# Configurations for predictive models: (!VITAL! Do Not Change unless you known what you're doing)
'''
Description of price & power model:
*Both riqian clean price & power are: 
1, given in D+4 while produced in [D-1] day by mechanism of market;
2, thus features training all kinds of riqian models should be [predictive features (D-1)];
In contrast, for shishi models:
*Both shishi clean price & power are: 
1, given in D+4 while produced in D day by mechanism of market;
2, thus features training all kinds of shishi models should be groud truth (gt), YET we won't have gt-data for shishi prediction----may use riqian_pred instead?;
'''

#1.1, for build riqian price model:  
#Note: may add daytime feature
riqian_price_model_features = [
'powerNeed_pred',
'powerOut_pred',
'powerWind_pred', 
'powerSolar_pred', 
'powerNew_pred', 
'posBackup', 
'negBackup',

'fore',
'fore_power', 
]

riqian_price_model_outputs = [
'riqianPriceClean'
]

#1.2, for build riqian power model:
riqian_power_model_features = [
'powerNeed_pred',
'powerOut_pred',
'powerWind_pred', 
'powerSolar_pred', 
'powerNew_pred', 
'posBackup', 
'negBackup',

'fore',
'fore_power', 
]

riqian_power_model_outputs = [
'riqianPowerClean', 
]

# Note: for shishi models 2.1&2.2, we still use suffix_pred since we won't have _gt data in real use)
#2.1, for build shishi price model:
#shishi_price_model_features = [
#'powerNeed_gt', 
#'powerOut_gt',
#'powerWind_gt', 
#'powerSolar_gt', 
#'powerNew_gt', 
#'posBackup', 
#'negBackup',
#
#'obs_true',   
#'power_true', 
#]
shishi_price_model_features = riqian_price_model_features 

shishi_price_model_outputs = [
'shishiPriceClean'
]

#2.2, for build shishi power model:
#shishi_power_model_features = [
#'powerNeed_gt',
#'powerOut_gt',
#'powerWind_gt', 
#'powerSolar_gt', 
#'powerNew_gt', 
#'posBackup', 
#'negBackup',
#
#'obs_true',
#'power_true', 
#]
shishi_power_model_features = riqian_power_model_features

shishi_power_model_outputs = [
'shishiPowerClean', 
]


#3, to summarize:
predictive_model_dicts = {
'riqianPowerClean': {'in': riqian_power_model_features, 'out': riqian_power_model_outputs},
'riqianPriceClean': {'in': riqian_price_model_features, 'out': riqian_price_model_outputs},
'shishiPowerClean': {'in': shishi_power_model_features, 'out': shishi_power_model_outputs},
'shishiPriceClean': {'in': shishi_price_model_features, 'out': shishi_price_model_outputs}
}




# Configurations for script (where you can modify if you want):
# train param:
val_days = 3 
feature_shift_steps = 0

# strategy constraints:
solar_hours = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#solar_hours = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# misc:
VISUALIZATION = False


