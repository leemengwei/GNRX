import os

json = {}

json['mode'] = 'Solar'
#json['mode'] = 'Wind'

json['feature_data_path'] = os.path.join('..', 'data', '%s_X'%json['mode'])
json['label_data_path'] = os.path.join('..', 'data', '%s_Y'%json['mode'])

json['code'] = 'GSDTLTCX'
#json['code'] = 'YNDFYK'

json['pred_start'] = '2022-01'
json['pred_end'] = '2022-01'


