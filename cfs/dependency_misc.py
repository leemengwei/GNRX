import argparse
import config_json

def arg_parse():
    '''In charge of getting param in config file and cmd line'''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--VISUALIZATION', "-V", action='store_true', default=False)
    args = parser.parse_args()
    json = config_json.json
    json['VISUALIZATION'] = args.VISUALIZATION
    return json

