import pandas as pd
from IPython import embed
import sys,os
import glob
import datetime
import numpy as np
import config
# personal dependencies
import dependency_data_labour
import dependency_build_models
import dependency_misc
import matplotlib.pyplot as plt
import pickle 

class ModelTrainer(object):
    def __init__(self, req_json, data):
        print("\n-------Comes to training part---------")
        self.req_json = req_json
        self.phase = req_json['mode'].upper()
        self.cap = req_json['cap']
        self.code = req_json['farm_code']
        self.data = data
        self.errors_dict = {}
        self.model_names = config.predictive_model_dicts.keys()

    def _data_pipelines(self, model_name):
        '''This pipeline will be used both in phase of train&deploy. What it does: shift-features, dropna(s), split train-val'''
        # before train (deploy), more features might be added: (feature column name will be modified as well)
        data, input_features, output_features = dependency_data_labour.more_shift_features(self.data, model_name)
        # take these data:
        if self.phase == 'TRAIN':
            dataset = data.dropna(subset=input_features + output_features)  # phase of train, must drop if input/output features don't exists
        elif self.phase == 'DEPLOY':
            dataset = data.dropna(subset=input_features)  # phase of deploy, only drop if input features don't exists
        else:
            print('Unkown phase: %s, please give [train] or [deploy] in json!'%self.phase)
        # split dataset:
        train_dataset, val_dataset = dependency_data_labour.split_dataset(dataset)
        self.X_train = train_dataset[input_features]
        self.Y_train = train_dataset[output_features]
        self.X_val = val_dataset[input_features]
        self.Y_val = val_dataset[output_features]
        self.X = self.X_train.append(self.X_val)
        self.Y = self.Y_train.append(self.Y_val)
   
    def _build_model_types(self, model_name):
        '''return model predicting clean-price of riqian | shishi & model predicting clean-power of riqian | shishi'''
        self.model = dependency_build_models.train_model(self.X_train, self.Y_train, config.VISUALIZATION, model_name)
        self.error = dependency_build_models.eval_model(self.model, self.X_val, self.Y_val, config.VISUALIZATION, model_name)
    
    def build_strategy_dependencies(self):
        '''save strategy dependencies'''
        pickle.dump(self.errors_dict, open('../output/strategy_dependencies.pkl', 'wb'))
        print("Pickle saved for strategy dependencies")
    
    def build_predictive_models(self):
        '''build models'''
        # sequentially build predictive models:
        for model_name in self.model_names:
            self._data_pipelines(model_name)
            self._build_model_types(model_name)
            self.errors_dict['%s_confidence'%model_name] = np.clip(self.error, -0.1, 1)
            # save each one:
            pickle.dump(self.model, open('%s/%s_%s_model.pkl'%(os.path.dirname(__file__).replace('/src', '/output'), self.code, model_name), 'wb'))
            print('model: %s saved.'%model_name)

    def run(self):
        '''train predictive models & save something (if needed) for strategy'''
        self.build_predictive_models()
        self.build_strategy_dependencies()
        self.rep_json = dependency_misc.form_rep(self)

if __name__ == '__main__':
    # config
    config = dependency_misc.arg_parse() 

    # req is given from C++ in practice
    req_json = dependency_misc.get_req('train')
     
    # get data
    raw_data = dependency_data_labour.mengxi_raw_data_prepare(req_json)

    # build models (price model & power model, strategy is given real time)
    Trainer = ModelTrainer(req_json, raw_data)
    Trainer.run()



