import pandas as pd
from IPython import embed
import sys,os
import glob
import datetime
import numpy as np
import config
# personal dependencies
import dependency_data_labour
import dependency_make_strategy
import dependency_compute_income
import dependency_misc
import matplotlib.pyplot as plt
import trading_agent_trainer
import pickle

class ModelRunner(object):
    def __init__(self, req_json, data):
        print("\n-------Comes to deploying part---------")
        self.req_json = req_json
        self.phase = req_json['mode'].upper()
        self.cap = req_json['cap']
        self.code = req_json['farm_code']
        self.yonghu = req_json['yonghu_short']
        self.data = data
        self.model_names = config.predictive_model_dicts.keys()
        self.midlong_price = req_json['midlong_price']   
        self.benchmark_price = req_json['benchmark_price']   
        self.farm_loss = req_json['farm_loss']    

        self.strategy_names = ['origin', 'yonghu', 'riqian_minimum', 'riqian_maximum', 'use_pulp']
        self.models = {}
        self.incomes = {}
        self.declarations = {}
        self.additional_income = {}
        self.calculated_data = {}

    def _data_pipelines(self, model_name):
        '''just use pipeline from Trainer, then self.dataset will be ready'''
        trading_agent_trainer.ModelTrainer._data_pipelines(self, model_name)

    def _load_models(self):
        '''load trained models & foward them to get output'''
        for model_name in self.model_names:
            self.models[model_name] = pickle.load(open('../output/%s_%s_model.pkl'%(self.code, model_name), 'rb'))
            print('model: %s loaded.'%model_name)

    def _forward_model(self, model_name):
        self._data_pipelines(model_name)  # get data for [self] using TRAINER class.
        out = pd.Series(self.models[model_name].predict(self.X), index=self.X.index)  
        if model_name == 'riqianPowerClean':  # riqianPowerClean models intend to get [clean ratio] rather than [actual clean value] with respect to fore_power, such that this ratio will not be impacted when later strategy works on riqian fore_power. Yet leave [shishi] alone since no shishi strategy will be performed.
            tmp = out/self.X['fore_power']
            tmp[np.where(np.abs(tmp)>1e10)[0]] = 1
            self.data['%s_ratio'%model_name] = tmp
        else:
            self.data['%s_pred'%model_name] = out

    def _forward_models(self):
        '''Implemented FOR IncomeCalculator'''
        for model_name in self.model_names:
           self._forward_model(model_name)
     
    def _run_strategy(self):
        '''Note: focus only on riqian declaration'''
        for strategy_name in self.strategy_names:
            # Step1, do strategies
            Strategy = dependency_make_strategy.StrategyAgent(self.cap, self.data, self.midlong_price, self.benchmark_price, self.farm_loss, self.yonghu)    # FYI: the reason we give this Agent [price info] is that in Pulp case, Agent will have to calculate income!
            new_declaration = Strategy.do_strategy(strategy_name)
            self.data['strategic_fore_power'] = new_declaration
            # Step2, calculate strategies' income
            IncomeCalculator = dependency_compute_income.MengXiIncomeCalculator(self.data, self.midlong_price, self.benchmark_price, self.farm_loss)
            IncomeCalculator.compute_income()
            # Summarize on strategy income
            if strategy_name == 'origin':income_all_origin=IncomeCalculator.income_all
            improvement = (IncomeCalculator.income_all - income_all_origin)/np.abs(income_all_origin)*100
            print('Case: %s, given: %s(point:%s), improved: %s%%'%(strategy_name, IncomeCalculator.income_all, IncomeCalculator.count, improvement))
            if config.VISUALIZATION:
                dependency_misc.plot_incomes(IncomeCalculator.income_parts, strategy_name)
            # record things we needed
            self.incomes[strategy_name] = IncomeCalculator.income_all
            self.declarations[strategy_name] = np.ravel(new_declaration)
            self.additional_income[strategy_name] = income_all_origin - IncomeCalculator.income_all
            self.calculated_data[strategy_name] = IncomeCalculator.data

    def _load_strategy_dependencies(self):
        '''twins of [build_strategy_dependencies] in Trainer'''
        self.errors_dict = pickle.load(open('../output/strategy_dependencies.pkl', 'rb'))
        self.riqianPriceClean_confidence = self.errors_dict['riqianPriceClean_confidence']
        self.shishiPriceClean_confidence = self.errors_dict['shishiPriceClean_confidence']
        self.riqianPowerClean_confidence = self.errors_dict['riqianPowerClean_confidence']
        self.shishiPowerClean_confidence = self.errors_dict['shishiPowerClean_confidence']
    
    def run(self):
        self._load_models() 
        self._forward_models() 
        self._run_strategy()
        self._load_strategy_dependencies()
        self.rep_json = dependency_misc.form_rep(self)
        if config.VISUALIZATION:
            self.data['strategic_fore_power'].plot.line(title='strategic riqian_fore_power')
            plt.show()

if __name__ == '__main__':
    # config
    config = dependency_misc.arg_parse() 

    # req is given from C++ in practice
    req_json = dependency_misc.get_req('deploy')
     
    # get data
    raw_data = dependency_data_labour.mengxi_raw_data_prepare(req_json)
    #raw_data = raw_data[req_json['wanted_day']]

    # deploy time
    Runner = ModelRunner(req_json, raw_data)
    Runner.run()



