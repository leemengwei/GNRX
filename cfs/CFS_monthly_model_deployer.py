import os,sys
import pandas as pd
from IPython import embed
from xgboost.sklearn import XGBRegressor
import pickle
import matplotlib.pyplot as plt
import dependency_misc 
from CFS_monthly_model_trainer import Trainer
import numpy as np

class Deployer(Trainer):
    def __init__(self, config_json):
        super().__init__(config_json)
        self.meteor_col_name = 'Downward_Short_Wave_Radiation_Flux' if self.mode=='Solar' else '10m_Wind'

    def load_and_run_model(self):
        print('Deploying...%s, %s'%(self.mode, self.code))
        model = pickle.load(open(os.path.join('..', 'output', '%s.pkl'%self.code), 'rb'))
        Y = self.X.copy()
        Y['predict_power'] = model.predict(self.X)
        return model, Y

    def get_output(self):
        self.subsitute_year = str(max(self.pred_Y.index.year)-1)
        powers = []
        meteors = []
        for i in self.predict_period:
            if i in self.pred_Y.index:
                # take lastest prediction
                powers.append(self.pred_Y.loc[i, 'predict_power'][-1])
                meteors.append(self.pred_Y.loc[i, self.meteor_col_name][-1])
            else:
                print(i)
                subsitute_i = pd.to_datetime('-'.join([self.subsitute_year, str(i.month)]))
                powers.append(self.pred_Y.loc[subsitute_i, 'predict_power'].mean())
                meteors.append(self.pred_Y.loc[subsitute_i, self.meteor_col_name].mean())
        powers = dict(zip(self.predict_period.strftime('%Y-%m'), powers))
        meteors = dict(zip(self.predict_period.strftime('%Y-%m'), meteors))
        return powers, meteors
 
    def eval_model(self):
        '''To be Implemented'''
        true = np.ravel(self.labels.reindex(self.output_powers.keys())['power'].values)
        pred = np.ravel(list(self.output_powers.values()))
        scores = {}
        scores['monthly_score'] = np.nanmean(np.clip(1 - np.abs(true - pred)/true, 0, 1))
        scores['yearly_score'] = 1 - (np.nansum(true) - np.nansum(pred))/np.nansum(true)
        return scores

    def run(self):
        self.datasets, self.X, self.Y = self.prepare_dataset()
        self.model, self.pred_Y = self.load_and_run_model()
        self.output_powers, self.output_meteors = self.get_output()
        self.scores = self.eval_model()
        if config_json.get('VISUALIZATION'):
            self.analayzer_helper()

    def analayzer_helper(self):
        ax = self.pred_Y['predict_power'].plot.line(label='predict power')
        ax = self.pred_Y[self.meteor_col_name].plot.line(label=self.meteor_col_name, ax=ax)
        self.labels.plot.line(label='true', ax=ax)
        plt.show()

def main(config_json):
    trainer = Deployer(config_json)
    trainer.run()
    print(trainer.code, trainer.scores)

if __name__ == '__main__':
    config_json = dependency_misc.arg_parse()
    main(config_json)

