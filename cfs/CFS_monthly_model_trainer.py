import os,sys
import pandas as pd
from IPython import embed
from xgboost.sklearn import XGBRegressor
import pickle
import matplotlib.pyplot as plt
import dependency_misc 

class Trainer(object):
    def __init__(self, config_json):
        self.code = config_json['code']
        self.mode = config_json['mode']
        self.input_data_at = os.path.join(config_json['feature_data_path'], '%s_%s_Monthly.csv'%(self.code, self.mode))
        self.label_data_at = os.path.join(config_json['label_data_path'], 'monthly_%s.csv'%self.code)
        self.predict_period = pd.date_range(config_json['pred_start'], config_json['pred_end'], freq='MS')

    def get_CFS_data(self):
        cfs_data = pd.read_csv(self.input_data_at, index_col=0)
        cfs_data.index = pd.to_datetime(cfs_data['year_predict'].astype(str)+'-'+cfs_data['month_predict'].astype(str))
        return cfs_data
 
    def get_label_data(self):
        label_data = pd.read_csv(self.label_data_at, index_col=0)
        label_data.index = pd.to_datetime(label_data.index, format='%Y%m')
        if 'Deployer' in str(self.__class__):
            pass  # Leave all label data for deployer
        else:
            label_data = label_data.loc[set(label_data.index) - set(self.predict_period)].sort_index()  # leave test part along
        return label_data

    def prepare_dataset(self):
        self.CFS = self.get_CFS_data()
        self.labels = self.get_label_data()
        datasets = pd.merge(self.CFS, self.labels, left_on=self.CFS.index, right_on=self.labels.index, left_index=True)
        datasets = datasets.drop(['key_0', 'year_start', 'year_predict'], axis=1)  # years are not factor needed neither in phase of train nor test.
        Y = datasets['power']
        X = datasets.drop(['wind', 'power'], axis=1)
        return datasets, X, Y

    def train_and_save_model(self):
        print('Training...%s, %s'%(self.mode, self.code))
        self.model = XGBRegressor()
        self.model.fit(self.X, self.Y)
        pickle.dump(self.model, open(os.path.join('..', 'output', '%s.pkl'%self.code), 'wb'))
        print('Model saved...')

    def run(self):
        self.datasets, self.X, self.Y = self.prepare_dataset()
        self.train_and_save_model()
        if config_json.get('VISUALIZATION'):
            self.analayzer_helper()

    def _get_prediction_by_startup_time(self, year, month, day):
        prediction = self.CFS[(self.CFS[['year_start', 'month_start', 'day_start']] == [int(year), int(month), int(day)]).sum(axis=1) == 3]
        return prediction
    
    def _get_prediction_by_prediction_time(self, year, month):
        startup_month = int(month)-1 if month!=1 else 12
        startup_year = year if month!=1 else year-1
        prediction = self.CFS[(self.CFS[['year_start', 'month_start', 'year_predict', 'month_predict']] == [int(startup_year), int(startup_month), int(year), int(month)]).sum(axis=1) == 4].iloc[0]
        return prediction

    def analayzer_helper(self):
        self.datasets.plot.line(title='Meteorilogical Uncertianty')
        plt.show()
        plt.plot(self.model.predict(self.X), label='predict')
        plt.plot(self.Y.values, label='label')
        plt.title("Predictions VS labels")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    config_json = dependency_misc.arg_parse()
    trainer = Trainer(config_json)
    trainer.run()

