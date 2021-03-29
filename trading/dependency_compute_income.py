import time
import numpy as np
from IPython import embed
import pandas as pd
import abc   # abstract class

class BaseIncomeCalculator(metaclass=abc.ABCMeta):
    def __init__(self, data, midlong_price, benchmark_price, farm_loss, backtest_on_origin=False):
        self.data = self._pretreat(data)
        self.income_parts = {}
        # other configs:
        self.midlong_contract_price = midlong_price   # RMB/Mwh
        self.base_price = benchmark_price   #RMB/Mwh
        self.loss = farm_loss    # loss between shishi power clean & online power given is given externally

    def compute_income(self):   # this function is NOT abstract because it's overall income is always composed of three parts, whatever province we concern.
        self._market_mechanism()
        # computing every parts
        self._get_income_futures()
        self._get_income_spot()
        self._get_deviation_penalty()
        # all = Part1 + Part2 + Part3
        self.income_all_pd = self.income_future_market + self.income_spot_market + self.deviation_penalty
        self.income_all = np.nansum(self.income_all_pd)
        self.income_parts['income_future_market'] = self.income_future_market
        self.income_parts['income_spot_market'] = self.income_spot_market
        self.income_parts['deviation_penalty'] = self.deviation_penalty
        self.income_parts['income_midlong_contract'] = self.income_midlong_contract
        self.income_parts['income_base'] = self.income_base
        self.income_parts['income_riqian_market'] = self.income_riqian_market
        self.income_parts['income_shishi_market'] = self.income_shishi_market
        self.income_parts['income_share_market'] = self.income_share_market
        self.count = self.income_all_pd.count()
   
    @abc.abstractmethod
    def _market_mechanism(self):
        # To get data ready for later income computation, some columns need to be in their corresponding position
        pass

    @abc.abstractmethod
    def _get_income_futures(self):
        # Implemented with child class
        pass
 
    @abc.abstractmethod
    def _get_income_spot(self):
        # Implemented with child class
        pass

    @abc.abstractmethod
    def _get_deviation_penalty(self):
        # Implemented with child class
        pass

    def _pretreat(self, data):
         # Note: may not be avaliable, may predict them instead, but just fill them for now.
        # fill na for some colunms:
        data['midlongQuantityDecomposeAll'] = data['midlongQuantityDecomposeAll'].fillna(1.0) #will serve as denominator in share
        data['riqianQuantityFixAll'] = data['riqianQuantityFixAll'].fillna(0.0)
        data['shishiQuantityUnbalanceAll'] = data['shishiQuantityUnbalanceAll'].fillna(0.0)
        return data

class ShanXiIncomeCalculator(BaseIncomeCalculator):
    def _market_mechanism(self):
        pass
    def _get_income_futures(self):
        pass
    def _get_income_spot(self):
        pass
    def _get_deviation_penalty(self):
        pass

class GanSuIncomeCalculator(BaseIncomeCalculator):
    def _market_mechanism(self):
        pass
    def _get_income_futures(self):
        pass
    def _get_income_spot(self):
        pass
    def _get_deviation_penalty(self):
        pass

class MengXiIncomeCalculator(BaseIncomeCalculator):
    '''
    class function validity checked with FILE: 2020-12-10~20_SanHanGuangFu.xls
    this function is to compute total income given data with necessary columns like clean price, clean power etc.
    '''
    def _get_income_spot(self):
        '''Computing for riqian & shishi income: *(0.25) for 15min data'''
        self.income_riqian_market = (self.data['riqianPowerClean'] - self.data['midlongPowerDecomposeStation'])*(0.25) * self.data['riqianPriceClean']
        self.income_shishi_market = (self.data['shishiPowerClean'] - self.data['riqianPowerClean'])*(0.25) * self.data['shishiPriceClean']
       
        # Market Rules for income_riqian_fix & income_shishi_unbalance: 
        self.income_riqian_fix = (self.data['midlongPowerDecomposeStation']*(0.25))/self.data['midlongQuantityDecomposeAll'] * self.data['riqianQuantityFixAll'] * self.data['riqianPriceClean']
        self.income_shishi_unbalance = (self.data['midlongPowerDecomposeStation']*(0.25))/self.data['midlongQuantityDecomposeAll'] * self.data['shishiQuantityUnbalanceAll'] * self.data['shishiPriceClean']
        # spot income as:
        self.income_share_market = self.income_riqian_fix + self.income_shishi_unbalance
        self.income_spot_market = self.income_riqian_market + self.income_shishi_market + self.income_share_market
    
    def _get_income_futures(self):
        '''
        future market is composed of two parts:
        1, midlong contract quantity
        2, base quantity  
        '''
        self.midlong_contract_quantity = pd.Series(data=0.0, index=self.data.index, name='contract')  #Note: Mwh, contract quantity is given externally
        self.spot_quantity = (self.data['shishiPowerClean'] - self.data['midlongPowerDecomposeStation'])*(0.25) + (self.data['midlongPowerDecomposeStation']*(0.25))/self.data['midlongQuantityDecomposeAll'] * self.data['riqianQuantityFixAll'] + (self.data['midlongPowerDecomposeStation']*(0.25))/self.data['midlongQuantityDecomposeAll'] * self.data['shishiQuantityUnbalanceAll'] 
        self.online_quantity = self.data['shishiPowerClean']*(0.25) * (1.0-self.loss)  # Note: we can only assert quantity online is shishiClean (which is given externally)
        self.base_quantity = self.online_quantity - self.midlong_contract_quantity - self.spot_quantity
        # thus: 
        self.income_midlong_contract = self.midlong_contract_quantity * self.midlong_contract_price
        self.income_base = self.base_quantity * self.base_price
        self.income_future_market = self.income_base + self.income_midlong_contract
 
    def _get_deviation_penalty(self):
        self.deviation_penalty = pd.Series(data=0.0, index=self.data.index, name='deviation_penalty')    # penalty could be omitted. E.g every 10Mwh of deviation will cost 1000RMB*0.2*0.3=60RMB/Mwh
   
    def _market_mechanism(self):
        '''
        before computing any, put key columns in positions where they have to be (decided by market mechanism):
        According to interpretation of trading rules:
        1, forecast will serve as reference of midlong decomposition, thus: fore_power --> midlong_decompose --> riqian_clean;
        2, unkown riqian power clean given by: first by PowerCut, then by model (above clause 1);
        3, unknow riqian price clean given by: market mechanism, by model;
        4, unkown shishi power clean given by: market mechanism, by model;
        5, unkown shishi price clean given by: market mechanism, by model;
        (might be a little bit precision difference)
        # if backtest on origin, leave columns unchanged (to verify income with excel)
        # if not backtest on origin, get subsitute predictive columns
        '''
        def _market_mechanism_from_declaration_to_riqian_clean(fore_power, ratio_series):
            midlong_decompose = fore_power      # Rules No.1, Note a model is needed
            '''Note: Clean of riqian strategy will be propotional to Clean of raw'''
            riqian_clean = midlong_decompose * ratio_series 
            return midlong_decompose, riqian_clean
        
        # First get midlong_decompose & riqian_powerclean from  forecast power.
        self.data['midlongPowerDecomposeStation'], self.data['riqianPowerClean_pred'] = _market_mechanism_from_declaration_to_riqian_clean(self.data['strategic_fore_power'] , self.data['riqianPowerClean_ratio']) 
        # Then put them into position
        # When work on deploy time, neither CleanPower nor CleanPrice is available for tomorrow, thus:
        self.data['riqianPowerClean'] = self.data['riqianPowerClean_pred']
        self.data['riqianPriceClean'] = self.data['riqianPriceClean_pred'] 
        self.data['shishiPowerClean'] = self.data['shishiPowerClean_pred']
        self.data['shishiPriceClean'] = self.data['shishiPriceClean_pred']
        #embed()


