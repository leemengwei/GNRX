import pulp
import pandas as pd
import sys,os
from IPython import embed
import config
import numpy as np
import dependency_compute_income

class StrategyAgent(object):
    '''The StrategyAgent will put its strategic declaration to its strategic_fore_power column for later income-computing'''
    def __init__(self, cap, data, midlong_price, benchmark_price, farm_loss, yonghu=[]):
        self.cap = cap
        self.data = data
        self.midlong_price = midlong_price
        self.benchmark_price = benchmark_price
        self.farm_loss = farm_loss
        self.yonghu = yonghu

    def do_strategy(self, name):
        # strategy of different scenarios:
        print('--using strategy: %s'%name)
        if name == 'origin':
            new_declaration = self.no_strategy() 
        elif name == 'yonghu':
            new_declaration = self.user_strategy()
        elif name == 'riqian_minimum':
            new_declaration = self.strategy_riqian_minimum()
        elif name == 'riqian_maximum':
            new_declaration = self.strategy_riqian_maximum()
        elif name == 'use_pulp':
            new_declaration = self.strategy_use_pulp() 
        else:
            print("Unkown strategy: %s"%name)
            sys.exit()
        # Will later put into use
        return new_declaration
    
    def no_strategy(self):
        '''blind strategy'''
        new_declaration = self.data['fore_power']   
        return new_declaration

    def user_strategy(self):
        '''blind strategy'''
        new_declaration = self.data['fore_power'].copy()
        if len(self.yonghu) > 0:
            new_declaration[-len(self.yonghu):] = self.yonghu
        else:
            pass
        return new_declaration

    def strategy_riqian_minimum(self):
        '''blind strategy'''
        new_declaration = 0
        return new_declaration

    def strategy_riqian_maximum(self):
        '''blind strategy'''
        new_declaration = self.cap
        return new_declaration

    def strategy_use_pulp(self):
        '''bright strategy'''
        # use pulp package:
        # step1, define problem:
        self.problem = pulp.LpProblem(name='spot_income_optimizer', sense=pulp.const.LpMaximize)
        # step2, set variables:
        X = []
        for i in range(0, len(self.data)):
            X.append(pulp.LpVariable('x%s'%i, lowBound=0, upBound=self.cap))
        # step3, set objective:
        self.data['strategic_fore_power'] = X  # Note: this will later serve as follows: fore_power --> midlong_compose --> riqian_clean
        IncomeCalculator = dependency_compute_income.MengXiIncomeCalculator(self.data, self.midlong_price, self.benchmark_price, self.farm_loss)
        IncomeCalculator.compute_income()
        z = IncomeCalculator.income_all
        # step4, define constraints:
        constraints = []
        if len(config.solar_hours)>0 and len(X)==96:
            for idx,x in enumerate(X): 
                if int(str(x).strip('x'))<config.solar_hours[0]*4 or int(str(x).strip('x'))>config.solar_hours[-1]*4:
                    constraints.append(X[idx] == 0) 
    
        # solve:
        for constraint in constraints:
            self.problem += constraint
        self.problem += z
        pulp.LpSolverDefault.msg=False
        status = pulp.LpStatus[self.problem.solve()]
        result = pulp.value(self.problem.objective)
        new_declaration = self._parse_pulp_solution()
        return new_declaration

    def _parse_pulp_solution(self):
        '''Mind param sequence'''
        try:
            seq = np.argsort(np.array([i.strip('x') for i in str(self.problem.variables())[1:-1].split(', ')], dtype=int))
        except Exception as e:
            print('Dummy solving variables, did you put columns in positions?', e)
        sorted_variables = np.array(self.problem.variables())[seq].tolist()
        solution = []
        for idx,var in enumerate(sorted_variables):
            solution.append(var.varValue)
        solution = pd.DataFrame(solution, index=self.data.index) 
        return solution




