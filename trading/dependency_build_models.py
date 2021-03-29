from sklearn.linear_model import LinearRegression, LassoCV, Ridge, Lasso
from xgboost.sklearn import XGBRegressor
from xgboost import plot_importance
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import explained_variance_score as EVS
from IPython import embed
import matplotlib.pyplot as plt 

def train_model(X_train, Y_train, VISUALIZATION=False, mode_info=''):
    model = XGBRegressor()
    model.fit(X_train, Y_train)
    
    predicted_train = model.predict(X_train)
    train_mae = MAE(Y_train, predicted_train)
    train_evs = EVS(Y_train, predicted_train)
    print('--train mae: %s, evs: %s'%(train_mae, train_evs))
    if VISUALIZATION:
        plot_importance(model)
        plt.show()
        plt.plot(X_train.index, predicted_train, label='train predicted', alpha=0.5)
        plt.plot(Y_train, label='train label', alpha=0.5)
        plt.title('Train: %s model'%mode_info)
        plt.legend()
    return model

def eval_model(model, X_val, Y_val, VISUALIZATION=False, mode_info=''):
    predicted_val = model.predict(X_val)
    val_mae = MAE(Y_val, predicted_val)
    val_evs = EVS(Y_val, predicted_val)
    print('--val mae: %s, evs: %s'%(val_mae, val_evs))
 
    # plot:
    if VISUALIZATION:
        plt.plot(X_val.index, predicted_val, label='test predicted')
        plt.plot(Y_val, label='test label')
        plt.title('Eval: %s model'%mode_info)
        plt.legend()
        plt.show()
    return val_evs


