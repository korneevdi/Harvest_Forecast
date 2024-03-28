# import seaborn as sns
import warnings
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
# from preprocess import Preprocess
from sklearn.feature_selection import mutual_info_regression, RFECV
from sklearn.model_selection import TimeSeriesSplit 

class FeatureSelection:
    def __init__(self, path_to_file):
        self._df = pd.read_csv(path_to_file)


    def show_mutal_importance(self):
        X = self._df.drop(columns=['date','target'])
        y = self._df['target']
        X = X.ffill()
        importances = mutual_info_regression(X, y, random_state=42)
        # Где data - ваш датасет; X, y – входные и выходные данные соответственно
        feature_importances = pd.Series(importances, X.columns)
        feature_importances.plot(kind='barh', color='teal', figsize=(20, 20))
        plt.show()
        return feature_importances

    @property
    def df(self):
        return self._df




def build_models(df_for_model, models, plot_result=True,   ksplit=3):
    df = df_for_model.sort_values('date').copy()
    y = df['target']
    X = df.drop(columns=['target', 'date'])
    
    if plot_result:
        fig, axes = plt.subplots(ksplit, len(models), figsize=(15, 20)) 
    scores = {}
    
    folds = TimeSeriesSplit(n_splits=ksplit)  
    for i, (train_index, test_index) in enumerate(folds.split(df)):
        scores[i] = {}
        for ind, model in enumerate(models):
            m = str(model)
            if 'ElasticNetCV' in m:
                name = 'ElasticNetCV'
                model.fit(X.iloc[train_index], y.iloc[train_index])
                pred = model.predict(X.iloc[test_index])
            else:
                name = m[:m.index('(')] 
                model.fit(X.iloc[train_index], y.iloc[train_index])
                pred = model.predict(X.iloc[test_index])
                
            temp = X.iloc[test_index].copy()
            temp['pred_' + name] = pred
            y_true = y.iloc[test_index]
            
            scores[i].update({'RMSE_'+ name: round(np.sqrt(mean_squared_error(y_true, temp['pred_' + name])), 4),
                           'MAPE_' + name: round(mean_absolute_percentage_error(y_true, temp['pred_' + name]), 4),
                           'R2_' +  name: round(r2_score(y_true, temp['pred_' + name]), 4),
    
                          })
    
            if plot_result:
                axes[i][ind].set_title(name)
    
                temp['date'] = df.iloc[test_index]['date'].values
                sns.lineplot(data = df.iloc[train_index[-10:]], x='date', y='target', ax=axes[i][ind], label='train',color="b")
                sns.lineplot(data = df.iloc[test_index], x='date', y='target', ax=axes[i][ind], label='Validation',color="g")
                sns.lineplot(data = temp, x='date', y='pred_' + name, ax=axes[i][ind], label='Prediction', color="r")
    
    return pd.DataFrame.from_dict(scores, orient='index')

