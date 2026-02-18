from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# these are all things that might be currently needed 
from statsforecast import StatsForecast
from functools import partial
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from utilsforecast.evaluation import evaluate
from sklearn.linear_model import Lasso, Ridge
from utilsforecast.feature_engineering import pipeline, trend , fourier
from utilsforecast.losses import rmse, mae, mape as _mape, mase, quantile_loss, mqloss
from statsforecast import StatsForecast
import datetime
from statsmodels.tsa.seasonal import STL, seasonal_decompose
import re
from statsforecast.utils import ConformalIntervals
from statsforecast.models import (
    HistoricAverage,
    Naive,
    RandomWalkWithDrift,
    SeasonalNaive,
    SklearnModel,
)
from utilsforecast.plotting import plot_series
class engine:
    def __init__(self,models_selection,steps,frequency,data):
        self.models_selection=models_selection
        self.steps=steps #h
        self.frequency=frequency
        self.data=data
    def create(self):
        train_features, valid_features = pipeline(
         self.data,
         features=[trend   ],
         freq=   self.frequency,
        h=self.steps,
        )
        self.tf=train_features
        self.vf=valid_features

       
    def train(self,option):
        if option==1:
            self.sf = StatsForecast(
            models=self.models_selection,
            freq=self.frequency,
            prediction_intervals=ConformalIntervals(n_windows=4, h=self.steps),
             n_jobs=-1,
            )
        else:
            
            self.sf = StatsForecast(
            models=self.models_selection,
            freq=self.frequency,
            n_jobs=-1,
             )
            
    def validation(self):
        self.cv_df=self.sf.cross_validation(
            df=self.data,
            h=self.steps,
            step_size=12,
            n_windows=2
        )
    def evaluate_cv(self,df,metric):
        models = df.columns.drop(['unique_id', 'ds', 'y', 'cutoff']).tolist()
        evals = metric(df, models=self.models_selection)
        evals['best_model'] = evals[models].idxmin(axis=1)
        return evals
    def predict(self):
   
        return self.sf
    
    