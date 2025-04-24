import pandas as pd
import statsmodels.api as sm
import numpy as np
from matplotlib.pyplot import subplots
from statsmodels.datasets import get_rdataset 
import sklearn.model_selection as skm
from ISLP import load_data , confusion_table 
from sklearn.tree import (DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR, plot_tree , export_text)
from sklearn.metrics import (accuracy_score, log_loss) 
from sklearn.ensemble import \
(RandomForestRegressor as RF, GradientBoostingRegressor as GBR)
from ISLP.bart import BART
from ISLP.models import (ModelSpec as MS, summarize)

weather = pd.read_csv("all_year.csv", usecols=['Total Rainfall', 'Mean Relative Humidity', 'Atmospheric Pressure', 'Wind Direction', 'Min Temp', 'Is_anomaly'])
weather = weather.dropna()
allvars = weather.columns.drop(['Is_anomaly'])
design = MS(allvars)
X = design.fit_transform(weather)
y = weather['Is_anomaly']
glm = sm.GLM(y, X, family = sm.families.Binomial())
results = glm.fit()
print(summarize(results))