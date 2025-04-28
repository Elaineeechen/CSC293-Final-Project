import numpy as np
import pandas as pd
import sklearn.model_selection as skm
from ISLP.models import ModelSpec as MS
from sklearn.tree import (DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR, plot_tree , export_text)
from sklearn.metrics import (accuracy_score, log_loss) 
from sklearn.ensemble import RandomForestClassifier as RFC
from ISLP.bart import BART
import matplotlib.pyplot as plt

# folder_path = "/Users/elainechen/Downloads"
# file_list = glob.glob(os.path.join(folder_path, "2023.csv"))
# header = pd.read_csv(file_list[0], skiprows=[0, 1, 3])
# rows = [pd.read_csv(f, skiprows = 4, header = None, names = header.columns, na_values = [], keep_default_na = False) for f in file_list[1:]]
# weather = pd.concat([header] + rows, ignore_index = True)
# weather = weather.dropna()

'''Bagging & Random Forest w/out Lag'''
weather = pd.read_csv("all_year.csv")
weather = weather.dropna()
model = MS(weather.columns.drop(['TIMESTAMP', 'Time of Daily Temp Max', 'Time of Min. Temp', 'Time of Max Wind Spd', 'Time of Min. Wind Spd.', 'Precipitation', 'Is_rain', 'Snow', 'Is_snow', 'Is_anomaly']), intercept=False)
D = model.fit_transform(weather)
feature_names = list(D.columns)
X = np.asarray(D)
# print(X.shape)
# print(weather['Is_anomaly'].shape)
(X_train, X_test, y_train, y_test) = skm.train_test_split(X, weather['Is_anomaly'], test_size = 0.3, random_state = 0)

# Bagging
bag_weather = RFC(max_features = X_train.shape[1], n_estimators = 500, random_state = 0) # Building 500 trees result in a similar MSE (0.04062) when compared to a default of building 100 trees (0.04278)
bag_weather.fit(X_train, y_train)
# ax = subplots(figsize=(8,8))[1] 
y_hat_bag = bag_weather.predict(X_test) 
# ax.scatter(y_hat_bag, y_test) 
classification_error = np.mean(y_hat_bag != y_test)
print(f"Bagging (w/out Lag) Classification Error: {classification_error}")
feature_imp = pd.DataFrame({'importance':bag_weather.feature_importances_}, index = feature_names)
print(feature_imp.sort_values(by = 'importance', ascending = False))
one_tree = bag_weather.estimators_[0]
plt.figure(figsize=(20, 10))
plot_tree(one_tree, feature_names=feature_names, filled=True, rounded=True, fontsize=10)
plt.show()
print()

# Random Forest
RF_weather = RFC(n_estimators = 500, random_state = 0) 
RF_weather.fit(X_train, y_train)
# ax = subplots(figsize=(8,8))[1] 
y_hat_rf = RF_weather.predict(X_test) 
# ax.scatter(y_hat_bag, y_test) 
classification_error = np.mean(y_hat_rf != y_test)
print(f"Random Forest (w/out Lag) Classification Error: {classification_error}")
feature_imp = pd.DataFrame({'importance':RF_weather.feature_importances_}, index = feature_names)
print(feature_imp.sort_values(by = 'importance', ascending = False))
print()


'''Bagging & Random Forest w/ Lag'''
weather['Is_rain_lag1'] = weather['Is_rain'].shift(1)
weather['Is_rain_lag2'] = weather['Is_rain'].shift(2)
weather['Is_rain_lag3'] = weather['Is_rain'].shift(3)
weather['Is_snow_lag1'] = weather['Is_snow'].shift(1)
weather['Is_snow_lag2'] = weather['Is_snow'].shift(2)
weather['Is_snow_lag3'] = weather['Is_snow'].shift(3)
weather = weather.dropna()
model = MS(weather.columns.drop(['TIMESTAMP', 'Time of Daily Temp Max', 'Time of Min. Temp', 'Time of Max Wind Spd', 'Time of Min. Wind Spd.', 'Precipitation', 'Is_rain', 'Snow', 'Is_snow', 'Is_anomaly']), intercept=False)
D = model.fit_transform(weather)
feature_names = list(D.columns)
X = np.asarray(D)
# print(X.shape)
# print(weather['Is_anomaly'].shape)
(X_train, X_test, y_train, y_test) = skm.train_test_split(X, weather['Is_anomaly'], test_size = 0.3, random_state = 0)

# Bagging
bag_weather = RFC(max_features = X_train.shape[1], n_estimators = 500, random_state = 0) # Building 500 trees result in a similar MSE (0.04062) when compared to a default of building 100 trees (0.04278)
bag_weather.fit(X_train, y_train)
# ax = subplots(figsize=(8,8))[1] 
y_hat_bag = bag_weather.predict(X_test) 
# ax.scatter(y_hat_bag, y_test) 
classification_error = np.mean(y_hat_bag != y_test)
print(f"Bagging (w/ Lag) Classification Error: {classification_error}")
feature_imp = pd.DataFrame({'importance':bag_weather.feature_importances_}, index = feature_names)
print(feature_imp.sort_values(by = 'importance', ascending = False))
print()

# Random Forest
RF_weather = RFC(n_estimators = 500, random_state = 0)
RF_weather.fit(X_train, y_train)
# ax = subplots(figsize=(8,8))[1] 
y_hat_rf = RF_weather.predict(X_test)
# ax.scatter(y_hat_bag, y_test) 
classification_error = np.mean(y_hat_rf != y_test)
print(f"Random Forest (w/ Lag) Classification Error: {classification_error}")
feature_imp = pd.DataFrame({'importance':RF_weather.feature_importances_}, index = feature_names)
print(feature_imp.sort_values(by = 'importance', ascending = False))
