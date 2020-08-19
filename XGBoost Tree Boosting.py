import numpy as np # Linear algebra.
import pandas as pd # Data processing.
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

#%%
data_train = pd.read_csv('/Users/carrotkr/Dropbox/House Prices - train.csv')
data_test = pd.read_csv('/Users/carrotkr/Dropbox/House Prices - test.csv')

#%% Missing data - train.
# Reference.
#   www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
print(data_train.isnull().sum().sort_values(ascending=False))
missing_data_train_sum = data_train.isnull().sum().sort_values(ascending=False)
missing_data_train_percent = (data_train.isnull().sum() / data_train.isnull().count())\
                        .sort_values(ascending=False)
missing_data_train = pd.concat([missing_data_train_sum, missing_data_train_percent],\
                               axis=1, keys=['missing_data', 'percent'])
print(missing_data_train)

print(missing_data_train['percent'] > 0.1)
print(missing_data_train[missing_data_train['percent'] > 0.1])
print(missing_data_train[missing_data_train['percent'] > 0.1].index)

data_train = data_train.drop(missing_data_train[missing_data_train['percent'] > 0.1].index, 1)
print(data_train.info())

#%% Missing data - test.
print(data_test.isnull().sum().sort_values(ascending=False))
missing_data_test_sum = data_test.isnull().sum().sort_values(ascending=False)
missing_data_test_percent = (data_test.isnull().sum() / data_test.isnull().count())\
                        .sort_values(ascending=False)
missing_data_test = pd.concat([missing_data_test_sum, missing_data_test_percent],\
                               axis=1, keys=['missing_data', 'percent'])
print(missing_data_test)

print(missing_data_test['percent'] > 0.1)
print(missing_data_test[missing_data_test['percent'] > 0.1])
print(missing_data_test[missing_data_test['percent'] > 0.1].index)

data_test = data_test.drop(missing_data_test[missing_data_test['percent'] > 0.1].index, 1)
print(data_test.info())

#%% Categorical feature - train.
from sklearn.preprocessing import LabelEncoder

ctg_feature_data_train = data_train.columns[(data_train.dtypes == object)].tolist()
print(ctg_feature_data_train)

# Reference [sklearn.preprocessing.LabelEncoder].
#   scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
for cols in ctg_feature_data_train:
    LabelEncoder().fit(list(data_train[cols].values))
    data_train[cols] = LabelEncoder().fit_transform(list(data_train[cols].values))
    
print(data_train.info())

#%% Categorical feature - test.
ctg_feature_data_test = data_test.columns[(data_test.dtypes == object)].tolist()
print(ctg_feature_data_test)

for cols in ctg_feature_data_test:
    LabelEncoder().fit(list(data_test[cols].values))
    data_test[cols] = LabelEncoder().fit_transform(list(data_test[cols].values))

print(data_test.info())

#%% Fill nan - train.
print(data_train.isnull().sum().sort_values(ascending=False))

print(data_train['GarageYrBlt'].mean())
data_train['GarageYrBlt'] = data_train['GarageYrBlt'].fillna(data_train['GarageYrBlt'].mean())

print(data_train['MasVnrArea'].mean())
data_train['MasVnrArea'] = data_train['MasVnrArea'].fillna(data_train['MasVnrArea'].mean())

print(data_train.isnull().sum().sort_values(ascending=False))

#%% Fill nan - test.
print(data_test.isnull().sum().sort_values(ascending=False))

print(data_test['GarageYrBlt'].mean())
data_test['GarageYrBlt'] = data_test['GarageYrBlt'].fillna(data_test['GarageYrBlt'].mean())

data_test['MasVnrArea'] = data_test['MasVnrArea'].fillna(data_test['MasVnrArea'].mean())
data_test['BsmtFullBath'] = data_test['BsmtFullBath'].fillna(data_test['BsmtFullBath'].mean())
data_test['BsmtHalfBath'] = data_test['BsmtHalfBath'].fillna(data_test['BsmtHalfBath'].mean())
data_test['TotalBsmtSF'] = data_test['TotalBsmtSF'].fillna(data_test['TotalBsmtSF'].mean())
data_test['GarageArea'] = data_test['GarageArea'].fillna(data_test['GarageArea'].mean())
data_test['BsmtUnfSF'] = data_test['BsmtUnfSF'].fillna(data_test['BsmtUnfSF'].mean())
data_test['BsmtFinSF1'] = data_test['BsmtFinSF1'].fillna(data_test['BsmtFinSF1'].mean())
data_test['GarageCars'] = data_test['GarageCars'].fillna(data_test['GarageCars'].mean())
data_test['BsmtFinSF2'] = data_test['BsmtFinSF2'].fillna(data_test['BsmtFinSF2'].mean())

print(data_test.isnull().sum().sort_values(ascending=False))

#%%
X_train = data_train.drop(['Id', 'SalePrice'], axis=1)

Y_train = data_train['SalePrice']
Y_train = Y_train.values.reshape(-1, 1)

X_test = data_test.drop(['Id'], axis=1)

#%%
from sklearn.preprocessing import StandardScaler

std_data_X = StandardScaler()
X_train = std_data_X.fit_transform(X_train)

std_data_Y = StandardScaler()
Y_train = std_data_Y.fit_transform(Y_train)

X_test = std_data_X.fit_transform(X_test)

#%%
from xgboost import XGBRegressor

xgb_model = XGBRegressor(colsample_bytree=0.2, learning_rate=0.01,\
                         max_depth=3, min_child_weight=1.5, n_estimators=5000,\
                         reg_alpha=0.75, reg_lambda=0.45, subsample=0.2)
xgb_model.fit(X_train,Y_train, verbose=False)

predict = xgb_model.predict(X_test)