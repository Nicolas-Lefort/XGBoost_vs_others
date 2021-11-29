# xgboost classification
# https://www.kaggle.com/shivamb/machine-predictive-maintenance-classification
# Machine Predictive Maintenance Classification Dataset
# improvements to explore: stratified split or undersampling

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from utils import extract, metrics, plot_correlation
from param_grid_search import grid_matrix

# import data
df = pd.read_csv('predictive_maintenance.csv')
# define target feature
target = 'Target'
# remove irrelevant features
df.drop(columns=["UDI","Failure Type"], inplace=True)
# keep where target not NaN
df = df[df[target].notna()]
# show number of counts per label
print("label counts :\n", df[target].value_counts())
# lower all string
df = df.applymap(lambda s: s.lower() if type(s) == str else s)
# remove features > 50 % missing values
s_null = df.isnull().sum()
low_feat = s_null[s_null > 0.5 * len(df)].index
df.drop(columns=low_feat.to_list(), inplace=True)
# extract data columns per type
numeric, categorical, ordinal, binary, label = extract(df.copy(), ordinal_features=None, target=target)
# plot correlations when required
#plot_correlation(df[numeric+[target]], target, top_n=20)
# sub pipeline numeric
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())])
# sub pipeline categorical
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
# sub pipeline ordinal
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('int', OrdinalEncoder())])
# sub pipeline binary
binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
# map data types to their pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric),
        ("cat", categorical_transformer, categorical),
        ("ord", ordinal_transformer, ordinal),
        ("bin", binary_transformer, binary)])
# define x and y
X, y = df.drop(columns=target), df[target]
# Preprocess labels
Le = LabelEncoder()
y_enc = Le.fit_transform(df[target])
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42) # shuffle True default
# initialize empty dataframe
results = pd.DataFrame()
#print(len(set(y_test)))
for param_grid, model, name in grid_matrix():
    # define main pipe main pipe
    mainpipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        (name, model)])
    # rename hyperparameter to fit grid search
    param_grid = {name+'__'+k:v for k, v in param_grid.items()}
    # create search object
    grid = RandomizedSearchCV(mainpipe,
                      param_distributions=param_grid,
                      scoring='accuracy',
                      n_iter=10,
                      n_jobs=-1)
    start = time.time()
    print("Fitting model " + name + "...")
    # fit data
    model = grid.fit(X_train, y_train)
    # store exec. time
    delta_t = round(time.time() - start, 2)
    # predict test data (and train data to check overfitting)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # calculate metrics
    data = metrics(y_train, y_train_pred, y_test, y_test_pred, delta_t)
    # store into dataframe
    if len(results)==0:
        results = pd.DataFrame(data, index=[name])
    else:
        results = results.append(pd.DataFrame(data, index=[name]))
    # show score
    print(name + " score :" + str(model.best_score_))
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(name)
    plt.savefig('conf_matrix_'+name+'.png')

print(results)
results.to_excel('results.xlsx', index=True)



