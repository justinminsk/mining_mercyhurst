import logging
import operator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz


logging.basicConfig(filename='model.txt', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

df = pd.read_parquet("processed_tweets.parquet")

print(df.head())

df = df.drop(['time_since_col', 'day_col'], axis=1)

x_data = df.drop(["interaction_target"], axis=1)
y_data = df.interaction_target.values

print(x_data.shape)
print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10)

print("x Train:", x_train.shape)
print("y Trian", y_train.shape)

dtr = DecisionTreeRegressor(criterion='mse', max_depth=24, max_features=None,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           presort=False, random_state=1, splitter='best')
           
dtr.fit(x_train, y_train)
fi_dict = dict(zip(x_train.columns, dtr.feature_importances_))
fi_dict_1 = { k:v for k, v in fi_dict.items() if v }
fi_dict_sorted = sorted(fi_dict_1.items(), key=lambda kv: kv[1], reverse=True)
logging.info(fi_dict_sorted)
print(dtr.score(x_test, y_test, ))
