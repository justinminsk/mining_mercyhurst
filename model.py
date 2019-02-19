import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor


logging.basicConfig(filename='model.txt', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

df = pd.read_parquet("processed_tweets.parquet")

df = df.drop(['time_since', 'day'], axis=1)

x_data = df.drop(["interaction"], axis=1)
y_data = df.interaction.values

print(x_data.shape)
print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10)

print("x Train:", x_train.shape)
print("y Trian", y_train.shape)

rfr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
           oob_score=False, random_state=1, verbose=0, warm_start=False)
           
rfr.fit(x_train, y_train)
fi_dict = dict(zip(x_train.columns, rfr.feature_importances_))
logging.info({ k:v for k, v in fi_dict.items() if v })
logging.info(" ")
logging.info(rfr.decision_path(x_train))
print(rfr.score(x_test, y_test, ))
