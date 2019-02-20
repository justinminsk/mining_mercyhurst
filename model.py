import logging
import operator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz


logging.basicConfig(filename='model.txt', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

def get_data():
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
    return x_train, x_test, y_train, y_test

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_data()
    params = {"max_depth" : max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
    grid_search_cv = RandomizedSearchCV(DecisionTreeRegressor(random_state=1), params, n_jobs=1, verbose=1)
    grid_search_cv.fit(x_train, y_train)
    logging.info("Decision Tree Grid Search:")
    logging.info(grid_search_cv.best_estimator_)
    logging.info(" ")

    dtr = grid_search_cv.best_estimator_
            
    dtr.fit(x_train, y_train)
    fi_dict = dict(zip(x_train.columns, dtr.feature_importances_))
    fi_dict_1 = { k:v for k, v in fi_dict.items() if v }
    fi_dict_sorted = sorted(fi_dict_1.items(), key=lambda kv: kv[1], reverse=True)
    logging.info(fi_dict_sorted)
    print(dtr.score(x_test, y_test))

    # Create the random grid
    params = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    grid_search_cv = RandomizedSearchCV(RandomForestRegressor(random_state=1), params, n_jobs=1, verbose=1)
    grid_search_cv.fit(x_train, y_train)
    logging.info("Random Forest Grid Search:")
    logging.info(grid_search_cv.best_estimator_)
    logging.info(" ")

    rfr = grid_search_cv.best_estimator_
            
    rfr.fit(x_train, y_train)
    fi_dict = dict(zip(x_train.columns, rfr.feature_importances_))
    fi_dict_1 = { k:v for k, v in fi_dict.items() if v }
    fi_dict_sorted = sorted(fi_dict_1.items(), key=lambda kv: kv[1], reverse=True)
    logging.info(fi_dict_sorted)
    print(rfr.score(x_test, y_test))

