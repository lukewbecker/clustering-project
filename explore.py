import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score

# modeling and evaluating
from sklearn.linear_model import LinearRegression, LassoLars
from sklearn.linear_model import TweedieRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
import acquire
import prepare
import wrangle


# Functions used in explore stage:

def elbow_plot(X_train_scaled, cluster_vars):
    # elbow method to identify good k for us, originally used range (2,20), changed for presentation
    ks = range(2,10)
    
    # empty list to hold inertia (sum of squares)
    sse = []

    # loop through each k, fit kmeans, get inertia
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_train_scaled[cluster_vars])
        # inertia
        sse.append(kmeans.inertia_)
    # print out was used for determining cutoff, commented out for presentation
    # print(pd.DataFrame(dict(k=ks, sse=sse)))

    # plot k with inertia
    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('Elbow method to find optimal k')
    plt.show()

def run_kmeans(X_train_scaled, X_train, cluster_vars, k, cluster_col_name):
    # create kmeans object
    kmeans = KMeans(n_clusters = k, random_state = 13)
    kmeans.fit(X_train_scaled[cluster_vars])
    # predict and create a dataframe with cluster per observation
    train_clusters = \
        pd.DataFrame(kmeans.predict(X_train_scaled[cluster_vars]),
                              columns=[cluster_col_name],
                              index=X_train.index)



def kmeans_transform(X_scaled, kmeans, cluster_vars, cluster_col_name):
    kmeans.transform(X_scaled[cluster_vars])
    trans_clusters = \
        pd.DataFrame(kmeans.predict(X_scaled[cluster_vars]),
                              columns=[cluster_col_name],
                              index=X_scaled.index)
    
    return trans_clusters


def get_centroids(kmeans, cluster_vars, cluster_col_name):
    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroids = pd.DataFrame(kmeans.cluster_centers_, 
             columns=centroid_col_names).reset_index().rename(columns={'index': cluster_col_name})
    
    return centroids

def split_scale(df):
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)
    
    # Assign variables
    X_train = train.drop(columns=['logerror'])
    X_validate = validate.drop(columns=['logerror'])
    X_test = test.drop(columns=['logerror'])
    X_train_explore = train

    # I need X_train_explore set to train so I have access to the target variable.
    y_train = train[['logerror']]
    y_validate = validate[['logerror']]
    y_test = test[['logerror']]

    # create the scaler object and fit to X_train (get the min and max from X_train for each column)
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X_train)

    # transform X_train values to their scaled equivalent and create df of the scaled features
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), 
                                  columns=X_train.columns.values).set_index([X_train.index.values])
    
    # transform X_validate values to their scaled equivalent and create df of the scaled features
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate),
                                    columns=X_validate.columns.values).set_index([X_validate.index.values])

    # transform X_test values to their scaled equivalent and create df of the scaled features   
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                                 columns=X_test.columns.values).set_index([X_test.index.values])

    # create the scaler object and fit to X_train (get the min and max from X_train for each column)
    scaler_explore = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X_train_explore)
    # transform X_train values to their scaled equivalent and create df of the scaled features
    X_train_explore_scaled = pd.DataFrame(scaler_explore.transform(X_train_explore), 
                                    columns=X_train_explore.columns.values).set_index([X_train_explore.index.values])
    
    return X_train, X_validate, X_test, X_train_explore, X_train_explore_scaled, y_train, y_validate, y_test, X_train_scaled, X_validate_scaled, X_test_scaled



def add_to_train_frame(cluster_col_name):
    # concatenate cluster id
    X_train = pd.concat([X_train, train_clusters], axis=1)

    # join on clusterid to get centroids
    X_train = X_train.merge(centroids, how='left', on=cluster_col_name).\
                    set_index(X_train.index)

    # concatenate cluster id
    X_train_scaled = pd.concat([X_train_scaled, train_clusters], 
                            axis=1)

    # join on clusterid to get centroids
    X_train_scaled = X_train_scaled.merge(centroids, how='left', on=cluster_col_name).\
                    set_index(X_train.index)

    # concatenate cluster id
    X_train_explore = pd.concat([X_train_explore, train_clusters], 
                            axis=1)

    # join on clusterid to get centroids
    X_train_explore = X_train_explore.merge(centroids, how='left', on=cluster_col_name).\
                    set_index(X_train.index)

    # concatenate cluster id
    X_train_explore_scaled = pd.concat([X_train_explore_scaled, train_clusters], 
                            axis=1)

    # join on clusterid to get centroids
    X_train_explore_scaled = X_train_explore_scaled.merge(centroids, how='left', on=cluster_col_name).\
                    set_index(X_train.index)
    
    return X_train, X_train_scaled, X_train_explore, X_train_explore_scaled


def add_clusters(train_clusters, centroids, X_train, X_train_scaled, cluster_col_name):
    # concatenate cluster id
    X_train2 = pd.concat([X_train, train_clusters], axis=1)

    # join on clusterid to get centroids
    X_train2 = X_train2.merge(centroids, how='left', 
                            on=cluster_col_name).\
                        set_index(X_train.index)
    
    # concatenate cluster id
    X_train_scaled2 = pd.concat([X_train_scaled, train_clusters], 
                               axis=1)

    # join on clusterid to get centroids
    X_train_scaled2 = X_train_scaled2.merge(centroids, how='left', 
                                          on=cluster_col_name).\
                            set_index(X_train.index)
    
    return X_train2, X_train_scaled2