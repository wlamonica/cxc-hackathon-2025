import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def correlation_analysis(df, features):
    if not features: 
        features = df.columns.to_list()
    
    X = df.loc[:, features].values

    return

def get_std_pca(X, features):    
    pca = PCA()
    pc_matrix = pca.fit_transform(X)
    
    return pca, pc_matrix

def get_comp_index(df, features):
    if not features: 
        features = df.columns.to_list()
    
    X = df.loc[:, features].values
    X = StandardScaler().fit_transform(X)
    mu = StandardScaler.mean_
    sigma = sqrt(StandardScaler.var_)

    pca, pc_vals = get_std_pca(X, features)

    # Arbitrary choice
    variance_threshold = 0.75
    num_components = 0
    sum = 1
    for proportion in pca.explained_variance_ratio_:
        if sum > variance_threshold:
            break
        sum += proportion
        num_components += 1
    
    pc_matrix = pca.components_[:num_components]
    var_vals = pca.explained_variance_ratio_[:num_components]
    mu = mu[:num_components]
    sigma = sigma[:num_components]

    composite_index_unnormalized = (1/var_vals.sum())*pc_matrix.dot(var_vals)

    return {
        "mu": mu,
        "sigma": sigma,
        "pc_matrix": pc_matrix,
        "var_vals": var_vals,
        "composite_index": composite_index_unnormalized
    }
