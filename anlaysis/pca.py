import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA

# Recieves a matrix and returns the PCA object as well as a matrix that im not sure what it does
def get_pca(X):    
    pca = PCA()
    pc_matrix = pca.fit_transform(X)
    
    return pca, pc_matrix

# Recieves a dataframe with outliers removed and no null values, and the features to use in the PCA
# and returns a dictionary with the power transformer, the principal components, their explained variance,
# and a ndarray that represents the composite index
def get_comp_index(df, features=[]):
    # If no features mentioned, use every column
    if not features: 
        features = df.columns.to_list()
    
    # Get a matrix from the dataframe
    x = df.loc[:, features].values

    # Power transform data to make it normally distributed
    pt = PowerTransformer()
    X = pt.fit_transform(x)

    # run the PCA analysis
    pca, pc_vals = get_pca(X)

    # Arbitrary choice
    VARIANCE_THRESHOLD = 0.75
    
    # Select the features that make up 100*VARANCE_THRESHOLD percent of the variance
    num_components = 0
    sum = 0
    for proportion in pca.explained_variance_ratio_:
        if sum > VARIANCE_THRESHOLD:
            break
        sum += proportion
        num_components += 1
    
    # Select the principal components to be used in the index
    pc_matrix = pca.components_[:num_components]
    var_vals = pca.explained_variance_ratio_[:num_components]

    # weighted average of the components by their contribution to the variance
    composite_index = (1/var_vals.sum())*np.transpose(pc_matrix).dot(var_vals)

    # Returns the information
    return {
        "pt": pt,
        "pc_matrix": pc_matrix,
        "var_vals": var_vals,
        "composite_index": composite_index
    }
