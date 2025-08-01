import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt

from rdkit.Chem import AllChem, Descriptors, Descriptors3D

import sys
sys.path.append('./')  # so utility scripts can be found
from scripts.utils.helpers import (
    z_scale_df,
    get_linear_model,
    get_descriptors,
    compute_vifs,
    PCA_plot,
    remove_high_vif_descriptors,
    kmeans_clustering,
    Gaussian_mixture_clustering,
)


cachedir = Path('cache/eadb')
activity_df = pd.read_parquet(cachedir / 'activity_matrix_filled.parquet')

def characterize_descriptors():
    # Convert the 'title' column to RDKit Mol objects
    mol_list = [AllChem.AddHs(AllChem.MolFromInchi(s)) for s in tqdm(activity_df.index, desc="Converting InChIs to RDKit Mol objects")]

    # Calculate descriptors for each molecule
    descriptor_parquet = cachedir / 'descriptors.parquet'
    use_cache = False
    if descriptor_parquet.exists() and use_cache:
        print(f"Loading existing descriptors from {descriptor_parquet}")
        descriptor_df = pd.read_parquet(descriptor_parquet)
    else:
        descriptor_vectors = [get_descriptors(
            mol,
            use_phthalate_set=False,
            use_general_set=True,
            # use_vectors=True,
        ) for mol in tqdm(mol_list, desc="Calculating descriptors")]
        descriptor_df = pd.DataFrame(descriptor_vectors, index=activity_df.index)
        descriptor_df.to_parquet(descriptor_parquet)

    # Data preprocessing: scale the descriptors
    X = z_scale_df(descriptor_df)
    Y = z_scale_df(activity_df)

    X = remove_high_vif_descriptors(X, vif_threshold=10)

    # Compute variance inflation factors (VIFs) to check for multicollinearity
    vif_table = compute_vifs(X)
    print("VIF Table:")
    print(vif_table)

    for descriptor in vif_table['descriptor']:
        if vif_table.loc[vif_table['descriptor'] == descriptor, 'VIF'].values[0] > 10:
            print(f"Warning: High VIF detected for descriptor '{descriptor}' (VIF={vif_table.loc[vif_table['descriptor'] == descriptor, 'VIF'].values[0]}). Consider removing it.")

# SECTION: Make a predictive model for EADB endpoints
def comma_remove(s):
    return s.replace(',', '')

def get_log_rba(eadb):
    # filter for logRBA endpoint
    log_rba = eadb.loc[eadb['EndpointName'] == 'logRBA', ['inchi', 'EndpointValue']]
    log_rba.rename(columns={'EndpointValue': 'logRBA'}, inplace=True)
    # set the InChI as index
    log_rba.set_index('inchi', inplace=True)
    log_rba = log_rba[log_rba.index.notna()]
    # clean the logRBA column
    log_rba['logRBA'] = log_rba['logRBA'].apply(comma_remove)
    # convert logRBA to numeric
    log_rba['logRBA'] = pd.to_numeric(log_rba['logRBA'], errors='coerce')
    # set sentinel values to NaN
    log_rba[log_rba < -10] = pd.NA
    # drop rows with NaN in logRBA
    log_rba = log_rba.dropna()
    # for duplicate InChIs, take the mean of logRBA values
    log_rba = log_rba.groupby(log_rba.index).mean()

    return log_rba

eadb = pd.read_parquet(cachedir / 'eadb.parquet')
log_rba = get_log_rba(eadb)

# make predictor and target DataFrames
index_intersection = activity_df.index.intersection(log_rba.index)
X = activity_df.loc[index_intersection]
y = log_rba.loc[index_intersection, 'logRBA']

def get_PCA():
    # Perform PCA on the activity matrix and print the explained variance
    print("Performing PCA on the activity matrix...")
    from sklearn.decomposition import PCA 
    pca = PCA(n_components=50, random_state=0)
    X_pca = pca.fit_transform(X)
    # print the explained variance by component
    # print("Explained variance by PCA components:", pca.explained_variance_ratio_)
    # print("Cumulative explained variance by PCA components:", np.cumsum(pca.explained_variance_ratio_))
    # plot the PCA explained variance
    import matplotlib.pyplot as plt
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='s')
    plt.show()

def get_histogram():
    # show a histogram of the logRBA values
    print("Plotting histogram of logRBA values...")

    plt.hist(y, bins=50, edgecolor='black',)
    plt.xlabel('logRBA')
    plt.ylabel('Frequency')
    plt.title('Histogram of logRBA Values')
    plt.show()

def get_linear():
    # Fit a linear regression model
    print("Fitting linear regression model...")
    ols = get_linear_model(X, y)

# get_linear()

def get_clusters():
    # Use k-means clustering, attempting to distinguish between high and low RBA
    kmeans_clustering(X, y, n_clusters=2, plot_clusters=True)
    # Gaussian_mixture_clustering(X, y, n_components=2, plot_clusters=True)  # not performing as well as k-means


y_binary = (y > 0).astype(int)  # 1 for high RBA, 0 for low RBA
print(np.mean(y_binary ))
sys.exit(0)

# construct a binary model for high vs low RBA
def get_binary_model():
    # Fit a logistic regression model
    print("Fitting logistic regression model...")
    logit_model = sm.Logit(y_binary, X).fit(disp=0)

    # Print the summary of the model
    print(logit_model.summary())
    
    return logit_model

get_binary_model()


def get_decision_tree_model():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate
    import numpy as np

    print("Fitting decision tree model with 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    dt_model = DecisionTreeClassifier(random_state=0)

    cv_results = cross_validate(
        dt_model,
        X,
        y_binary,
        cv=cv,
        scoring=['accuracy', 'roc_auc'],
        return_estimator=True,
        n_jobs=-1,
    )

    print(f"Mean CV accuracy: {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
    print(f"Mean CV ROC-AUC: {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}")

    # Select the model with the best ROC-AUC
    best_idx = np.argmax(cv_results['test_roc_auc'])
    best_model = cv_results['estimator'][best_idx]

    # Print feature importances from the best fold
    feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    print("Top feature importances (best fold):")
    print(feature_importances.sort_values(ascending=False)[:10])

    return best_model


get_decision_tree_model()
