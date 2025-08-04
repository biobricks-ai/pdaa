import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt

from rdkit.Chem import AllChem

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

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
eadb = pd.read_parquet(cachedir / 'eadb.parquet')

def comma_remove(s):
    return s.replace(',', '')

def get_endpoint_series(df, endpoint, *, p_conversion = False, sentinel_threshold=-10):
    # filter for logRBA endpoint
    endpoint_series = df.loc[df['EndpointName'] == endpoint, ['inchi', 'EndpointValue']]
    endpoint_series.rename(columns={'EndpointValue': endpoint}, inplace=True)
    # set the InChI as index
    endpoint_series.set_index('inchi', inplace=True)
    endpoint_series = endpoint_series[endpoint_series.index.notna()]
    # clean the logRBA column
    endpoint_series[endpoint] = endpoint_series[endpoint].apply(comma_remove)
    # convert logRBA to numeric
    endpoint_series[endpoint] = pd.to_numeric(endpoint_series[endpoint], errors='coerce')
    # convert to pIC50, pKi, etc. if applicable
    if p_conversion:
        endpoint_series[endpoint] = -np.log10(endpoint_series[endpoint])
    # replace inf with NaN
    endpoint_series = endpoint_series.replace([np.inf, -np.inf], np.nan)
    # set sentinel values to NaN
    if sentinel_threshold is not None:
        endpoint_series[endpoint_series < sentinel_threshold] = np.nan
    # drop rows with NaN in logRBA
    endpoint_series = endpoint_series.dropna()
    # for duplicate InChIs, take the mean of logRBA values
    endpoint_series = endpoint_series.groupby(endpoint_series.index).mean()

    return endpoint_series


log_rba = get_endpoint_series(eadb, 'logRBA', sentinel_threshold=-10)

def x_in_range(x, values):
    """
    Check if x is within the interquartile range of values.
    """
    q_low  = values.quantile(0.25).values
    q_high = values.quantile(0.75).values
    return q_low <= x <= q_high

def process_eadb_endpoint(endpoint):
    """
    Process a specific EADB endpoint, cleaning and preparing the data for analysis.
    
    Args:
        endpoint (str): The name of the endpoint to process.
        
    Returns:
        pd.Series: Cleaned and processed values for the specified endpoint.
    """

    # Convert to pIC50, pKi, etc. if applicable
    if endpoint in [
        'Ki',
        'IC50',
        # 'INH',
        # 'ED50',
        'GI50',
        # 'Antagonism',
        # 'Agonism',
        'EC50',
        'Kd',
        'Ka',
        'IC30',
        'REC10'
    ]:
        p_conversion = True
    else:
        p_conversion = False
    adj_endpoint = f"p{endpoint}" if p_conversion else endpoint

    # filter out sentinel values
    if endpoint in ['logRBA', 'logRA', 'logRE', 'logRP', 'logRPE',]:
        sentinel_threshold = -100
    elif endpoint in ['Ki']:
        sentinel_threshold = -6
    elif endpoint in ['INH', 'Antagonism', 'Agonism',]:
        sentinel_threshold = 0
    else:
        sentinel_threshold = None

    vals = get_endpoint_series(eadb, endpoint, p_conversion=p_conversion, sentinel_threshold=sentinel_threshold)

    # get threhold values for conversion to binary
    if p_conversion and x_in_range(0, vals):
        threhold = 0
    elif (not p_conversion) and x_in_range(50, vals):
        threhold = 50
    else:
        threhold = vals.median()

    return vals, adj_endpoint, threhold

"""
array(['logRBA', 'logRA', 'logRE', 'logRPP', 'logRP', 'Ki', 'IC50', 'INH',
       'logRA10', 'ED50', 'GI50', 'Antagonism', 'Agonism', 'EC50',
       'logRPE', 'Kd', 'Ka', 'IC30', 'REC10'], dtype=object)
"""
# for endpoint in eadb.EndpointName.unique():
#     vals, adj_endpoint, _ = process_eadb_endpoint(endpoint)
#     print(f"Processed {endpoint} with {len(vals)} values, adjusted endpoint: {adj_endpoint}")
    # vals.describe()  # print summary statistics
    # # # pause for user to read
    # # input(f"Press Enter to continue with histogram for {endpoint}...")

    # # n, bins, patches = plt.hist(
    # #     vals,
    # #     alpha=0.5
    # # )
    
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.ecdf(vals, label=endpoint)
    # # add horizontal line at y=0.5
    # ax.axhline(y=0.5, color='r', linestyle='--', label='Median')
    # ax.legend()
    # ax.set_xlabel(adj_endpoint)
    # # ax.set_title(f"Histogram of {endpoint} values")
    # ax.set_title(f"ECDF of {adj_endpoint} values")
    # # # Set xticks at the center of each bar
    # # plt.xticks((bins[:-1] + bins[1:]) / 2, rotation=90)
    # plt.show()

    

# make predictor and target DataFrames
index_intersection = activity_df.index.intersection(log_rba.index)
X = activity_df.loc[index_intersection]
y = log_rba.loc[index_intersection, 'logRBA']

def transform_X(X, transform_type: str = ''):
    # ---- pick feature matrix ----
    if transform_type == '':
        X_trans = X
    elif transform_type == 'z_scale':
        X_trans = z_scale_df(X)
    elif transform_type == 'binary':
        X_trans = (X > 0.5).astype(int)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    
    return X_trans

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
print(f"{100*np.mean(y_binary ):.2f}% of substances have high RBA (logRBA > 0)\n")

# construct a binary model for high vs low RBA
def get_binary_model():
    # Fit a logistic regression model
    print("Fitting logistic regression model...")
    logit_model = sm.Logit(y_binary, X).fit(disp=0)

    # Print the summary of the model
    print(logit_model.summary())
    
    return logit_model

# get_binary_model()

def get_decision_tree_model():
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


# get_decision_tree_model()

def get_decision_tree_model_feature_selection(
    transform_type='',
):
    pipe = Pipeline([
        ("filter_mi", SelectKBest(mutual_info_classif, k=154)),  # tune k
        ("clf", DecisionTreeClassifier(random_state=0)),
    ])

    X_trans = transform_X(X, transform_type)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_results = cross_validate(
        pipe,
        X_trans,
        y_binary,
        cv=cv,
        scoring=['accuracy', 'roc_auc'],
        n_jobs=-1
    )
    print("Results for Decision Tree with feature selection:")
    print(f"\tMean CV accuracy: {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
    print(f"\tMean CV ROC-AUC: {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}")
    
    # # Select the model with the best ROC-AUC
    # best_idx = np.argmax(cv_results['test_roc_auc'])
    # best_model = cv_results['estimator'][best_idx]

    # # Print feature importances from the best fold
    # feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    # print("\n\tTop feature importances (best fold):")
    # for f, i in feature_importances.sort_values(ascending=False)[:10].index:
    #     print(f"\t{f}: {i:.4f}")

get_decision_tree_model_feature_selection(
    transform_type='',
)

def get_random_forest_regressor_feature_selection(
    transform_type='',
    k=154,  # number of top features to keep
    n_estimators=200,
    max_depth=None,
):
    """
    Train a RandomForestRegressor with optional feature selection and data transform.

    Parameters
    ----------
    transform_type : str
        '', 'z_scale', or 'binary'  (same semantics as your decision-tree helper)
    k : int
        Number of top features to keep with SelectKBest(f_regression).
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum depth of the trees.

    Returns
    -------
    best_model : RandomForestRegressor
        Estimator from the CV fold with the best R².
    """
    from sklearn.feature_selection import f_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    X_trans = transform_X(X, transform_type)

    # ----- build pipeline -----
    pipe = Pipeline([
        ("filter_f", SelectKBest(f_regression, k=k)),
        ("rf", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=0,
            n_jobs=-1,
        )),
    ])

    # ----- cross-validate -----
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_results = cross_validate(
        pipe,
        X_trans,
        y,                        # continuous target
        cv=cv,
        scoring=['r2', 'neg_root_mean_squared_error'],
        return_estimator=True,
        n_jobs=-1,
    )

    mean_r2 = cv_results['test_r2'].mean()
    std_r2 = cv_results['test_r2'].std()

    rmse = -cv_results['test_neg_root_mean_squared_error']  # negate to get +RMSE
    mean_rmse = rmse.mean()
    std_rmse = rmse.std()

    print("Results for RandomForestRegressor with feature selection:")
    print(f"\tMean CV R²:   {mean_r2:.3f} ± {std_r2:.3f}")
    print(f"\tMean CV RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}")

    # ----- return best model -----
    best_idx = np.argmax(cv_results['test_r2'])
    best_model = cv_results['estimator'][best_idx]
    return best_model

# get_random_forest_regressor_feature_selection(
#     transform_type='binary',
#     k=154,
#     n_estimators=200,
#     max_depth=None,
# )

def get_xgb_classifier_feature_selection(
    X: pd.DataFrame,
    y_binary: pd.Series,
    *,
    transform_type: str = '',
    k: int = 154,
    n_iter: int = 30,
    random_state: int = 0,
):
    """
    Extreme Gradient Boosting (binary classification) with MI feature selection
    and nested cross-validation hyper-parameter optimisation.

    Parameters
    ----------
    transform_type : str
        '', 'z_scale', or 'binary':  same semantics as earlier helpers.
    k : int
        Number of top mutual-information features to keep.
    n_iter : int
        Number of RandomizedSearchCV trials per inner CV.
    random_state : int
        RNG seed for full reproducibility.

    Returns
    -------
    best_model : xgboost.XGBClassifier
        Best estimator found across outer CV folds.
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.model_selection import RandomizedSearchCV
    from xgboost import XGBClassifier
    import warnings

    X_trans = transform_X(X, transform_type)

    # ---- base pipeline ----
    base_pipe = Pipeline([
        ("filter_mi", SelectKBest(mutual_info_classif, k=k)),
        ("clf", XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',      # needed to silence deprecation warnings
            tree_method='hist',         # fast histogram-based split finding
            # use_label_encoder=False,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    # ---- hyper-parameter space (lists work fine for RandomizedSearchCV) ----
    param_dist = {
        'clf__n_estimators':       [300, 500, 800, 1000],
        'clf__max_depth':         [3, 4, 5, 6, 7],
        'clf__learning_rate':     [0.01, 0.03, 0.05, 0.1],
        'clf__subsample':         [0.7, 0.8, 0.9, 1.0],
        'clf__colsample_bytree':  [0.6, 0.8, 1.0],
        'clf__gamma':             [0, 0.1, 0.25, 1],
        'clf__min_child_weight':  [1, 3, 5, 10],
    }

    # ---- nested CV ----
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
    )

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # silence fit/predict_proba overlap msgs
        cv_results = cross_validate(
            search,
            X_trans,
            y_binary,
            cv=outer_cv,
            scoring=['accuracy', 'roc_auc'],
            return_estimator=True,
            n_jobs=-1,
        )

    mean_acc  = cv_results['test_accuracy'].mean()
    std_acc   = cv_results['test_accuracy'].std()
    mean_auc  = cv_results['test_roc_auc'].mean()
    std_auc   = cv_results['test_roc_auc'].std()

    metrics = f"""
Mean CV accuracy: {mean_acc:.3f} ± {std_acc:.3f}
Mean CV ROC-AUC: {mean_auc:.3f} ± {std_auc:.3f}
"""
    # print(f"Mean CV accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    # print(f"Mean CV ROC-AUC: {mean_auc:.3f} ± {std_auc:.3f}")

    # ---- pull the best outer-fold model ----
    best_idx   = np.argmax(cv_results['test_roc_auc'])
    best_model = cv_results['estimator'][best_idx].best_estimator_

    # Optional: inspect its top feature importances
    importances = best_model.named_steps['clf'].feature_importances_
    kept_feats  = best_model.named_steps['filter_mi'].get_feature_names_out(X_trans.columns)
    imp_series  = (pd.Series(importances, index=kept_feats)
                     .sort_values(ascending=False)
                     .head(20))
    # print("Top 20 features (best outer fold):")
    # print(imp_series)

    return best_model, metrics, imp_series

with open(cachedir / 'xgb_classifier_feature_selection.txt', 'w') as f:
    for endpoint in eadb.EndpointName.unique():
        vals, adj_endpoint, threshold = process_eadb_endpoint(endpoint)
        
        index_intersection = activity_df.index.intersection(vals.index)
        X = activity_df.loc[index_intersection]
        # y_binary = vals.loc[index_intersection] > threshold  # binary target based on threshold
        y_binary = (vals.loc[index_intersection] > threshold).squeeze() # binary target based on threshold

        best_model, metrics, imp_series = get_xgb_classifier_feature_selection(
            X,
            y_binary,
        )

        f.write("#" + "="*80 + "\n")
        f.write(f"Processed {endpoint} with {len(vals)} values, adjusted endpoint: {adj_endpoint}\n")
        f.write(metrics)
        f.write("Top 20 features (best outer fold):")
        f.write(imp_series.to_string())
        f.write("\n\n")

# get_xgb_classifier_feature_selection()

    
