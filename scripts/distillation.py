import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from rdkit import Chem

from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
sys.path.append('./')
from scripts.utils.helpers import (
    get_activity_df,
    smiles_to_inchi,
    remove_high_vif_descriptors,
    inchis_to_morgan_df,
    kmeans_clustering,
    Gaussian_mixture_clustering,
)

def load_edkb(path: Path, activity_field: str):
    # rename *.txt to *.sdf* if you prefer
    suppl = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)
    records = []
    for mol in suppl:
        if mol is None:              # skip bad parses
            continue
        log_rba = float(mol.GetProp(activity_field))
        if log_rba <= -5_000:        # sentinel for “inactive”
            continue                 # or keep and label 0 for classification
        records.append({
            "smiles": Chem.MolToSmiles(mol),
            "log_rba": log_rba
        })
    return pd.DataFrame(records)

def get_combined_edkb(resourcedir: Path):
    edkb_file = resourcedir / 'edkb.parquet'

    if edkb_file.exists():
        edkb = pd.read_parquet(edkb_file)
    else:
        def get_endpoint(dataset_name: str):
            endpoint = 'NCTRlogRBA'
            if dataset_name == 'Androgen':
                endpoint = 'ar' + endpoint

            return endpoint
        
        dataset_names = ['Androgen', 'Estrogen']
        datasets = {
            name: load_edkb(resourcedir / f'EDKB_{name}.sdf', get_endpoint(name))
            for name in dataset_names
        }

        # add InChI column to the datasets      
        for df in datasets.values():
            df['inchi'] = df['smiles'].apply(smiles_to_inchi)

        # concatenate the dataframes
        edkb = pd.concat(datasets.values()).reset_index(drop=True)
        edkb.to_parquet(edkb_file)

    return edkb

def get_edkb_log_rba(resourcedir: Path):
    edkb_file = resourcedir / 'edkb_log_rba.parquet'
    edkb = pd.read_parquet(edkb_file)
    return edkb

def load_dataframes(cachedir: Path, outdir: Path):
    activity_df = get_activity_df(cachedir)  # chemicals × assays
    descriptor_parquet = outdir / 'descriptors.parquet'
    descriptor_df = pd.read_parquet(descriptor_parquet)  # chemicals × descriptors

    return activity_df, descriptor_df

def fit_lin_nonlin(X, y):
    # 1. split (random example)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. linear baseline
    enet = GridSearchCV(
        ElasticNet(max_iter=10_000),
        param_grid={'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]},
        cv=5
    ).fit(X_train, y_train)

    print("ElasticNet MAE:", mean_absolute_error(y_test, enet.predict(X_test)))

    # 3. boosted trees
    gbr = GridSearchCV(
        HistGradientBoostingRegressor(),
        param_grid={'learning_rate': [0.03, 0.1],
                    'max_depth': [None, 6, 10],
                    'l2_regularization': [0.0, 1.0]},
        cv=5
    ).fit(X_train, y_train)

    print("GradientBoosting MAE:", mean_absolute_error(y_test, gbr.predict(X_test)))

    return enet, gbr

def plot_linear(X, y, title=''):
    # construct a linear model of the log(RBA) activity based on X
    model = LinearRegression()
    model.fit(X, y)
    print(title)
    # print("Linear Regression coefficients:", model.coef_)
    # print("Linear Regression intercept:", model.intercept_)
    print("Linear Regression MAE:", mean_absolute_error(y, model.predict(X)))
    print("Linear Regression R^2:", model.score(X, y))

    # plot predictions vs. actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(model.predict(X), y, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Predicted Log RBA')
    plt.ylabel('Actual Log RBA')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process activity matrix for entity similarity.")
    parser.add_argument('--cachedir', type=str, default='cache/edkb',
                        help='Directory to cache the activity matrix.')
    parser.add_argument('--outdir', type=str, default='cache/edkb',
                        help='Directory to cache the descriptors.')
    parser.add_argument('--histogram', action='store_true',
                        help='Whether to plot the histogram of log RBA values.')
    parser.add_argument('--plot_linear', action='store_true',
                        help='Whether to plot the linear regression model.')
    parser.add_argument('--plot_ridge', action='store_true',
                        help='Whether to plot the polynomial ridge regression.')
    parser.add_argument('--random_forest', action='store_true',
                        help='Whether to fit a random forest model.')
    parser.add_argument('--remove_high_vif', action='store_true',
                        help='Whether to remove high VIF descriptors before modeling.')
    parser.add_argument('--clustering', action='store_true',
                        help='Whether to perform clustering on the activity matrix.')
    parser.add_argument('--binary_classifier', action='store_true',
                        help='Whether to use a binary classifier to predict the log RBA activity.')
    args = parser.parse_args()

    # define the important directories
    cachedir = Path(args.cachedir)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    activity_df, descriptor_df = load_dataframes(cachedir, outdir)

    if args.remove_high_vif:
        descriptor_df = remove_high_vif_descriptors(descriptor_df, vif_threshold=10)

    # edkb = get_combined_edkb(Path('resources'))
    edkb = get_edkb_log_rba(Path('resources'))

    # Select rows from activity_df whose index is in edkb['inchi']
    X_activity = activity_df.loc[activity_df.index.intersection(edkb['inchi'])]
    X_descriptors = descriptor_df.loc[X_activity.index]
    # scale the descriptors
    X_descriptors = StandardScaler().fit_transform(X_descriptors)
    y = edkb[['inchi', 'log_rba']]
    y = y[y['inchi'].isin(X_activity.index)].drop_duplicates(subset=['inchi'])
    y = y.set_index('inchi').loc[:, 'log_rba']
    print(f'Predicting using {len(X_activity)} chemicals.')

    if args.plot_linear:
        # plot_linear(X_activity, y, title='Linear Model Based on Activity Matrix')
        plot_linear(X_descriptors, y, title='Linear Model Based on Descriptors')

    if args.histogram:
        # show histogram of log_rba values
        fig, ax = plt.subplots(tight_layout=True)
        hist = ax.hist(y, color='blue', alpha=0.7)
        ax.set_title('Log RBA Distribution')
        ax.set_xlabel('Log RBA')
        ax.set_ylabel('Frequency')
        plt.show()

    # Fit nonlinear models
    if args.plot_ridge:
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge   # L2‑regularised regression
        from sklearn.model_selection import cross_val_score

        poly_ridge = Pipeline([
            ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
            ('scale',  StandardScaler()),        # keeps regularisation well‑behaved
            ('ridge',  Ridge(alpha=1.0))         # alpha is the ℓ2 penalty strength
        ])

        cv_rmse = (-cross_val_score(poly_ridge,
                                    X_descriptors,
                                    y,
                                    cv=5,
                                    scoring='neg_root_mean_squared_error')
                .mean())

        print(f'5-fold CV-RMSE: {cv_rmse:.3f}')

        poly_ridge.fit(X_descriptors, y)         # fit on the full set if you like the CV score
        y_hat = poly_ridge.predict(X_descriptors)
        print("Polynomial Ridge MAE:", mean_absolute_error(y, y_hat))
        print("Polynomial Ridge R^2:", poly_ridge.score(X_descriptors, y))
        # plot predictions vs. actual values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_hat, y, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Predicted Log RBA')
        plt.ylabel('Actual Log RBA')
        plt.title('Polynomial Ridge Regression Predictions')
        plt.show()
    
    if args.random_forest:
        # -------------------- 1. split ------------------------------------------------
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_descriptors, y,
            test_size=0.20,          # 20 % hold‑out
            random_state=0,
        )

        # -------------------- 2. fit --------------------------------------------------
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(
                n_estimators=100,    # more trees → lower variance
                # max_depth=None,      # let trees grow until leaves are pure or min_samples_leaf reached
                max_depth=6,      
                min_samples_leaf=1,  # avoids single‑point leaves; improves generalisation
                n_jobs=-1,           # use all cores
                random_state=0,
                max_features='sqrt',  # use sqrt of features for each split
                bootstrap=True,
        )
        rf.fit(X_tr, y_tr)

        # -------------------- 3. evaluate --------------------------------------------
        from sklearn.metrics import r2_score
        y_pred = rf.predict(X_te)
        print(f'R² (test): {r2_score(y_te, y_pred):.3f}')

        # -------------------- 4. plot -------------------------------------------------
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.scatter(y_te, y_pred, alpha=0.7)
        plt.plot([y_te.min(), y_te.max()],
                [y_te.min(), y_te.max()],
                '--', linewidth=1.5, color='red'
        )
        plt.xlabel('Actual Log RBA')
        plt.ylabel('Predicted Log RBA')
        plt.title('Random Forest: Actual vs. Predicted')
        plt.tight_layout()
        plt.show()

    if args.clustering:

        from sklearn_extra.cluster import KMeansConstrained  # pip install scikit-learn-extra
        from sklearn.metrics import silhouette_score

        def balanced_kmeans_clustering(X, y, *, n_clusters=2,
                                    size_min=50, size_max=300,
                                    print_tag=''):
            pipe = Pipeline([
                ('scale', StandardScaler()),
                ('k', KMeansConstrained(n_clusters=n_clusters,
                                        size_min=size_min,
                                        size_max=size_max,
                                        n_init=30,
                                        random_state=0))
            ])
            labels = pipe.fit_predict(X)
            print('Silhouette', print_tag + ':', silhouette_score(X, labels))
            df = pd.DataFrame({'logRBA': y, 'cluster': labels})
            print(df.groupby('cluster')['logRBA'].describe())

        fp_df = inchis_to_morgan_df(X_descriptors)
        kmeans_clustering(fp_df, y, print_tag='(fingerprints, k-means)')
        balanced_kmeans_clustering(fp_df, y, print_tag='(fingerprints, balanced k-means)')
        Gaussian_mixture_clustering(fp_df, y, print_tag='(fingerprints, GMM)')

    # use a binary classifier to predict the log RBA activity
    if args.binary_classifier:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

        # Convert log RBA to binary activity
        y_binary = (y > 0).astype(int)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_descriptors, y_binary, test_size=0.2, random_state=42,
            stratify=y_binary,
        )

        def score_classifier(clf, X_test, y_test):
            # Make predictions
            y_pred = clf.predict(X_test)

            # Print classification report
            print("Classification Report:")
            print(classification_report(y_test, y_pred))

            # Print confusion matrix
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

            print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))

        def rf_score_classifier(X_train, X_test, y_train, y_test):
            # Fit the random forest classifier
            rf_clf = RandomForestClassifier(
                n_estimators=100, random_state=42,
                class_weight='balanced',  # handle class imbalance
            )
            rf_clf.fit(X_train, y_train)

            # Score the classifier
            return score_classifier(rf_clf, X_test, y_test)
        
        def logistic_score_classifier(X_train, X_test, y_train, y_test):
            # Fit the logistic regression classifier
            log_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            log_clf.fit(X_train, y_train)

            # Score the classifier
            return score_classifier(log_clf, X_test, y_test)

        # # predict using test/train split
        # score_classifier(X_train, X_test, y_train, y_test)
        # predict on entire dataset
        # print("Random Forest Classifier:")
        # rf_score_classifier(X_descriptors, X_descriptors, y_binary, y_binary)
        print("Logistic Regression Classifier:")
        logistic_score_classifier(X_descriptors, X_descriptors, y_binary, y_binary)
        # logistic_score_classifier(X_train, X_test, y_train, y_test)