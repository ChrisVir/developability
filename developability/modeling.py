# Tools for modeling
import mlflow
from mlflow.models import infer_signature

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, logspace, linspace
from sklearn.metrics import (PredictionErrorDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, average_precision_score,
                             mean_squared_error, r2_score, roc_auc_score,
                             accuracy_score, f1_score)
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              RandomForestRegressor, ExtraTreesRegressor)
from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge,
                                  Lasso, ElasticNet)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from pathlib import Path
from tqdm import tqdm

# Utils


def select_features(df, suffixes=None):
    """Selects columns with suffix"""

    if not suffixes:
        suffixes = ['net', 'neg', 'pos']

    def has_suffix(col):
        for suffix in suffixes:
            if col.endswith(suffix):
                return True
        else:
            return False

    return df[[col for col in df.columns if has_suffix(col)]]


def get_average_predictions(X, y, model, cv=None, regression=True):
    """Get average predictions from cross validation
    Args:
        X (pd.DataFrame): features
        y (pd.Series): target
        model (sklearn model): model to use
        cv (sklearn KFold, iterable): cross validation object
        regression (bool): whether it is a regression model

    Returns:
        mean_predictions (pd.DataFrame): mean predictions
        predictions_df (pd.DataFrame): predictions for each fold
    """
    if cv is None:
        cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=32)
        splitter = cv.split(X)
    else:
        splitter = cv

    predictions = {}

    for train_index, test_index in splitter:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, _ = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        if regression:
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict_proba(X_test)[:, 1]

        for i, idx in enumerate(test_index):
            predictions.setdefault(idx, []).append(y_pred[i])

    predictions_df = pd.DataFrame(predictions).T.sort_index()
    predictions_df.index = X.index
    # predictions_df.index.name = 'antibody'

    mean_predictions = pd.DataFrame(
        predictions_df.mean(axis=1), columns=['y_pred_mean'])
    mean_predictions['y_true'] = y
    return mean_predictions, predictions_df


def get_feature_importances(model):
    """Get feature importances from a model"""

    if hasattr(model[-1], 'feature_importances_'):
        importances = model[-1].feature_importances_

    if hasattr(model[-1], 'coef_'):
        importances = model[-1].coef_
        shape = importances.shape
        if len(shape) == 2:
            importances = importances[0]

    features = model[:-1].get_feature_names_out()
    s = (pd.Series(importances, index=features)
         .sort_values()
         )
    return s

# Visualization


def plot_feature_importances(model, n_features=20, ax=None, model_name=None):
    """
    Plot feature importances or coefficients of a model.

    Args:
        model: The trained model object.
        n_features: The number of top features to display (default is 10).
        ax: The matplotlib axes object to plot the feature importances
             (default is None).
        model_name: The name of the model (optional).
        axes: The matplotlib axes object to plot the feature importances
              (default is None).

    Returns:
        ax: The matplotlib axes object containing the feature importances plot.

    Note:
        This function assumes that the model object has either a
        'feature_importances_' or 'coef_' attribute.

    """
    s = get_feature_importances(model).iloc[:n_features]
    if hasattr(model[-1], 'feature_importances_'):
        title = 'Feature Importances'
    if hasattr(model[-1], 'coef_'):
        title = 'Coefficients'
    else:
        title = ""

    if model_name:
        title = f'{model_name} {title}'

    if ax:
        fig = plt.gca()
        ax = s.plot(kind='barh', ax=ax)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.set_size_inches(9, 5)
        ax = s.plot(kind='barh', ax=ax)

    ax.set_title(title)
    plt.tight_layout()
    sns.despine()

    return fig, ax


def plot_regression_performance(model, X, y, model_name=None,
                                dataset='validation', figsize=(8, 3)):

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    _ = PredictionErrorDisplay.from_estimator(
        model, X, y, kind="actual_vs_predicted", ax=axes[0])

    _ = PredictionErrorDisplay.from_estimator(
        model, X, y, kind="residual_vs_predicted", ax=axes[1])
    sns.despine()
    fig.suptitle(
        f'Regression Performance for {model_name} on {dataset} dataset',
        fontsize=14)
    fig.tight_layout()
    return fig, axes


def plot_regression_performance_from_predictions(y_true, y_pred,
                                                 model_name=None,
                                                 dataset='validation',
                                                 figsize=(8, 4),
                                                 set_lim=False):
    """Plot regression performance from predictions
    Args:
        y_true (pd.Series): true values
        y_pred (pd.Series): predicted values
        model_name (str): name of model
        dataset (str): name of dataset
        figsize (tuple): size of figure
    Returns:
        fig, axes (tuple): figure and axes
    """

    if set_lim:
        max_val = int(np.ceil(max(y_true.max(), y_pred.max())))+1
        min_val = int(np.floor(min(y_true.min(), y_pred.min())))-1

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax = axes[0]
    _ = PredictionErrorDisplay.from_predictions(
        y_true, y_pred, kind="actual_vs_predicted", ax=ax)

    if set_lim:
        ax.set(xlim=(min_val, max_val), ylim=(min_val, max_val))
        ax.set_xticks(range(min_val, max_val, 2))
        ax.set_yticks(range(min_val, max_val, 2))

    ax = axes[1]
    _ = PredictionErrorDisplay.from_predictions(
        y_true, y_pred, kind="residual_vs_predicted", ax=ax)

    if set_lim:
        ax.set(xlim=(min_val, max_val))
        ax.set_xticks(range(min_val, max_val, 2))

    sns.despine()
    fig.suptitle(
        f'Regression Performance for {model_name} on {dataset} dataset',
        fontsize=14)
    fig.tight_layout()
    return fig, axes


def plot_classification_performance_from_predictions(y_true, y_pred,
                                                     model_name=None,
                                                     dataset='validation',
                                                     figsize=(8, 4)):
    """Plot classification performance from predictions
    Args:
        y_true (pd.Series): true values
        y_pred (pd.Series): predicted values
        model_name (str): name of model
        dataset (str): name of dataset
        figsize (tuple): size of figure
    Returns:
        fig, axes (tuple): figure and axes
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax = axes[0]
    _ = RocCurveDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set(xlim=(0, 1), ylim=(0, 1))

    ax = axes[1]
    _ = PrecisionRecallDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    sns.despine()
    fig.suptitle(
        f'Classification Performance for {model_name} on {dataset} dataset',
        fontsize=14)
    fig.tight_layout()
    return fig, axes


def bin_data_into_quantiles(data, num_bins=10):
    """
    Bins the data into quantiles.

    Args:
        data (pandas.Series or pandas.DataFrame): The data to be binned.
        num_bins (int): The number of quantiles to create.

    Returns:
        pandas.Series or pandas.DataFrame: The binned data.
    """
    if isinstance(data, pd.Series):
        return pd.qcut(data, num_bins, labels=False)
    elif isinstance(data, pd.DataFrame):
        return data.apply(lambda x: pd.qcut(x, num_bins, labels=False))
    else:
        raise ValueError(
            "Unsupported data type. Expected pd.Series or pd.DataFrame.")


def split_continous(X, y, n_splits=5, n_repeats=10, random_state=42):
    """Split data into quantiles and then split into train and test sets
    Args:
        X (pandas.DataFrame): The data to be binned.
        y (pandas.Series): The number of quantiles to create.
        n_splits (int): number of splits
        n_repeats (int): number of repeats
        random_state (int): random state
    Yields:
        train_index, test_index (tuple): train and test indices.
    """

    y_bined = bin_data_into_quantiles(y, num_bins=n_splits)
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    for train_index, test_index in rskf.split(X, y_bined):
        yield train_index, test_index


def round_dataframe(df, decimals=3):
    """Round all values in dataframe to specified number of decimals"""
    for col in df.columns:
        if df[col].dtype == int or df[col].dtype == float:
            df[col] = df[col].round(decimals)

    return df


def extract_train_test_cv_scores(grid, scores=None, decimals=3, n_total=50):
    """Extract train and test scores from gridsearchcv object and returns a
       dictionary of scores with keys having name of score and value being
       the score.
    Args:
        grid: GridSearchCV object
        score_name: Name of score to extract.
                    If None, all scores are extracted.
        decimals: Number of decimals to round scores to
    Returns:
        scores: Dictionary of scores with keys having name of score and value
                being the score.
    """

    def rename(score):
        names = {'neg_mean_squared_error': 'mse',
                 'neg_mean_absolute_error': 'mae',
                 'neg_root_mean_squared_error': 'rmse',
                 'r2': 'r2',
                 'roc_auc': 'roc_auc',
                 'precision_recall': 'precision_recall',
                 'accuracy': 'accuracy',
                 'f1': 'f1'}

        if score in names:
            return names[score]
        else:
            return score

    if not scores:
        scores = ['score']

    scores_df = (pd.DataFrame(grid.cv_results_)
                 .sort_values(f'rank_test_{scores[0]}')
                 .head(1)
                 )

    scores_df = round_dataframe(scores_df, decimals=decimals)

    scores_dict = {}
    for score in scores:
        name = rename(score)

        scores_dict[f'train_cv_{name}'] = np.abs(
            scores_df[f'mean_train_{score}'].values[0])

        scores_dict[f'train_cv_{name}_sem'] = np.round(
            scores_df[f'std_train_{score}'].values[0]/np.sqrt(n_total),
            decimals)

        scores_dict[f'test_cv_{name}'] = np.abs(
            scores_df[f'mean_test_{score}'].values[0])

        scores_dict[f'test_cv_{name}_sem'] = np.round(
            scores_df[f'std_test_{score}'].values[0]/np.sqrt(n_total),
            decimals)

    return scores_dict


#############################################################################

# Functions for doing ml flow.
classifiers = {'rf': RandomForestClassifier(random_state=42, n_jobs=4),
               'logistic': LogisticRegression(solver='saga', random_state=42),
               'et': ExtraTreesClassifier(random_state=42, n_jobs=4)
               }


regressors = {'rf': RandomForestRegressor(random_state=42, n_jobs=4),
              'linear': LinearRegression(),
              'ridge': Ridge(),
              'lasso': Lasso(),
              'elasticnet': ElasticNet(),
              'et': ExtraTreesRegressor()
              }


# parameters for grid search
classifier_params = {'rf': {'rf__n_estimators': [50, 100, 200],
                            'rf__max_depth': [3, 5, 7, 9],
                            'rf__max_features': ['sqrt', 'log2']},

                     'logistic': {'logistic__penalty': ['l1', 'l2',
                                                        'elasticnet'],
                                  'logistic__C': logspace(-4, 4, 9),
                                  'logistic__l1_ratio': linspace(0, 1, 10)},

                     'et': {'et__n_estimators': [50, 100, 200],
                            'et__max_depth': [3, 5, 7, 9],
                            'et__max_features': ['sqrt', 'log2']
                            }
                     }

regressor_params = {'rf': {'rf__n_estimators': [50, 100, 200],
                           'rf__max_depth': [3, 5, 7, 9],
                           'rf__max_features': ['sqrt', 'log2']},

                    'linear': {'linear__fit_intercept': [True]},

                    'ridge': {'ridge__alpha': logspace(-4, 4, 9)},

                    'lasso': {'lasso__alpha': logspace(-4, 4, 9)},

                    'elasticnet': {'elasticnet__alpha': logspace(-4, 4, 9),
                                   'elasticnet__l1_ratio': arange(0, 1.01, 0.1)
                                   },

                    'et': {'et__n_estimators': [50, 100, 200],
                           'et__max_depth': [3, 5, 7, 9],
                           'et__max_features': ['sqrt', 'log2']}
                    }


def get_model(model_name, regression=True):
    """gets the model and params for grid search
    Args:
        model_name (str): name of model
        regression (bool): whether it is a regression model

    Returns:
        model (sklearn model): model
        params (dict): parameters for grid search
    """
    if regression:
        return regressors[model_name], regressor_params[model_name]
    else:
        return classifiers[model_name], classifier_params[model_name]


class MLFlowExperiment:

    def __init__(self, data_path, target, experiment_name, regression=True,
                 n_splits=5, n_repeats=10, random_state=42, scoring=None,
                 scaler=None, models=None, tracking_uri=None,
                 feature_sets=None, fit_single_features=False):
        """ class to organize experiment"""

        self.data_path = Path(data_path)
        self.data = self.__load_data__()
        self.target = target
        self.experiment_name = experiment_name
        self.regression = regression
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.scoring = self.__set_scoring__(scoring)

        if not tracking_uri:
            tracking_uri = "http://127.0.0.1:8080"
        self.tracking_uri = tracking_uri

        self.set_experiment()

        # features sets to train on.
        if not feature_sets:
            feature_sets = dict(
                pos_cols=[col for col in self.data.columns if 'pos' in col and
                          ('1' in col or '2' in col or '3' in col or
                           '4' in col)],
                net_cols=[col for col in self.data.columns if 'net' in col and
                          ('1' in col or '2' in col or '3' in col or
                           '4' in col)]
            )
        self.feature_sets = feature_sets

        self.fit_single_features = fit_single_features
        if self.fit_single_features:
            self.single_features = {
                col: [col] for col in self.data.columns if col != self.target}

        # list of models to train for multivariate feature sets.
        if not models:
            if regression:
                models = ['ridge', 'lasso', 'elasticnet', 'et', 'rf']
            else:
                models = ['rf', 'logistic', 'et']
        self.models = models

        if not scaler:
            scaler = StandardScaler
        self.scaler = scaler

    def __set_scoring__(self, scoring):
        """set scoring for regression or classification
        Args:
            scoring (list): list of scoring metrics
        Returns:
            scoring (list): list of scoring metrics
        """
        if scoring:
            return scoring
        elif self.regression:
            scoring = ['neg_root_mean_squared_error', 'r2']
        else:
            scoring = ['roc_auc', 'average_precision', 'balanced_accuracy',
                       'accuracy', 'f1', 'precision', 'recall']
        return scoring

    def __load_data__(self):
        """load data from path"""
        if self.data_path.name.endswith('.csv'):
            return pd.read_csv(self.data_path)
        elif (self.data_path.name.endswith('.parquet') or
              self.data_path.name.endswith('.pq')):
            return pd.read_parquet(self.data_path)

    def get_cv_splitter(self, X, y):
        if self.regression:
            return split_continous(X, y, n_splits=self.n_splits,
                                   n_repeats=self.n_repeats,
                                   random_state=self.random_state)
        else:
            return RepeatedStratifiedKFold(n_splits=self.n_splits,
                                           n_repeats=self.n_repeats,
                                           random_state=self.random_state
                                           ).split(X, y)

    def gridsearch(self, model_name, features, single_feature=False):
        """Train model with gridsearch and log results to mlflow"""

        # set up the model and training data
        model, params = get_model(model_name, regression=self.regression)
        n_jobs_for_search = 4 if model_name in ['rf', 'et'] else -1

        pipes = Pipeline([('scaler', self.scaler()),
                          (model_name, model)])

        if single_feature:
            cols = self.single_features[features]
        else:
            cols = self.feature_sets[features]

        X, y = self.data[cols].copy(), self.data[self.target].copy()
        cv = self.get_cv_splitter(X, y)
        grid = GridSearchCV(pipes, params, scoring=self.scoring, cv=cv,
                            n_jobs=n_jobs_for_search, refit=self.scoring[0],
                            verbose=1, return_train_score=True)

        with mlflow.start_run():

            mlflow.set_tag('model', model_name)
            mlflow.set_tag('feature_set', features)
            mlflow.set_tag('scoring', self.scoring)
            mlflow.set_tag('Number_of_features', len(cols))

            grid.fit(X, y)

            # Log metrics to mlflow.
            cv = self.get_cv_splitter(X, y)
            mean_predictions, predictions_df = get_average_predictions(
                X, y, grid.best_estimator_, cv=cv, regression=self.regression)
            y_pred = mean_predictions['y_pred_mean']

            predictions_file_name = Path().cwd() / 'predictions.csv'
            predictions_df.to_csv(predictions_file_name)
            mlflow.log_artifact(predictions_file_name)
            predictions_file_name.unlink()

            try:
                fig, _ = plot_feature_importances(
                    grid.best_estimator_, model_name=model_name)
                mlflow.log_figure(fig, 'feature_importances.png')
            except ValueError:
                pass

            scores = extract_train_test_cv_scores(
                grid, scores=self.scoring, n_total=(self.n_splits *
                                                    self.n_repeats))
            mlflow.log_metrics(scores)

            if self.regression:
                mlflow.log_metrics({'mean_cv_mse': mean_squared_error(y,
                                                                      y_pred),
                                    'mean_cv_r2': r2_score(y, y_pred)})

                fig, _ = plot_regression_performance_from_predictions(
                    y, y_pred, model_name=model_name)
                mlflow.log_figure(fig, 'prediction_error.png')

            else:
                mets = {'roc_auc': roc_auc_score(y, y_pred),
                        'precision_recall': average_precision_score(y, y_pred),
                        'accuracy': accuracy_score(y, (y_pred > 0.5)),
                        'f1': f1_score(y,  (y_pred > 0.5))}
                mlflow.log_metrics(mets)

                fig, _ = plot_classification_performance_from_predictions(
                    y, y_pred, model_name=model_name)
                mlflow.log_figure(fig, 'classification_performance.png')

            mlflow.log_params(grid.best_params_)
            try:
                mlflow.log_params(grid.cv_results_)
            except ValueError:
                pass

            signature = infer_signature(X, y)
            mlflow.sklearn.log_model(
                grid.best_estimator_, model_name, signature=signature)

            return grid.best_estimator_, grid.best_params_, grid.best_score_

    def set_experiment(self):
        """Set up mlflow experiment"""

        mlflow.set_tracking_uri(uri=self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def train_models(self):
        """Train models and log results to mlflow"""
        for features in tqdm(self.feature_sets, total=len(self.feature_sets)):
            for model_name in self.models:
                self.gridsearch(model_name, features)

        if self.fit_single_features:
            self.train_single_features()

    def train_single_features(self):
        """Train models on single features and log results to mlflow"""
        if self.regression:
            model_name = 'linear'
        else:
            model_name = 'logistic'

        for col in tqdm(self.single_features, total=len(self.single_features)):
            self.gridsearch(model_name, col, single_feature=True)
