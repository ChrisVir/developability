### Tools for modeling 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PredictionErrorDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold




### Utils 
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
        splitter  = cv.split(X)
    else: 
        splitter = cv

    predictions= {}
    
    for train_index, test_index in splitter:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index] 
        model.fit(X_train, y_train)
        if regression:
            y_pred = model.predict(X_test)
        else: 
            y_pred = model.predict_proba(X_test)[:,1]

        for i, idx in enumerate(test_index):
            predictions.setdefault(idx, []).append(y_pred[i])
        
    predictions_df = pd.DataFrame(predictions).T.sort_index()
    predictions_df.index = X.index
    #predictions_df.index.name = 'antibody'

    mean_predictions = pd.DataFrame(predictions_df.mean(axis = 1), columns = ['y_pred_mean'])
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

### Visualization
def plot_feature_importances(model, n_features=10, ax=None, model_name=None):
    """
    Plot feature importances or coefficients of a model.

    Args:
    - model: The trained model object.
    - n_features: The number of top features to display (default is 10).
    - ax: The matplotlib axes object to plot the feature importances (default is None).
    - model_name: The name of the model (optional).
    - axes: The matplotlib axes object to plot the feature importances (default is None).

    Returns:
    - ax: The matplotlib axes object containing the feature importances plot.

    Note:
    - This function assumes that the model object has either a 'feature_importances_' or 'coef_' attribute.

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
        fig, ax = plt.subplots(1,1, figsize=(8, 6))
        fig.set_size_inches(9, 5)
        ax = s.plot(kind='barh', ax=ax)

    ax.set_title(title)
    plt.tight_layout()
    sns.despine()

    
    return fig, ax


def plot_regression_performance(model, X, y, model_name= None, dataset='validation', figsize = (8, 3)):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    _ = PredictionErrorDisplay.from_estimator(model, X, y, kind="actual_vs_predicted", ax=axes[0])
    
    _ = PredictionErrorDisplay.from_estimator(model, X, y, kind="residual_vs_predicted", ax=axes[1])
    sns.despine()
    fig.suptitle(f'Regression Performance for {model_name} on {dataset} dataset', fontsize=14)
    fig.tight_layout()
    return fig, axes



def plot_regression_performance_from_predictions( y_true, y_pred, model_name= None, dataset='validation', figsize = (8, 4)):
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

    max_val = int(np.ceil(max(y_true.max(), y_pred.max())))+1
    min_val = int(np.floor(min(y_true.min(), y_pred.min())))-1
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax = axes[0]
    _=PredictionErrorDisplay.from_predictions( y_true, y_pred, kind="actual_vs_predicted", ax=ax)
    ax.set(xlim=(min_val, max_val), ylim=(min_val, max_val))
    ax.set_xticks(range(min_val, max_val, 2) )
    ax.set_yticks(range(min_val, max_val, 2) )
    

    ax = axes[1]
    _ = PredictionErrorDisplay.from_predictions( y_true, y_pred, kind="residual_vs_predicted", ax=ax)
    ax.set(xlim=(min_val, max_val))
    ax.set_xticks(range(min_val, max_val, 2) )
    
    sns.despine()
    fig.suptitle(f'Regression Performance for {model_name} on {dataset} dataset', fontsize=14)
    fig.tight_layout()
    return fig, axes


def plot_classification_performance_from_predictions( y_true, y_pred, model_name= None, dataset='validation', figsize = (8, 4)):
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
    _ = RocCurveDisplay.from_predictions( y_true, y_pred, ax=ax)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    
    ax = axes[1]
    _ = PrecisionRecallDisplay.from_predictions( y_true, y_pred, ax=ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    sns.despine()
    fig.suptitle(f'Classification Performance for {model_name} on {dataset} dataset', fontsize=14)
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
        raise ValueError("Unsupported data type. Expected pandas.Series or pandas.DataFrame.")


def split_continous(X, y, n_splits=5, n_repeats =10, random_state = 42): 
    """Split data into quantiles and then split into train and test sets
    Args:
        X (pandas.DataFrame): The data to be binned.
        y (pandas.Series): The number of quantiles to create.
        n_splits (int): number of splits
        n_repeats (int): number of repeats
        random_state (int): random state
    Yields:
        train_index, test_index (tuple): train and test indices    
    """

    y_bined = bin_data_into_quantiles(y, num_bins = n_splits)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    for train_index, test_index in rskf.split(X, y_bined):
        yield train_index, test_index
    