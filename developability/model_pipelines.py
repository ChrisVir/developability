from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OrdinalEncoder,
                                   FunctionTransformer, OneHotEncoder)
from sklego.preprocessing import FormulaicTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

from sklearn import set_config
set_config(transform_output="pandas")


def create_conditional_intercept_formula(intercept, cols):
    """ Generates a formula for conditional intercept"""
    formula = [intercept] + cols
    return ' + '.join(formula)


def create_conditional_intercept_slope_formula(intercept, cols):
    """generates a formula for conditional intercept and slope"""
    formula = [intercept] + cols + [f'{intercept}:{col}' for col in cols]
    return ' + '.join(formula)


def create_formula(intercept, cols, include_slope=False):
    if include_slope:
        return create_conditional_intercept_slope_formula(intercept, cols)
    else:
        return create_conditional_intercept_formula(intercept, cols)


def make_formula_transform(intercept, features, include_slope=False,
                           categories=None):
    """ create a tranform pipeline for using intercept"""

    formula = create_formula(intercept, features, include_slope)
    transformer = FormulaicTransformer(formula, return_type='pandas')
    return transformer


def make_column_tranformer(intercept, descriptors, categories=None,
                           scaler=MinMaxScaler):
    if not categories:
        categories = 'auto'

    intercept_transformer = ('intercept',
                             OrdinalEncoder(categories=categories,
                                            dtype=np.int32),
                             [intercept]
                             )

    descriptor_transformer = ('descriptors', scaler(), descriptors)

    return ColumnTransformer(transformers=[intercept_transformer,
                                           descriptor_transformer],
                             verbose_feature_names_out=False)


def make_formula_pipeline(descriptors, intercept=None, model=None,
                          include_slope=True, categories=None,
                          scaler=MinMaxScaler, model_name='model'):
    """create a formula pipeline for features"""

    if not intercept:
        intercept = descriptors[0]
        descriptors = descriptors[1:]
    else:
        descriptors = [col for col in descriptors
                       if col != intercept]

    ct = make_column_tranformer(intercept, descriptors, categories)
    ft = make_formula_transform(intercept, descriptors, include_slope)

    steps = [('column_transform', ct), ('formula_transform', ft)]
    if model is not None:
        steps.append((model_name,  model))
    pipeline = Pipeline(steps)
    return pipeline


def get_formulaic_column_names(pipeline, step=1):
    return pipeline[step].model_spec_.column_names


def make_pca_pipeline(descriptors, intercept=None, model=None,
                      categories='auto', scaler=MinMaxScaler,
                      model_name=None, n_components=5,
                      return_column_transformer=False,
                      use_intercept=True):

    descriptors = [col for col in descriptors if col != intercept]

    if use_intercept:
        intercept_transformer = ('intercept',
                                 OneHotEncoder(categories=categories,
                                               dtype=np.int32,
                                               handle_unknown='ignore',
                                               sparse_output=False),
                                 [intercept]
                                 )
    # set up the steps for pca.
    steps = [('scaler', scaler()),
             ('pca', PCA(n_components=n_components))
             ]

    pca_transformer = ('pca', Pipeline(steps), descriptors)

    if use_intercept:
        transformers = [intercept_transformer, pca_transformer]
    else:
        transformers = [pca_transformer]

    ct = ColumnTransformer(transformers=transformers)

    if return_column_transformer:
        return ct

    if model is None:
        print(model_name)
        model = LinearRegression()
        model_name = 'linear'

    pipes = Pipeline([('column_transformer', ct), (model_name, model)])

    return pipes


def make_pca_pipeline_with_intercept(descriptors, intercept=None, model=None,
                                     categories='auto', scaler=MinMaxScaler,
                                     model_name=None, n_components=5,
                                     return_column_transformer=False):
    """Returns a pca_pipeline using intercept.
    """

    return make_pca_pipeline(descriptors, intercept, model, categories,
                             MinMaxScaler, model_name, n_components,
                             return_column_transformer, use_intercept=True)


def normalize_by_row(df):
    """Normalizes a row by dividing by total
    Args:
        df(pd.DataFrame)
    Returns:
        pd.DataFrame
    """
    total = df.sum(axis=1)
    return df.divide(total, axis=0)


def row_total(df):
    """computes the total
    Args:
        df(pd.DataFrame)
    Returns:
        pd.DataFrame
    """
    total = pd.DataFrame(df.sum(axis=1))
    total.columns = ['Total']
    return total['Total']


def make_normalize_transformer():
    """Creates a Function Transformer with normalize_by_row"""
    return FunctionTransformer(normalize_by_row,
                               feature_names_out='one-to-one')


def total_name():
    return 'Total'


def make_row_total_transformer(name='charge'):
    return FunctionTransformer(row_total, feature_names_out=total_name)


def make_normalize_transformer_pipeline(descriptors, total_column=None,
                                        intercept='Project',
                                        model=None, scaler=MinMaxScaler,
                                        categories='auto', model_name=None,
                                        return_column_transformer=False,
                                        use_intercept=True):

    if not total_column:
        total_column = [col for col in descriptors if 'total' in col.lower()]
    else:
        if isinstance(total_column, str):
            total_column = [total_column]

    descriptors = [col for col in descriptors
                   if col != intercept
                   and 'total' not in col.lower()]
    if use_intercept:
        # Consider refactoring this to own function DRY
        intercept_transformer = ('intercept',
                                 OneHotEncoder(categories=categories,
                                               dtype=np.int32,
                                               handle_unknown='ignore',
                                               sparse_output=False),
                                 [intercept]
                                 )

    normalize_transformer = ('normalize_by_row',
                             make_normalize_transformer(),
                             descriptors)

    total_transformer = ('total',
                         scaler(),
                         total_column)

    if use_intercept:
        transformers = [intercept_transformer, normalize_transformer,
                        total_transformer]
    else:
        transformers = [normalize_transformer, total_transformer]

    ct = ColumnTransformer(transformers=transformers)

    if return_column_transformer:
        return ct

    if model is None:
        model = LinearRegression()
        model_name = 'linear'

    pipes = Pipeline([('column_transformer', ct), (model_name, model)])

    return pipes
