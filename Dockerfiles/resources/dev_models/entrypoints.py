"""Main module."""
import click
import pandas as pd
from pathlib import Path
from datetime import datetime
from mlflow import pyfunc


base_path = Path('/app/dev_models/')
model_paths = {'heparin_regression': base_path/'hep_regressor',
               'heparin_classification': base_path/'heparin_classifier'
               }


def load_model(model_name):
    model_path = model_paths[model_name]
    return pyfunc.load_model(model_path)


def predict(model, X):
    """ predict with a model"""
    predictions = pd.DataFrame(model.predict(X), index=X.index)
    return predictions


def load_input_data(filepath):
    filepath = Path(filepath)
    if filepath.suffix in ['.pq', '.parquet']:
        return pd.read_parquet(filepath)

    elif filepath.suffix == '.csv':
        return pd.read_csv(filepath)


def save_predictions(predictions, filename):
    if filename.suffix in ['.pq', '.parquet']:
        predictions.to_parquet(filename)

    if filename.suffix == '.csv':
        predictions.to_csv(filename)


@click.command()
@click.argument('input_data_path')
@click.option('--output_filename', default=None, help='name for output file')
@click.option('--output_dir', default=None, help='directory for output file')
def predict_heparin_binding(input_data_path, output_filename=None,
                            output_dir=None):
    """ predict model """

    input_data_path = Path(input_data_path)
    X = load_input_data(input_data_path)

    model = load_model('heparin_regression')
    predictions = predict(model, X)

    if not output_dir:
        output_dir = input_data_path.parent

    if not output_filename:
        today = datetime.today().strftime('%m-%d-%Y')
        output_filename = f'heparin_regression_results_{today}.csv'

    save_predictions(predictions, output_dir/output_filename)


@click.command()
@click.argument('input_data_path')
@click.option('--output_filename', default=None, help='name for output file')
@click.option('--output_dir', default=None, help='directory for output file')
def classify_heparin_binding(input_data_path, output_filename=None,
                             output_dir=None):
    """ classify heparin_binding """

    input_data_path = Path(input_data_path)
    X = load_input_data(input_data_path)

    model = load_model('heparin_classification')
    predictions = predict(model, X)

    if not output_dir:
        output_dir = input_data_path.parent

    if not output_filename:
        today = datetime.today().strftime('%m-%d-%Y')
        output_filename = f'heparin_classification_results_{today}.csv'

    save_predictions(predictions, output_dir/output_filename)
