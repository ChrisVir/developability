"""Main module."""
import click
import pandas as pd
from pathlib import Path
from datetime import datetime
from mlflow import sklearn
from mlflow.models import get_model_info


base_path = Path('/app/dev_models/')
model_paths = {'heparin_regression': base_path/'hep_regressor',
               'heparin_classification': base_path/'hep_classifier'
               }


def extract_metadata_from_signature(model_path):
    """extracts the model target names from a model signature"""

    def get_outputs(signature):
        return [output['name'] for output in signature.outputs.to_dict()]

    def get_inputs(signature):
        return [input['name'] for input in signature.inputs.to_dict()]

    signature = get_model_info(str(model_path)).signature

    return get_inputs(signature), get_outputs(signature)


def load_model(model_path):
    return sklearn.load_model(str(model_path))


def load_input_data(filepath, columns=None):
    filepath = Path(filepath)
    if filepath.suffix in ['.pq', '.parquet']:
        return pd.read_parquet(filepath, columns=columns)

    elif filepath.suffix == '.csv':
        return pd.read_csv(filepath, columns=columns)


def save_predictions(predictions, filename):
    """Save predictions in dataframe to file"""
    if filename.suffix in ['.pq', '.parquet']:
        predictions.to_parquet(filename)

    if filename.suffix == '.csv':
        predictions.to_csv(filename)


def batch_predict(model_path, input_data_path, predict_proba=False):
    """ Loads model and predicts"""

    model = load_model(model_path)
    inputs, targets = extract_metadata_from_signature(model_path)

    X = load_input_data(input_data_path, columns=inputs)

    if predict_proba:
        preds = model.predict_proba(X)[:, 1]
    else:
        preds = model.predict(X)

    predictions = pd.DataFrame(preds, index=X.index,
                               columns=targets)
    predictions.index.name = 'antibody'
    return predictions


@click.command()
@click.argument('input_data_path')
@click.option('--output_filename', default=None, help='name for output file')
@click.option('--output_dir', default=None, help='directory for output file')
def predict_heparin_binding(input_data_path, output_filename=None,
                            output_dir=None):
    """ predict model """

    model_path = model_paths['heparin_regression']
    predictions = batch_predict(model_path, input_data_path)

    if not output_dir:
        output_dir = Path(input_data_path).parent

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

    model_path = model_paths['heparin_classification']
    predictions = batch_predict(model_path, input_data_path, True)

    if not output_dir:
        output_dir = Path(input_data_path).parent

    if not output_filename:
        today = datetime.today().strftime('%m-%d-%Y')
        output_filename = f'heparin_classification_results_{today}.csv'

    save_predictions(predictions, output_dir/output_filename)
