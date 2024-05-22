import numpy as np
import pandas as pd
import mlflow
from mlflow.sklearn import (get_default_conda_env,
                            save_model)


def runs_to_dataframe(runs):
    """Converts a Mflow

    Args:
        runs (mlflow.store.entities.paged_list.PagedList): a list of
             mlflow.entities.run.Run json like objects that stores metadata.

    Returns:
        pd.DataFrame: dataframe with parsed metadata.
    """
    info = {
        "run_id": [],
        "experiment_id": [],
        "status": [],
        "artifact_uri": [],
        "start_time": [],
        "end_time": [],
    }
    params, metrics, tags = ({}, {}, {})
    PARAM_NULL, METRIC_NULL, TAG_NULL = (None, np.nan, None)
    for i, run in enumerate(runs):
        info["run_id"].append(run.info.run_id)
        info["experiment_id"].append(run.info.experiment_id)
        info["status"].append(run.info.status)
        info["artifact_uri"].append(run.info.artifact_uri)
        info["start_time"].append(pd.to_datetime(
            run.info.start_time, unit="ms", utc=True))
        info["end_time"].append(pd.to_datetime(
            run.info.end_time, unit="ms", utc=True))

        # Params
        param_keys = set(params.keys())
        for key in param_keys:
            if key in run.data.params:
                params[key].append(run.data.params[key])
            else:
                params[key].append(PARAM_NULL)
        new_params = set(run.data.params.keys()) - param_keys
        for p in new_params:
            # Fill in null values for all previous runs
            params[p] = [PARAM_NULL] * i
            params[p].append(run.data.params[p])

        # Metrics
        metric_keys = set(metrics.keys())
        for key in metric_keys:
            if key in run.data.metrics:
                metrics[key].append(run.data.metrics[key])
            else:
                metrics[key].append(METRIC_NULL)
        new_metrics = set(run.data.metrics.keys()) - metric_keys
        for m in new_metrics:
            metrics[m] = [METRIC_NULL] * i
            metrics[m].append(run.data.metrics[m])

        # Tags
        tag_keys = set(tags.keys())
        for key in tag_keys:
            if key in run.data.tags:
                tags[key].append(run.data.tags[key])
            else:
                tags[key].append(TAG_NULL)
        new_tags = set(run.data.tags.keys()) - tag_keys
        for t in new_tags:
            tags[t] = [TAG_NULL] * i
            tags[t].append(run.data.tags[t])

    data = {}
    data.update(info)
    for key, value in metrics.items():
        data["metrics." + key] = value
    for key, value in params.items():
        data["params." + key] = value
    for key, value in tags.items():
        data["tags." + key] = value

    return pd.DataFrame(data)


def get_columns(df, suffixes=None, prefixes=None, infixes=None, exclude=None):
    """Select columns from a dataframe using simple rules.
    Args:
        df (pd.DataFrame): Dataframe.
        suffixes (list[str]): If has suffix keep. Defaults to None.
        prefixes (list[str]): If have prefix keep. Defaults to None.
        infixes (list[str]): If have internal infix keep. Defaults to None.
        exclude (list[str]): If in exclude do not keep. Defaults to None.
    """

    def keep(col):
        for ex in exclude:
            if ex in col:
                return False

        return True

    cols = []
    if prefixes:
        for prefix in prefixes:
            cols.extend([col for col in df.columns if col.startswith(
                prefix) and col not in cols])
    if infixes:
        for infix in infixes:
            cols.extend(
                [col for col in df.columns if 'infix' in col
                 and col not in cols])

    if suffixes:
        for suffix in suffixes:
            cols.extend([col for col in df.columns if col.endswith(
                suffix) and col not in cols])

    if exclude:
        cols = [col for col in cols if keep(col)]

    return df[cols]


def extract_metadata_from_signature(model_path):
    """extracts the model target names from a model signature"""
    from mlflow.models import get_model_info

    def get_outputs(signature):
        return [output['name'] for output in signature.outputs.to_dict()]

    def get_inputs(signature):
        return [input['name'] for input in signature.inputs.to_dict()]

    signature = get_model_info(model_path).signature

    return get_inputs(signature), get_outputs(signature)


def save_sklearn_model_to_new_location(source_path, destination_path):
    """saves an sklearn model to a new location
    Note can not use predict proba with this.
    """
    model = mlflow.sklearn.load_model(source_path)
    signature = mlflow.pyfunc.load_model(source_path).metadata.signature

    if not destination_path.exists():
        destination_path.mkdir()

    save_model(sk_model=model,
               path=destination_path,
               signature=signature,
               serialization_format='pickle',
               conda_env=get_default_conda_env(False))
