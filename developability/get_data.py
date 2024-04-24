import sarge
from io import StringIO
import pandas as pd
from pathlib import Path
import developability.data as data

experimental_data_path = Path(data.__path__[0]) / 'experimental_data'


def get_all_data_for_project(project, savepath=None, filename=None):
    """Uses curl to get the data
    Args:
        project(str)
    Returns:
        None
    """

    print(f'Downloading data for project {project}')
    cmd = f'curl daisydata.dev.ds.vir.bio/joined/{project}'
    res = sarge.get_both(cmd)

    data_df = (pd.read_csv(StringIO(res[0]))
               .rename(columns={'Unnamed: 0': 'antibody'})
               )
    data_df.insert(0, 'Project', project)

    if savepath:
        savepath = Path(savepath)

        if not filename:
            filename = f'{project}_daisy_data.csv'

        path = savepath/filename
        print(f'Saving data for project {project} to {path}')
        data_df.to_csv(path)

    return data_df


def get_all_data_for_all_projects(projects=None,
                                  savepath=experimental_data_path):
    """Downloads all the data"""

    if savepath:
        if isinstance(savepath, str):
            savepath = Path(savepath)

        if not savepath.exists():
            savepath.mkdir()

    if not projects:
        projects = projects = ["FNI9v81", "MPK176", "MPK190", "MPK201",
                               "MPK65", "PIA38", "RSD5"]

    return {project: get_all_data_for_project(project, savepath) for project
            in projects}
