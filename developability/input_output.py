import pandas as pd
from pathlib import Path


def read_pqr(pqr_file):
    """ reads a pqr file in as a dataframe
    Args:
        pqr_file(Path|str):
    Returns
        pd.DataFrame
    """
    names = ['Field_name', 'Atom_number', 'Atom_name', 'Residue_name',
             'Residue_number', 'X', 'Y', 'Z', 'Charge', 'Radius']
    df = pd.read_csv(pqr_file,
                     delim_whitespace=True,
                     header=None,
                     names=names)
    return df


def pqr_to_xyzr(pqr_file, output_path=None):
    """ convert pqr file to xyzr format for use in nanoshaper
    Args:
        pqr_file(Path|str):
        output_path(str|Path): path to directory for xyzr file.
    Returns
        pandas DataFrame
    """

    if not output_path:
        output_path = Path(pqr_file).with_suffix('.xyzr')

    # read in the pqr_file and select cols of interest.
    pqr = read_pqr(pqr_file)
    cols = ['X', 'Y', 'Z', 'Radius']
    xyzr = pqr.loc[pqr['Field_name'] == 'ATOM'][cols]

    xyzr.to_csv(output_path, sep=' ', index=False, header=False)
    return xyzr


def read_xyzr(xyzr_file):
    """ Reads a xyzr file in as a dataframe.
    Args:
        file(Path|str): path to the file
    Returns:
        pandas DataFrame
    """
    names = ['X', 'Y', 'Z', 'Radius']
    df = pd.read_csv(xyzr_file, delim_whitespace=True,
                     header=None, names=names)
    return df


def extract_xyz_atoms_from_off(off_file):
    """ Extracts the xyz coordinates and associated atoms from
        the given off surface file.
    Args:
        off_file(str|path): path to the off surface file from nanoshaper
    Returns:
        atoms_df(pd.DataFrame): dataframe with x, y, z, and atom columns
    NOTE: This should be moved.
    """
    text = Path(off_file).read_text().split('\n')
    n_vertices = int(text[3].split()[0])

    # read in the desired rows
    df = pd.read_csv(off_file,
                     skiprows=4,
                     nrows=n_vertices,
                     header=None,
                     names=['x', 'y', 'z', 'atom'],
                     delim_whitespace=True)
    return df


def save_surface_xyz_to_csv(off_file):
    """ Save the xyz atoms columns from surface off file to csv.
    Note this save the file to the same location.
    Args:
        off(Path/str): path to the file
    Returns:
        None
    """
    off_file = Path(off_file)
    df = extract_xyz_atoms_from_off(off_file)
    df.to_csv(off_file.with_suffix('.csv'),
              header=False,
              index=False,
              columns=['x', 'y', 'z']
              )
    return None


def save_surface_xyz_atoms_to_csv(off_file):
    """ Save the xyz and atom columns from off file to csv
    Args:
        off(Path/str): path to the file
    Returns:
        None
    """

    off_file = Path(off_file)
    df = extract_xyz_atoms_from_off(off_file)

    df.to_csv(off_file.with_suffix('.atom.csv'),
              index=False,
              columns=['x', 'y', 'z', 'atom']
              )
    return None


def read_text(file):
    """Reads a text file
    Args:
        file(str|Path): path to text file
    Returns:
        str

    """
    return Path(file).read_text()


def read_potential_file(potential_file):
    """read in a file with surface vertices and potential"""
    df = pd.read_csv(potential_file, header=None)
    df.columns = ['x', 'y', 'z', 'potential']
    return df


if __name__ == '__main__':
    pass
