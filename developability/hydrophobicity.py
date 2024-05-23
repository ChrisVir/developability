import pandas as pd
from pathlib import Path
import numpy as np
import faiss
from developability.descriptors import map3to1
from developability.input_output import read_pqr
import developability
from matplotlib import pyplot as plt


def read_hydrophobicity_scales():
    """Read hydrophobicity scales from csv file.
    Args:
        None
    Returns:
        pd.DataFrame: dataframe of hydrophobicity scales.
    """
    fname = 'hydrophobicity_scales.csv'
    return pd.read_csv(Path(developability.__path__[0])/fname, index_col=0)


def load_hydrophobicity_scale(scale='Eisenberg'):
    """Load a hydrophobicity scale.
    Args:
        scale (str): name of scale to load.
    Returns:
        dict: dictionary of amino acid hydrophobicity values.
    """
    scales = pd.read_csv(Path(developability.__path__[
                         0])/'hydrophobicity_scales.csv', index_col=0)
    return scales[scale].to_dict()


def map_hydrophobicity_scale_to_residues_in_pqr(pqr_df, scale='Eisenberg'):
    """Map a hydrophobicity scale to a pqr file.
    Args:
        pqr_df(pd.DataFrame): dataframe of pqr file.
        scale (str): name of scale to load.
    Returns:
        pd.DataFrame: dataframe of pqr file with hydrophobicity column.
    """

    pqr_df = pqr_df.copy().dropna()
    mapping = load_hydrophobicity_scale(scale)
    residues = pd.Series(map3to1(pqr_df['Residue_name'].values))
    pqr_df['Hydrophobicity'] = residues.map(mapping)
    return pqr_df


def find_nearby_vertices(vertices, radius=0.8):
    """Find nearby vertices within a given radius using Meta Faiss.
    Args:
        vertices (pd.DataFrame or np.array): DataFrame or array of vertices.
        radius (float): Radius to search within.

    Returns:
        dict: Dictionary of vertices and their neighbors.
    """

    index = faiss.IndexFlatL2(3)

    if isinstance(vertices, pd.DataFrame):
        vectors = vertices[['x', 'y', 'z']].values.astype('float32')
    elif isinstance(vertices, np.ndarray):
        vectors = vertices.astype('float32')
    else:
        raise ValueError('Vertices must be a DataFrame or numpy array')

    index.add(vectors)
    lims, D, index = index.range_search(vectors, radius)

    neighbors = {}
    for i in range(len(lims)-1):
        neighbors[i] = index[lims[i]:lims[i+1]]

    return neighbors, lims, D, index


def compute_feature_averages(df, radius=0.8, feature='Hydrophobicity',
                             weight='uniform'):
    """Compute weighted averages of a feature for each vertex.
    Args:
        df(pd.DataFrame): dataframe of vertices.
        neighbors(dict): dictionary of vertices and their neighbors.
        feature(str): name of feature to compute weighted average for.
    Returns:
        pd.DataFrame: dataframe of vertices with weighted averages.
    """

    neighbors, lims, D, index = find_nearby_vertices(df, radius=radius)

    df = df.copy()
    new_feature = []
    for i in range(len(df)):
        if weight == 'distance':
            weights = 1/D[lims[i]:lims[i+1]]
        elif weight == 'uniform':
            weights = np.ones(len(neighbors[i]))
        elif weights == 'softmax':
            weights = np.exp(-D[lims[i]:lims[i+1]])
            weights = weights/np.sum(weights)
        else:
            raise ValueError('weight must be distance, uniform or softmax')
        new_feature.append(np.average(
            df.loc[neighbors[i], feature], weights=weights))

    df[f'{feature}_avg'] = new_feature
    return df


def aggregate_hydrophobicity_over_residues(pqr_df, atom_surface_df, radius=0.8,
                                           weight='uniform', scale='Eisenberg',
                                           agg_func=np.mean):
    """Aggregate hydrophobicity over residues.
    Args:
        pqr_df(pd.DataFrame): dataframe of pqr file.
        atom_surface_df(pd.DataFrame): dataframe of atom surface.
        radius(float): radius to search within.
        weight(str): type of weighting to use.
        scale(str): name of hydrophobicity scale to use.
        agg_func(function): function to aggregate over atoms and residues.
                            Default = np.mean.
    Returns:
        pd.DataFrame: dataframe of vertices with weighted averages.
    """
    hydros = map_hydrophobicity_scale_to_residues_in_pqr(
        pqr_df, scale=scale)

    atom_hydros = atom_surface_df.merge(right=hydros[['Atom_number',
                                                      'Hydrophobicity']],
                                        left_on='atom',
                                        right_on='Atom_number')

    vertex_hydros = compute_feature_averages(
        atom_hydros, radius=radius, feature='Hydrophobicity', weight=weight)

    atom_hydros = (vertex_hydros.groupby('atom')
                   .agg({'Hydrophobicity_avg': agg_func})
                   .reset_index()
                   )

    residue_hydros = pqr_df.merge(
        right=atom_hydros, left_on='Atom_number', right_on='atom', how='left')

    residue_hydros = (residue_hydros
                      .groupby(['Residue_number', 'Residue_name'])
                      .agg({'Hydrophobicity_avg': agg_func})
                      .reset_index()
                      .rename(columns={'Hydrophobicity_avg': 'Hydrophobicity'})
                      )

    return atom_hydros, vertex_hydros, residue_hydros


class HydrophobicSurface:
    """
    Represents a hydrophobic surface.

    Attributes:
        pqr (str): The path to the PQR file.
        atom_surface (str): The path to the atom surface file.
        radius (float): The radius used for aggregation.
        weight (str): The weight function used for aggregation.
        scale (str): The scale used for hydrophobicity calculation.
        agg_func (function): The func to aggregate hydrophobicty.
        atom_hydros (pd.DataFrame): hydrophobicities for atoms.
        vertex_hydrophobicities (pd.DataFrame): hydrophobicities for vertices.
        residue_hydrophobicities (pd.DataFrame): hydrophobicities for residues.
    """

    def __init__(self, pqr, atom_surface, radius=0.8, weight='uniform',
                 scale='Eisenberg', agg_func=np.mean):

        if isinstance(pqr, str) or isinstance(pqr, Path):
            pqr = read_pqr(pqr)
        self.pqr = pqr

        if isinstance(atom_surface, str) or isinstance(atom_surface, Path):
            atom_surface = pd.read_csv(atom_surface)
        self.atom_surface = atom_surface
        self.radius = radius
        self.weight = weight
        self.scale = scale
        self.agg_func = agg_func
        hydros = aggregate_hydrophobicity_over_residues(
            pqr, atom_surface, radius=radius, weight=weight, scale=scale,
            agg_func=agg_func)

        self.atom_hydros, self.vertex_hydros, self.residue_hydros = hydros

    def save(self, fname):
        """Save the HydropbicSurface object to a file.
        Args:
            fname(str): name of file to save to.
        Returns:
            None
        """
        self.residue_hydros.to_csv(fname, index=False)
        return None

    @property
    def residues(self):
        """Return the residues in the HydropbicSurface object.
        Args:
            None
        Returns:
            pd.DataFrame: dataframe of residues.
        """
        return self.residue_hydros

    @property
    def vertices(self):
        """Return the vertices in the HydropbicSurface object.
        Args:
            None
        Returns:
            pd.DataFrame: dataframe of vertices.
        """
        return self.vertex_hydros

    @property
    def atoms(self):
        """Return the atoms in the HydropbicSurface object.
        Args:
            None
        Returns:
            pd.DataFrame: dataframe of atoms.
        """
        return self.atom_hydros

    def plot(self, feature='Hydrophobicity', cmap='viridis', vmin=None,
             vmax=None, ax=None, **kwargs):
        """Plot the HydropbicSurface object.
        Args:
            feature(str): name of feature to plot.
            cmap(str): name of colormap to use.
            vmin(float): minimum value for colormap.
            vmax(float): maximum value for colormap.
            ax(matplotlib.axes): axes to plot on.
            **kwargs: additional arguments to pass to matplotlib.pyplot.
        Returns:
            matplotlib.axes: axes containing plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(self.vertices['x'], self.vertices['y'],
                   c=self.vertices[feature], cmap=cmap,
                   vmin=vmin, vmax=vmax, **kwargs)
        return ax


def region_hydrophobicities(residue_hydro_df, hydro_col='Hydrophobicity'):
    """ Calculates the potentials for regions
    TODO: refactor this function to make cleaner.
    Args:
        residue_hydrophobicity_df(pd.DataFrame): residue hydrophobicities
        hydrophobicity_col(str): name of column with hydrophobicity values
    Returns:
        vals(dict): dict with values
    """

    def calc_hydro(region, chain, sign='pos'):
        """ Calculates the hydrophobicity
        Args:
            region(str|list[str]): the region to calculate
            chain(str|list[str]): the chain to calculate
            sign(str): (pos, neg, net)
        Returns:
            float: the potential for the region
        """

        if isinstance(region, str):
            region = [region]

        if isinstance(chain, str):
            chain = [chain]

        if sign == 'pos':
            df = residue_hydro_df.loc[residue_hydro_df[hydro_col] > 0]
        elif sign == 'neg':
            df = residue_hydro_df.loc[residue_hydro_df[hydro_col] < 0]
        elif sign == 'net':
            df = residue_hydro_df
        else:
            raise ValueError('hydrophobicity must be pos, neg or net')
        return (df.loc[df.FV_region.isin(region) & df.FV_chain.isin(chain)]
                [hydro_col].sum()
                )

    cdrs = ['CDR1', 'CDR2', 'CDR3']
    frameworks = ['FR1', 'FR2', 'FR3', 'FR4']
    all_regions = cdrs + frameworks
    chains = ['H', 'L']

    hydros = ['pos', 'neg', 'net']

    vals = {f'{chain}{region}_HYDRO_{hydro_sign}':
            calc_hydro(region, chain, hydro_sign) for
            chain in chains for region in all_regions for
            hydro_sign in hydros}

    vals.update({f'{chain}CDR_HYDRO_{hydro}': calc_hydro(cdrs, chain, hydro)
                 for chain in chains for hydro in hydros})

    vals.update({f'{chain}FR_HYDRO_{hydro}': calc_hydro(frameworks,
                                                        chain, hydro)
                 for chain in chains for hydro in hydros})

    vals.update({f'{chain}C_HYDRO_{hydro}': calc_hydro(all_regions,
                                                       chain, hydro)
                 for chain in chains for hydro in hydros})

    vals.update({f'TOTAL_CDR_HYDRO_{hydro}': calc_hydro(cdrs, chains, hydro)
                 for hydro in hydros})

    vals.update({f'TOTAL_FR_HYDRO_{hydro}': calc_hydro(frameworks,
                                                       chains, hydro)
                 for hydro in hydros})

    vals.update({f'TOTAL_HYDRO_{hydro}': calc_hydro(all_regions, chains, hydro)
                 for hydro in hydros})

    return vals
