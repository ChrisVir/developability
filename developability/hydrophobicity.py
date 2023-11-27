import pandas as pd
from pathlib import Path
import numpy as np
import faiss
from developability.descriptors import map3to1
import developability
from matplotlib import pyplot as plt


def load_hydrophobicity_scale(scale='Eisenberg'): 
    """Load a hydrophobicity scale. 
    Args:
        scale (str): name of scale to load.
    Returns: 
        dict: dictionary of amino acid hydrophobicity values.
    """
    scales = pd.read_csv(Path(developability.__path__[0])/'hydrophobicity_scales.csv', index_col=0)
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
    pqr_df['Hydrophobicity']= residues.map(mapping)
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
    lims, D, I = index.range_search(vectors, radius)

    neighbors = {}
    for i in range(len(lims)-1): 
        neighbors[i] = I[lims[i]:lims[i+1]]

    return neighbors, lims, D, I


def compute_feature_averages(df, radius=0.8, feature='Hydrophobicity', weight = 'uniform'): 
    """Compute weighted averages of a feature for each vertex. 
    Args:
        df(pd.DataFrame): dataframe of vertices.
        neighbors(dict): dictionary of vertices and their neighbors.
        feature(str): name of feature to compute weighted average for.
    Returns:
        pd.DataFrame: dataframe of vertices with weighted averages.
    """

    neighbors, lims, D, I = find_nearby_vertices(df, radius=radius)

    df = df.copy()
    new_feature = []
    for i in range(len(df)): 
        if weight == 'distance': 
            weights = 1/D[lims[i]:lims[i+1]]
        elif weight == 'uniform': 
            weights = np.ones(len(neighbors[i]))
        elif weights=='softmax': 
            weights = np.exp(-D[lims[i]:lims[i+1]])
            weights = weights/np.sum(weights)
        else: 
            raise ValueError('weight must be distance, uniform or softmax')
        new_feature.append( np.average(df.loc[neighbors[i], feature], weights=weights))
    
    df[f'{feature}_avg'] = new_feature
    return df


def aggregate_hydrophobicity_over_residues(pqr_df, atom_surface_df, radius=0.8, weight='uniform', scale='Eisenberg', 
                                           agg_func=np.mean): 
    """Aggregate hydrophobicity over residues. 
    Args:
        pqr_df(pd.DataFrame): dataframe of pqr file.
        atom_surface_df(pd.DataFrame): dataframe of atom surface.
        radius(float): radius to search within.
        weight(str): type of weighting to use.
        scale(str): name of hydrophobicity scale to use.
        agg_func(function): function to aggregate over atoms and residues. Default = np.mean. 
    Returns:
        pd.DataFrame: dataframe of vertices with weighted averages.
    """
    hydrophobicities = map_hydrophobicity_scale_to_residues_in_pqr(pqr_df, scale='Eisenberg')
    atom_hydrophobicities = atom_surface_df.merge(right = hydrophobicities[['Atom_number', 'Hydrophobicity']], left_on='atom', right_on='Atom_number')
    vertex_hydrophobicities = compute_feature_averages(atom_hydrophobicities, radius=radius, feature='Hydrophobicity', weight=weight)
    
    atom_hydrophobicities = (vertex_hydrophobicities.groupby('atom')
                             .agg({'Hydrophobicity_avg': agg_func})
                             .reset_index()
                             )
    
    residue_hydrophobicities = pqr_df.merge(right=atom_hydrophobicities, left_on='Atom_number', right_on='atom', how = 'left')

    residue_hydrophobicities = (residue_hydrophobicities
                                .groupby(['Residue_number', 'Residue_name'])
                                .agg({'Hydrophobicity_avg': agg_func})
                                .reset_index()
                                .rename(columns={'Hydrophobicity_avg': 'Hydrophobicity'})
                                )   

    return atom_hydrophobicities, vertex_hydrophobicities, residue_hydrophobicities


class HydrophobicSurface: 
    """
    Represents a hydrophobic surface.

    Attributes:
        pqr (str): The path to the PQR file.
        atom_surface (str): The path to the atom surface file.
        radius (float): The radius used for aggregation.
        weight (str): The weight function used for aggregation.
        scale (str): The scale used for hydrophobicity calculation.
        agg_func (function): The aggregation function used for hydrophobicity calculation.
        atom_hydrophobicities (pd.DataFrame): DataFrame of hydrophobicities for each atom.
        vertex_hydrophobicities (pd.DataFrame): DataFrame of hydrophobicities for each vertex.
        residue_hydrophobicities (pd.DataFrame): DataFrame of hydrophobicities for each residue.
    """

    def __init__(self, pqr, atom_surface, radius=0.8, weight='uniform', scale='Eisenberg', agg_func=np.mean): 
        self.pqr = pqr
        self.atom_surface = atom_surface
        self.radius = radius
        self.weight = weight
        self.scale = scale
        self.agg_func = agg_func
        hydrophobicities = aggregate_hydrophobicity_over_residues(pqr, atom_surface, radius=radius, weight=weight, scale=scale, agg_func=agg_func)
        self.atom_hydrophobicities, self.vertex_hydrophobicities, self.residue_hydrophobicities = hydrophobicities
        self.vertex_hydrophobicities = self.vertex_hydrophobicities.merge(right=self.pqr[['Atom_number', 'Residue_number', 'Residue_name']], left_on='atom', right_on='Atom_number')
        
        self.residue_hydrophobicities = self.residue_hydrophobicities.merge(right=self.pqr[['Atom_number', 'Residue_number', 'Residue_name']], left_on='Residue_number', right_on='Residue_number')
        self.residue_hydrophobicities = self.residue_hydrophobicities.drop(columns=['Atom_number'])

    def __repr__(self): 
        return f'HydropbicSurface(pqr={self.pqr}, atom_surface={self.atom_surface}, radius={self.radius}, weight={self.weight}, scale={self.scale}, agg_func={self.agg_func})'
    
    def __str__(self): 
        return f'HydropbicSurface(pqr={self.pqr}, atom_surface={self.atom_surface}, radius={self.radius}, weight={self.weight}, scale={self.scale}, agg_func={self.agg_func})'
    
    def save(self, fname): 
        """Save the HydropbicSurface object to a file.
        Args:
            fname(str): name of file to save to.
        Returns:
            None
        """
        self.residue_hydrophobicities.to_csv(fname, index=False)
        return None
    
    @property
    def residues(self): 
        """Return the residues in the HydropbicSurface object.
        Args:
            None
        Returns:
            pd.DataFrame: dataframe of residues.
        """
        return self.residue_hydrophobicities
    
    @property
    def vertices(self): 
        """Return the vertices in the HydropbicSurface object.
        Args:
            None
        Returns:
            pd.DataFrame: dataframe of vertices.
        """
        return self.vertex_hydrophobicities
    
    @property
    def atoms(self): 
        """Return the atoms in the HydropbicSurface object.
        Args:
            None
        Returns:
            pd.DataFrame: dataframe of atoms.
        """
        return self.atom_hydrophobicities
    
    def plot(self, feature='Hydrophobicity', cmap='viridis', vmin=None, vmax=None, ax=None, **kwargs): 
        """Plot the HydropbicSurface object.
        Args:
            feature(str): name of feature to plot.
            cmap(str): name of colormap to use.
            vmin(float): minimum value for colormap.
            vmax(float): maximum value for colormap.
            ax(matplotlib.axes): axes to plot on.
            **kwargs: additional arguments to pass to matplotlib.pyplot.scatter.
        Returns:
            matplotlib.axes: axes containing plot.
        """
        if ax is None: 
            fig, ax = plt.subplots()
        ax.scatter(self.vertices['x'], self.vertices['y'], c=self.vertices[feature], cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        return ax
