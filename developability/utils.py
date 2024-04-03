# utils 

from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from abnumber import Chain, ChainParseError
from .descriptors import map3to1


def bytes_to_str(byte_list): 
    """Converts a list of bytes to a string
    Args: 
        byte_list(list[b])
    Returns: 
        (str)
    """
    return ''.join([str(b, encoding='utf-8') for b in byte_list])
 

def ls(path, names_only=False): 
    """ list the contenst of a dir
    Args: 
        path (str|Path): the dir
        with_dir (bool): if True, return the name of the file only. 
    Returns: 
        files(list[Path]): list of the paths or name of files
    
    """
    path = Path(path)
    if names_only: 
        files = [f.name for f in path.iterdir()]
    else:
        files = [f for f in path.iterdir()]

    return files

def renumber_pdb(input_pdb, output_pdb=None):
    """Renumbers residues for pdb file
    Args: 
        input_pdb(str|Path): path to input pdb
        output_pdb(str|Path): path to pdb for output
    Returns: 
        None
    """

    # parse the pdb and update numbers for each chain
    parser = PDBParser()
    struct= parser.get_structure('pdb', str(input_pdb))
    for model in struct: 
        for chain in model:
            num = 1
            for residue in chain:
                residue.id = (' ', num, ' ')
                num+=1

    # save the pdb
    if not output_pdb: 
        output_pdb = Path(input_pdb).with_suffix('.renumbered.pdb')
    pdb_io = PDBIO()
    pdb_io.set_structure(struct)
    pdb_io.save(str(output_pdb))


def extract_sequence_from_pdb(pdb):
    """Extracts the sequences from pdb file or pdb Structure as a 1 letter code
    Args: 
        pdb(str|Path|structure): path to pdb file
    Returns: 
        sequences(dict): dict of sequences of individual chains. 
    """
    
    sequences = {}
    if isinstance(pdb, str) or isinstance(pdb, Path):
        parser=  PDBParser()
        struct= parser.get_structure('pdb', str(pdb))
    else:
        struct = pdb
    
    for model in struct: 
        for chain in model:
            id = chain.id
            sequences.setdefault(id,[])
            for residue in chain:
                sequences[id].append(residue.get_resname())
    for seq in sequences: 
        sequences[seq] = ''.join(map3to1(sequences[seq]))
    return sequences
    
def clean_logs():
    """cleans logs in running directory"""
    cwd = Path().cwd() 
    _ = [file.unlink() for file in ls(cwd, False) if file.name.startswith('log') & (file.name.endswith('.err') | file.name.endswith('.out'))]
    return None


def determine_chain_type(seqs, scheme='kabat'): 
    """Given a dict of sequences from antibody Fab region, determine which is sequence 
    is heavy or light chain respectively, and returns light and then heavy chain 

    Args:
        seqs(dict): dict with key as Chain Name and value as seq. 
        scheme(str): the scheme for numbering and identifying heavy/light chain. 
    Returns: 
        tuple(str, str): returns the light chain and heavy chain seqs 
    """

    chains = {}
    
    for seq in seqs.values(): 
        try:
            chain = Chain(seq,scheme )
            if chain.is_heavy_chain(): 
                chains.setdefault('H', seq)
            elif chain.is_light_chain(): 
                chains.setdefault('L', seq)
            else: 
                pass
        except ChainParseError:
            pass
    
    return chains['L'], chains['H']


#### Viz
def plot_correlogram(d, cmap='RdBu',title = None,  method = 'pearson', figsize = (11,9), compute_corr=True, 
                     annot=False, annot_kws=None, vmin=None, vmax=None, linewidths=0.25, tick_label_size=None):
    """Plots a correlogram for a dataframe
    Args: 
        d(pd.DataFrame): dataframe
        cmap(str): color map
        title(str): title for plot
        method(str): method for correlation
        figsize(tuple): size of figure
        compute_corr(bool): if True, compute correlation matrix
        annot(bool): if True, annotate the cells
        annot_kws(dict): dict to control annotation. 
        vmin(float): min of color bar
        vmax(float): max of color bar
        linewidths(float): line widths for grid lines. 
        tick_label_size(float): size for both tick labels. 
    Returns:
        ax(matplotlib.axes): axes for plot
    """

    if compute_corr:
        # Compute the correlation matrix
        corr = d.corr(method = method)
    else: 
        corr = d

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=linewidths, cbar_kws={"shrink": .5}, 
                ax = ax, annot=annot, annot_kws=annot_kws, vmin=vmin, vmax = vmax)
    
    if tick_label_size: 
        ax.tick_params(axis = 'both', which = 'major', labelsize = tick_label_size)
    
    if title: 
        ax.set(title = title)

    return ax 