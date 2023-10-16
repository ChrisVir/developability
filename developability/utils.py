# utils 
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from abnumber import Chain
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
    """Extracts the sequences from pdb file as a 1 letter code
    Args: 
        pdb(str|Path): path to pdb file
    Returns: 
        sequences(dict): dict of sequences of individual chains. 
    """
    parser=  PDBParser()

    sequences = {}
    struct= parser.get_structure('pdb', str(pdb))
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
        chain = Chain(seq,scheme )
        if chain.is_heavy_chain(): 
            chains.setdefault('H', seq)
        elif chain.is_light_chain(): 
            chains.setdefault('L', seq)
        else: 
            pass
    
    return chains['L'], chains['H']