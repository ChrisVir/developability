# functions for manipulating /cleaning PDBs. 
from pathlib import Path
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from biopandas.pdb import PandasPdb
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO, Select
from Bio.SeqUtils import seq3
from abnumber import Chain, ChainParseError

from .utils import extract_sequence_from_pdb, determine_chain_type

def remove_chains(fixer, chains_to_keep): 
    """Uses fixer.remove_chains to remove chains not present in chain_to_keep. Keeps the first occurence of each chain. 

    Args:
        fixer (pdbfixer.pdbfixer.PDBFixer): fixer object for fixing pdbs. 
        chains_to_keep (list[str]): list of chain ids to keep, all others removed
    Returns: 
        None
    """
    seen = set()
    indices_to_remove = []
    for i, chain in enumerate(fixer.topology.chains()):
        if chain.id not in seen and chain.id in chains_to_keep: 
            seen.add(chain.id)
        else:
            indices_to_remove.append(i) 

    fixer.removeChains(indices_to_remove)   
    return fixer


def fix_antibody(antibody_file, chains_to_keep=[], output_file_name=None, output_path=None, keep_ids=False, 
                 fix_internal_residues_only=True): 
    """Uses PDB fixer to fix antibody
    Args:
        antibody_file(str|Path): file path
        chains_to_keep(list[str]): list of Fab chains to keep
        output_file_name (str): name of file. Defaults to None.
        output_path (str|Path): dir for output. Defaults to None.
        keep_ids (bool, optional): If true, keep the ids and numbering from input pdb. Defaults to False.
        fix_internal_residues_only (bool, optional): If true, fix only internal residues. Defaults to True.

    Returns:
        PDBFixer: the fixer object. 
    """
    fixer = PDBFixer(str(antibody_file))

    if chains_to_keep: 
        remove_chains(fixer, chains_to_keep)
    fixer.findMissingResidues()

    if fix_internal_residues_only: 
        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        keys_to_remove = []
        for key in keys:
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                keys_to_remove.append(key)

        for key in keys_to_remove: 
            del fixer.missingResidues[key]

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # save
    if not output_file_name:
        output_file_name = antibody_file.name

    if not output_path: 
        output_path = antibody_file.parent

    output = str(output_path /output_file_name)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(output, 'w'), keepIds = keep_ids)
    return fixer


def count_oxt( fixer): 
    """counts the number of OXT atoms in pdb via fixer

    Args:
        fixer (pdbfixer.PDBFixer): fixer object

    Returns:
        int: the number of OXT atoms in all chains. 
    """
    return sum(1 for atom in fixer.topology.atoms() if atom.name == 'OXT')


def correct_oxt(pdb, output_filename=None): 
    """ converts interal OXT to O for chains
    Args: 
        pdb (PDBFixer|str|path): path or fixer object
        output_filename(str|Path, optional): if not None, path to write out. Defaults to None. 
    Returns 
        None
    """
    if isinstance(pdb, (str, Path)):
        fixer = PDBFixer(str(pdb))
    elif isinstance(pdb, PDBFixer): 
        fixer = pdb

    num_oxt_fixed = 0
    for chain in fixer.topology.chains(): 
        n = len(list(chain.residues()))
        for atom in chain.atoms(): 
            if atom.residue.id!=str(n) and atom.residue.id!=str(0) and atom.name=='OXT': 
                atom.name = 'O'
                num_oxt_fixed+=1

    print(f'The number of OXT atoms is {count_oxt(fixer)}.')
    print(f'The number of OXT->O is {num_oxt_fixed}.')

    if output_filename: 
        write_fixer_pdb(fixer, output_filename)
    

def write_fixer_pdb(fixer, output_filename, keep_ids=True): 
    """Writes out the pdb fixer.

    Args:
        fixer (pdbfixer.PDBFixer): fixer object
        output_filename (str|Path):  path for output file 
        keep_ids(bool, optional): If True keep original ids. 
    Returns: 
        None
    """
    PDBFile.writeFile(fixer.topology, fixer.positions, open(str(output_filename), 'w'), 
                        keepIds = keep_ids)
    

def download_pdb(pdb_id, protein_name='',output_path=None, atoms_only=False): 
    """Uses BioPandas to download pdb from Protein DataBank

    Args:
        pdb_id (str): The PDB
        protein_name (str, optional): Name of the protein. Defaults to ''.
        output_path (str|Path, optional): path for directory to save. Defaults to None.
        atoms_only (bool, optional): If true, write out the atoms only. . Defaults to False.
    """

    pdb= PandasPdb()
    p = pdb.fetch_pdb(pdb_id)

    if not output_path: 
        output_path = Path().cwd()
    
    if atoms_only:
        p.to_pdb(output_path/f'{protein_name}_{pdb_id}.pdb', records =['ATOM'])
    else:
        p.to_pdb(output_path/f'{protein_name}_{pdb_id}.pdb')


class ChainSelect(Select): 
    def __init__(self, chain_ids):
        """Select for getting specific chains

        Args:
            chain_ids (list[str]): list of chain ids
        """
        self.chain_ids = chain_ids

    def accept_chain(self, chain): 
        if chain.get_id() in self.chain_ids: 
            return True
        else: 
            return False


def save_pdb_with_select_chains(input_pdb, chains, output_path=None): 
    """Saves pdbs as new file with select chains only
    Args:
        input_pdb (Path|str): Path to the input_pdb
        chains (list[str]): list of names for chain
        output_path (Path|str, optional): Output path. Defaults to None.
    Returns:
        None
    """
    input_pdb = Path(input_pdb)
    name = input_pdb.name.split('.')[0]
    output_name = f'{name}_{"".join(chains)}.pdb'

    if not output_path: 
        output_path = input_pdb.parent
        
    parser = PDBParser()
    struct= parser.get_structure(name, str(input_pdb))
    
    io = PDBIO()
    io.set_structure(struct)
    fname = output_path/output_name
    io.save(str(output_path/output_name),ChainSelect(chains))
    return fname


def extract_fv_from_pdb(pdb, output_pdb=None, scheme='kabat'): 
    """extracts the fv region from pdb and saves pdb. 
    The PDB has VH then VL. 
    Args:
        pdb (str|path): path to pdb file with ab
        output_pdb (str|path): path to output
    Returns: 
        Path: to new object
    """
    
    seqs = extract_sequence_from_pdb(pdb)
    
    # dicts to hold info. 
    fv_sequences = {}
    fv_chains ={}
    
    for name, seq in seqs.items(): 
        try:
            chain = Chain(seq, scheme = scheme)
            fv_sequences[name] = chain.seq
            if chain.is_heavy_chain(): 
                fv_chains[name]="H"
            else: 
                fv_chains[name]="L"
        except ChainParseError:
            pass
    
    pdb_name = Path(pdb).name.split('.')[0]
    parser = PDBParser()
    struct= parser.get_structure(pdb_name,pdb)

    # now get the regions of interest for each chain. 
    for chain  in struct.get_chains(): 
        chain_id = chain.id

        new_child_dict = {}
        new_child_list = []
        
        # find the locations of the fv region in the PDB chain object
        fv = fv_sequences[chain_id]
        seq = seqs[chain_id]
        start = seq.find(fv) 
        end = start+ len(fv)
        
        # iterate through the PDB chain object and add the residues of interest.
        residue_num = 1
        num = 0
        for residue_id, residue in chain.child_dict.items(): 
            
            if (num>= start) & (num <end):
                new_id = (residue_id[0], residue_num, residue_id[2] )
                new_residue = residue.copy()
                new_residue.id = new_id
                
                new_child_dict[new_id] = new_residue
                new_child_list.append(new_residue)
                residue_num+=1
            
            num+=1

        # update the chains
        chain.child_dict = new_child_dict
        chain.child_list = new_child_list
        chain.id=fv_chains[chain_id]

    #update the order of chains.
    model = struct.child_dict[0]
    model.child_list = [model.child_dict['H'], model.child_dict['L']]
    
    # save the pdb
    if not output_pdb: 
        output_pdb = Path(pdb).with_suffix('.fv_only.pdb')
    pdb_io = PDBIO()
    pdb_io.set_structure(struct)
    pdb_io.save(str(output_pdb))

    return output_pdb


########################################################################################################################################################################
# Mutate proteins
########################################################################################################################################################################


def mutate_protein(pdb, mutations, output_path=None, output_filename=None, keep_ids=True, transform_mutants=True, pH=7.0):
    """Uses PDBFixer to mutate antibody, save the file

    Args:
        pdb (str|path): path to pdb file
        mutations (dict): dict with keys for chain and values of list of mutations
        output_filename (str, optional): name of file. Defaults to None.  
        output_path (str|Path, optional):output path. Defaults to None.
        keep_ids (bool, optional): whether to keep original chain ids and numbering. Defaults to True.
        transform_mutants (bool, optional): If True, transform 1 letter code to 3 letter code. Defaults to True.
        pH (float, optional): pH for adding hydrogens. Defaults to 7.0.

    Returns:
        _type_: _description_
    """

    fixer = PDBFixer(str(pdb))
    for chain, muts in mutations.items(): 
        if transform_mutants:
            fixer.applyMutations(transform_mutant_tuples(muts), chain)
        else:
            fixer.applyMutations(muts, chain)
    
    # this add the extra atoms to the topology. 
    fixer.findMissingResidues()     
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()        
    fixer.addMissingHydrogens(pH)

    if not output_filename: 
        name = Path(pdb).name.split('.')[0]
        for chain, muts in mutations.items(): 
            if muts: 
                muts = [''.join(str(c) for c in m) for m in muts]
                name += '-' + chain + '-' + '-'.join(muts)
        output_filename = name + '.pdb'

    if not output_path:
        output_path = Path(pdb).parent
    
    output = output_path /output_filename
    PDBFile.writeFile(fixer.topology, fixer.positions, open(str(output), 'w'), keepIds = keep_ids)
    return output


def transform_mutant_tuple(mutant):
    """Converts a tuple of 1 letter aa pos aa tuple into three letter aa. 
    Args: 
        mutant(tuple)
    Returns: 
        str: 
    """

    return f'{seq3(mutant[0]).upper()}-{int(mutant[1])}-{seq3(mutant[2]).upper()}'

def transform_mutant_tuples(muts): 
    """ Transforms multiple mutants. """
    return [transform_mutant_tuple(mut) for mut in muts]

def generate_mutations_dict(row, lc_length, hc_length):
    mutations = {'L': [mutation for mutation in row['Vl mutations'] if mutation[1]<=lc_length],
                 'H': [mutation for mutation in row['Vh mutations'] if mutation[1]<=hc_length]
                 }
    return mutations

