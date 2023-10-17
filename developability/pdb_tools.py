# functions for manipulating /cleaning PDBs. 

from pdbfixer import PDBFixer
from openmm.app import PDBFile

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


def corrext_oxt(fixer): 
    """ converts interal OXT to O for chains
    Args: 
        fixer(pdbfixer.PDBFixer)
    Returns 
        fixer
    """
    num_oxt_fixed = 0
    for chain in fixer.topology.chains(): 
        n = len(list(chain.residues()))
        for atom in chain.atoms(): 
            if atom.residue.id!=str(n) and atom.residue.id!=str(0) and atom.name=='OXT': 
                atom.name = 'O'
                num_oxt_fixed+=1

    print(f'The number of OXT atoms is {count_oxt(fixer)}.')
    print(f'The number of OXT->O is {num_oxt_fixed}.')