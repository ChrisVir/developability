import pandas as pd
from ImmuneBuilder import ABodyBuilder2
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from tqdm import tqdm


def renumber_pdb(input_pdb, output_pdb=None):
    """Renumbers residues for pdb file
    Args:
        input_pdb(str|Path): path to input pdb
        output_pdb(str|Path): path to pdb for output
    Returns:
        None
    """

    def renumber_chain(chain):
        new_chain = []

        for i, residue in enumerate(chain.get_residues()):
            new_residue = residue.copy()
            res_id = residue.id
            new_residue.id = (res_id[0], i+1, res_id[2])
            new_chain.append(new_residue)

        chain.child_list = new_chain
        chain.child_dict = {res.id: res for res in new_chain}

    # parse the pdb and update numbers for each chain
    parser = PDBParser()
    struct = parser.get_structure('pdb', str(input_pdb))
    for model in struct:
        for chain in model:
            renumber_chain(chain)

    # save the pdb
    if not output_pdb:
        output_pdb = Path(input_pdb).with_suffix('.renumbered.pdb')
    pdb_io = PDBIO()
    pdb_io.set_structure(struct)
    pdb_io.save(str(output_pdb))


def predict_antibody_structures(sequences, output_dir=None):
    """Given a dataframe with sequences, use ABodyBuilder2 model to predict
       sequences.
    Args:
        sequence(pd.DataFrame|str|Path): data frame with columns 'Name', 'VH',
                                        'VL'
        output_dir(path|Str): location to save models.
    Returns:
        None
    """

    predictor = ABodyBuilder2()
    if isinstance(sequences, str) or isinstance(sequences, Path):
        sequences = pd.read_csv(sequences)
    elif isinstance(sequences, pd.DataFrame):
        pass
    else:
        print('Not antibodies')

    sequences = sequences.dropna()
    errors = []

    iterator = zip(sequences['Name'], sequences['VH'], sequences['VL'])
    n = len(sequences)
    for name, hc, lc in tqdm(iterator, total=n):
        sequences = {'H': hc, 'L': lc}
        try:
            antibody = predictor.predict(sequences)

            if output_dir:
                output_file = output_dir/f'{name}.pdb'
                antibody.save(str(output_file))
        except AssertionError:
            errors.append(name)
    return errors
