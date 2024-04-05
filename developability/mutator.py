import pandas as pd
from pathlib import Path
from Bio.SeqUtils import seq3

from .pdb_tools import (mutate_protein, fix_antibody, get_fv_chains,
                        correct_oxt, extract_fv_from_pdb)


def parse_mutant_string(mutant_string):
    """Parses a string of mutant 1 code and returns as three code captilized
      for input to mutate_protein
    Args:
        mutant_string(str)
    Returns:
        (list[tuples])
    """

    mutants = []
    for m in mutant_string.split(','):
        m = m.strip()
        aa1 = seq3(m[0]).upper()
        pos = int(m[1:-1])
        aa2 = seq3(m[-1]).upper()
        mutants.append([aa1, pos, aa2])
    return mutants


def convert_mutant_tuples_to_strings(mutant_tuples):
    """Converts a list of mutant tuples to a list of strings in AA-pos-AA 
       format
    Args:
        mutant_tuples(list[tuple]): list of tuples with aa1, pos, aa2
    Returns:
        (str)
    """
    return [f'{m[0]}-{m[1]}-{m[2]}' for m in mutant_tuples]


def generate_dict_of_mutations(light_chain_mutations, heavy_chain_mutations,
                               lc_length=120, hc_length=120, lc_id='L',
                               hc_id='H'):
    """ Generates dictionary holding mutations in appropriate format
    Args:
        light_chain_mutations(str): comma delimited string of mutations
                                   in aa-pos-aa format eg. G25Y.
        heavy_chain_mutations(str): comma delimited string of mutations
                                    in aa-pos-aa format eg. G25Y.
        lc_length(int): length of light chain
        hc_length(int): length of heavy chain
        lc_id(str): name of light chain
        hc_id(str): name of heavy chain
    Returns:
        mutations(dict)
    """

    mutations = {lc_id: [m for m in parse_mutant_string(light_chain_mutations)
                         if m[1] <= lc_length],
                 hc_id: [m for m in parse_mutant_string(
                     heavy_chain_mutations) if m[1] <= hc_length]
                 }

    return mutations


class Mutator:

    def __init__(self, parent_pdb, mutation_df, light_chain_mutations='VL',
                 heavy_chain_mutations='VH', filename=None, output_path=None,
                 extract_FV_chains=True, should_corrext_oxt=True,
                 pH=7
                 ):
        """Class to handle mutations with PDBFixer
        Args:
            parent_pdb(str|Path): location for pdb
            mutation_df(str|Path|pd.DataFrame): df with mutations.
            mutation_col(str): column with mutations, default to mutants.
            light_chain_mutations(str): column with lc mutations, default to VL
            heavy_chain_mutations(str): column with hc mutations, default to VH
            name(str|None): column with output names, if None, from parent_pdb.
            output_path(str|Path|None): Path for the mutants. If None, created.
            extract_FV_chains(bool): if True, create a PDB with FV only
            ph(float): pH for the mutations

        NOTE: I assmue that the mutations are in a comma delimited string 
        with parent-pos-mutation format. e.g  G25Y. Glycine at 25 to lysine.
        """

        self.parent_pdb = parent_pdb
        self.fv_only_pdb = parent_pdb.parent / f'{parent_pdb.stem}_fv_only.pdb'

        self.fv_names, self.fv_seqs = get_fv_chains(parent_pdb)
        self.lc_length = len(self.fv_seqs['light'])
        self.hc_length = len(self.fv_seqs['heavy'])
        self.light_chain_id = self.fv_names['light']
        self.heavy_chain_id = self.fv_names['heavy']
        self.chains = [self.light_chain_id, self.heavy_chain_id]

        if isinstance(mutation_df, str) or isinstance(mutation_df, Path):
            mutation_df = pd.read_csv(mutation_df)

        self.mutation_df = mutation_df
        self.light_chain_mutations = light_chain_mutations
        self.heavy_chain_mutations = heavy_chain_mutations

        self.filename = filename

        if output_path is None:
            prefix = self.parent_pdb.name.replace('.pdb', '')
            self.output_path = self.parent_pdb.parent/f'{prefix}_output'

        if not self.output_path.exists():
            self.output_path.mkdir()

        self.should_correct_oxt = should_corrext_oxt
        self.extract_FV_chains = extract_FV_chains
        self.pH = pH

    def preprocess_parent_antibody(self):
        """Preprocess_parent_antibody. It fixes the pdb, removes oxt
           if needed and extracts the FV region"""

        print('Fixing antibody.')
        fixer = fix_antibody(self.parent_pdb, chains_to_keep=self.chains,
                             output_file_name=self.fv_only_pdb, save_pdb=False)

        if self.should_correct_oxt:
            print('Correct Oxt residues. ')
            correct_oxt(fixer, self.fv_only_pdb)

        print('Removing FV domain from Antibody')
        if self.extract_FV_chains:
            extract_fv_from_pdb(self.fv_only_pdb, self.fv_only_pdb)

    def generate_mutants(self):
        """ generate all the mutants"""
        print('Generating mutants')
        n_mutants = len(self.mutation_df)
        filenames = [self.generate_mutant(idx) for idx in range(n_mutants)]
        return filenames

    def generate_mutant(self, idx):
        """generate the mutant protein at the idx"""

        lc_mutations = self.mutation_df[self.light_chain_mutations].iloc[idx]
        hc_mutations = self.mutation_df[self.heavy_chain_mutations].iloc[idx]

        if self.filename:
            filename = self.mutation_df['filename'].iloc[idx]
        else:
            stem = self.fv_only_pdb.stem
            lc_mutations_str = lc_mutations.replace(', ', '-')
            hc_mutations_str = hc_mutations.replace(', ', '-')
            filename = f'{stem}_L_{lc_mutations_str}_H_{hc_mutations_str}.pdb'

        mutations = generate_dict_of_mutations(lc_mutations, hc_mutations,
                                               lc_length=self.lc_length,
                                               hc_length=self.hc_length)

        mutations = {k: convert_mutant_tuples_to_strings(v) for k, v in
                     mutations.items()}

        mutate_protein(self.fv_only_pdb, mutations,
                       output_path=self.output_path,
                       output_filename=filename,
                       pH=self.pH, transform_mutants=False)

        return filename

    def __repr__(self):
        return f'Mutator for {self.parent_pdb}'
