from Bio.SeqUtils import seq3
from .pdb_tools import (mutate_protein, extract_sequence_from_pdb,
                        fix_antibody, correct_oxt, extract_fv_from_pdb)


def parse_mutant_string(mutant_string):
    """Parses a string of mutant 1 code and returns as three code captilized
      for input to mutate_protein
    Args:
        mutant_string(str)
    Returns:
        (list[tuples])
    """
    mutants = [m.strip() for m in mutant_string.split(',')]
    return [[seq3(m[0]).upper(), int(m[1:-1]), seq3(m[-1]).upper()]
            for m in mutants]


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
                 light_chain_id='L', heavy_chain_id='H',
                 extract_FV_chains=True, should_corrext_oxt=True,
                 ph=7
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
            light_chain_id(str): name of the light chain
            heavy_chain_id(str): name of the heavy chain
            extract_FV_chains(bool): if True, create a PDB with FV only
            ph(float): pH for the mutations

        NOTE: I assmue that the mutations are in a comma delimited string 
        with parent-pos-mutation format. e.g  G25Y. Glycine at 25 to lysine.
        """

        self.parent_pdb = parent_pdb
        self.fv_only_pdb = parent_pdb.parent / f'{parent_pdb.stem}_fv_only.pdb'
        self.mutation_df = mutation_df
        self.light_chain_mutations = light_chain_mutations
        self.heavy_chain_mutations = heavy_chain_mutations

        self.filename = filename

        if output_path is None:
            prefix = self.parent_pdb.name.replace('.pdb', '')
            self.output_path = self.parent_pdb.parent/f'{prefix}_output'

        self.light_chain_id = 'L'
        self.heavy_chain_id = 'H'
        self.chains = [light_chain_id, heavy_chain_id]
        self.should_correct_oxt = should_corrext_oxt
        self.lc_length = None
        self.hc_length = None
        self.ph = ph

    def __preprocess_parent_antibody__(self):
        """Preprocess_parent_antibody. It fixes the pdb, removes oxt
           if needed and extracts the FV region"""

        print('Fixing antibody.')
        fixer = fix_antibody(self.parent_pdb, self.chains, self.fv_only_pdb)

        if self.correct_oxt:
            print('Correct Oxt residues. ')
            correct_oxt(fixer, self.fv_only_pdb)

        print('Removing FV domain from Antibody')
        extract_fv_from_pdb(self.fv_only_pdb, self.fv_only_pdb)
        chains = extract_sequence_from_pdb(self.fv_only_pdb)
        self.lc_length = len(chains['L'])
        self.hc_length = len(chains['H'])

    def generate_mutants(self):
        """ generate all the mutants"""

        _ = [self.mutate_protein(idx) for idx in range(len(self.mutation_df))]

    def generate_mutant(self, idx):
        """generate the mutant protein at the idx"""

        if self.filename is not None:
            filename = self.mutation_df['filename'].iloc[idx]
        else:
            filename = None

        lc_mutations = self.mutation_df[self.light_chain_mutations].iloc[idx]
        hc_mutations = self.mutation_df[self.heavy_chain_mutations].iloc[idx]

        mutations = generate_dict_of_mutations(lc_mutations, hc_mutations,
                                               lc_length=self.lc_length,
                                               hc_length=self.hc_length)

        return mutate_protein(self.parent_pdb, mutations,
                              output_path=self.output_path,
                              filename=filename, ph=self.ph)
