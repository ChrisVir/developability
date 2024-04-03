import pandas as pd
from pathlib import Path


class Mutator:

    def __init__(self, parent_pdb, mutation_df, light_chain_mutations='VL', heavy_chain_mutations='VH', filename=None,output_path=None,
                light_chain_id='L', 
                heavy_chain_id='H',
                extract_FV_chains=True, 
                ): 
        """Class to handle mutations with PDBFixer
        Args: 
            parent_pdb(str|Path): location for pdb
            mutation_df(str|Path|pd.DataFrame): 
            mutation_col(str): name of column with mutations, default to mutants
            light_chain_mutations(str): column with parental light, default to VL
            heavy_chain_mutations(str): column with mutations, default to VH
            name(str|None): column with output names, if None, constructed from parent_pdb name. 
            output_path(str|path|None): path for output of mutated PDBs. if None, creates new subdirectory in location of parent pdb. 
            light_chain_id(str): name of the light chain
            heavy_chain_id(str): name of the heavy chain
            extract_FV_chains(bool): if True, create a PDB with FV only. (This is what is required for calculating downstream stuff )
        """

        self.parent_pdb = parent_pdb
        self.df = mutation_df
        self.light_chain_mutations = light_chain_mutations
        self.heavy_chain_mutations = heavy_chain_mutations

        if filename is None: 
            pass
        if output_path is None: 
            pass

        self.light_chain_id = 'L'
        self.heavy_chain_id = 'H'
        


        









