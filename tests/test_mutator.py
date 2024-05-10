#!/usr/bin/env python

import pandas as pd
from pathlib import Path
from Bio.SeqUtils import seq3

from developability.developability import mutate_antibody
from developability.mutator import (generate_dict_of_mutations, Mutator,
                                    parse_mutant_string)

from developability.pdb_tools import (extract_sequence_from_pdb, get_fv_chains)

import warnings
warnings.filterwarnings("ignore")


data_path = Path(__file__).parent / 'data'


def is_none(x):
    return x is None


def test_generate_dict_of_mutations():
    lc_mutations = 'I2Q, V3L'
    hc_mutations = 'V2Q, Q3L, L4V'
    expected_output = {'L': [['ILE', 2, 'GLN'], ['VAL', 3, 'LEU']],
                       'H': [['VAL', 2, 'GLN'], ['GLN', 3, 'LEU'],
                             ['LEU', 4, 'VAL']]
                       }

    assert (generate_dict_of_mutations(lc_mutations, hc_mutations) ==
            expected_output)


def test_Mutator():
    pdb = data_path / 'abciximab_6v4p.pdb'
    mutant_df = pd.DataFrame(dict(VH=['V2Q, Q3L, L4V'], VL=['I2Q, V3L']))
    mutator = Mutator(pdb, mutant_df)
    __, fv_seqs = get_fv_chains(pdb)
    assert mutator.light_chain_id == 'D'
    assert mutator.heavy_chain_id == 'C'
    assert mutator.chains == ['D', 'C']
    assert mutator.should_correct_oxt is True
    assert mutator.lc_length == len(fv_seqs['light'])
    assert mutator.hc_length == len(fv_seqs['heavy'])
    assert mutator.pH == 7


def test_Mutator_preprocess_parent_antibody():
    pdb = data_path / 'abciximab_6v4p.pdb'
    mutant_df = pd.DataFrame(dict(VH=['V2Q, Q3L, L4V', 'V2Q, Q3L, L4K'],
                                  VL=['I2Q, V3L', 'I2Q, V3K']))
    mutator = Mutator(pdb, mutant_df)
    mutator.preprocess_parent_antibody()
    fv_only_pdb = mutator.fv_only_pdb
    assert fv_only_pdb.exists()

    # maker suere the seqs are as desired
    fv_only_seqs = extract_sequence_from_pdb(fv_only_pdb)
    _, fv_seqs = get_fv_chains(pdb)
    assert len(fv_only_seqs) == 2
    assert len(fv_only_seqs['H']) <= len(fv_seqs['heavy'])
    assert len(fv_only_seqs['L']) <= len(fv_seqs['light'])

    # clean the fv_only_pdb
    fv_only_pdb.unlink()


def test_Mutator_generate_mutants():
    pdb = data_path / 'abciximab_6v4p.pdb'
    mutant_df = pd.DataFrame(dict(VH=['V2Q, Q3L, L4V',  '', 'V2Q, Q3L, L4V'],
                                  VL=['I2Q, V3L', 'I2Q, V3L', '']))

    mutator = Mutator(pdb, mutant_df)
    mutator.preprocess_parent_antibody()
    file_names = mutator.generate_mutants()
    print(file_names)

    file_paths = [mutator.output_path / name for name in file_names]
    # print(file_paths)

    for i, f in enumerate(file_paths):
        assert f.exists()
        seqs = extract_sequence_from_pdb(f)
        assert len(seqs) == 2
        assert len(seqs['H']) == mutator.hc_length
        assert len(seqs['L']) == mutator.lc_length

        lc_mutations = parse_mutant_string(mutant_df['VL'].iloc[i])
        hc_mutations = parse_mutant_string(mutant_df['VH'].iloc[i])

        for m in lc_mutations:
            assert seq3(seqs['L'][m[1]-1]).upper() == m[2]

        for m in hc_mutations:
            assert seq3(seqs['H'][m[1]-1]).upper() == m[2]

    # clean the generated pdb and fv_only_pdb
    mutator.fv_only_pdb.unlink()
    for f in file_paths:
        f.unlink()


def test_mutate_antibody_cli():
    pdb = data_path / 'abciximab_6v4p.pdb'
    mutations = data_path / 'mutations.csv'

    mutant_df = pd.DataFrame(dict(VH=['V2Q, Q3L, L4V', 'V2Q, Q3L, L4K'],
                                  VL=['I2Q, V3L', 'I2Q, V3K']))

    mutant_df.to_csv(mutations, index=False)
    file_paths = mutate_antibody(pdb, mutations)
    for f in file_paths:
        assert f.exists()
        f.unlink()


if __name__ == '__main__':
    # print(test_generate_dict_of_mutations())
    test_Mutator()
    test_Mutator_preprocess_parent_antibody()
    test_Mutator_generate_mutants()
    print('All tests passed!')
