#!/usr/bin/env python

from developability.mutator import generate_dict_of_mutations, Mutator


def is_none(x):
    return x is None


def test_generate_dict_of_mutations():
    lc_mutations = 'G31Y,G34L'
    hc_mutations = 'G31Y,G34L'
    expected_output = {'L': [['GLY', 31, 'TYR'], ['GLY', 34, 'LEU']],
                       'H': [['GLY', 31, 'TYR'], ['GLY', 34, 'LEU']]}

    assert (generate_dict_of_mutations(lc_mutations, hc_mutations) ==
            expected_output)


def test_Mutator():
    mutator = Mutator()
    assert mutator.light_chain_id == 'L'
    assert mutator.heavy_chain_id == 'H'
    assert mutator.chains == ['L', 'H']
    assert mutator.should_correct_oxt is True
    assert is_none(mutator.lc_length)
    assert is_none(mutator.hc_length)
    assert is_none(mutator.ph)


if __name__ == '__main__':
    test_generate_dict_of_mutations()
    test_Mutator()

