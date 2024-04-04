#!/usr/bin/env python

from developability.mutator import generate_dict_of_mutations


def test_generate_dict_of_mutations():
    lc_mutations = 'G31Y,G34L'
    hc_mutations = 'G31Y,G34L'
    expected_output = {'L': [['GLY', 31, 'TYR'], ['GLY', 34, 'LEU']],
                       'H': [['GLY', 31, 'TYR'], ['GLY', 34, 'LEU']]}

    assert (generate_dict_of_mutations(lc_mutations, hc_mutations) ==
            expected_output)


if __name__ == '__main__':
    test_generate_dict_of_mutations()
