# calculates descriptors from csv file
import pandas as pd
from abnumber import Chain
from Bio.SeqUtils import seq1


def map3to1(residues, custom_map=None):
    """ uses dict to map proteins 3 letter code  to 1 letter code
    Args:
        residues(list): list of 1 code residues
        custom_map(dict): dict of custom mappings
    Returns
        three_code(list)
    """
    if not custom_map:
        custom_map = {'CYX': 'C', 'HIE': 'H'}

    return [seq1(aa, custom_map) for aa in residues]


def chain_to_df(chain):
    """Converts a chain from Abnumber to dataframe for merging"""

    regions = chain.regions

    region_list = []
    residues = []

    for reg in regions:
        aminos = regions[reg]
        for aa in aminos.items():
            region_list.append(reg)
            residues.append(aa[1])

    regions_df = pd.DataFrame(dict(Residue=residues,
                                   Region=region_list,
                                   Residue_number=range(1, len(region_list)+1)
                                   )
                              )
    if chain.chain_type == "H":
        chain_type = "H"
    else:
        chain_type = "L"

    regions_df['Chain'] = chain_type
    return regions_df


def extract_sequence_from_residue_potential_df(residue_potential_df):
    """ Extract the sequence from a residue potential dataframe
    Args:
        residue_potential_df (pd.DataFrame): dataframe with the residue.
         potential.
    Returns:
        seq (str): sequence of the antibody
    """

    seq = ''.join(map3to1(residue_potential_df['Residue_name'].values))
    return seq


def annotate_residues_with_fv_regions(sequences, residue_pot_df,
                                      scheme='kabat'):
    """Args:
        sequences (dict): dictionary of the sequences of the antibody
        residue_pot_df (pd.DataFrame): dataframe with the residue potential

    Returns:
        residue_pot_df (pd.DataFrame): dataframe with the residue potential
        annotated with the FV regions.
    """

    target_sequence = extract_sequence_from_residue_potential_df(
        residue_pot_df)

    # Use Abnumber to identify CDR regions for both light and heavy chain and
    #  merge into a single dataframe.
    light_chain_seq, heavy_chain_seq = sequences['L'], sequences['H']

    light_chain = Chain(light_chain_seq, scheme=scheme)
    heavy_chain = Chain(heavy_chain_seq, scheme=scheme)

    light_regions = chain_to_df(light_chain)
    heavy_regions = chain_to_df(heavy_chain)

    if sequences['H'] + sequences['L'] == target_sequence:
        residue = list(sequences['H'] + sequences['L'])

        fv_chain = ['H']*len(sequences['H']) + ['L']*len(sequences['L'])

        fv_number = list(range(1, len(sequences['H'])+1)) + \
            list(range(1, len(sequences['L'])+1))

        fv_region = list(heavy_regions['Region']) + \
            list(light_regions['Region'])

    elif sequences['L'] + sequences['H'] == target_sequence:
        residue = list(sequences['L'] + sequences['H'])

        fv_chain = ['L']*len(sequences['L']) + ['H']*len(sequences['H'])

        fv_number = list(range(1, len(sequences['L'])+1)) + \
            list(range(1, len(sequences['H'])+1))

        fv_region = list(light_regions['Region']) + \
            list(heavy_regions['Region'])

    else:
        raise ValueError('Could not determine chain order')

    # New values to the df
    residue_pot_df['Residue'] = residue
    residue_pot_df['FV_chain'] = fv_chain
    residue_pot_df['FV_Residue_number'] = fv_number
    residue_pot_df['FV_region'] = fv_region

    return residue_pot_df


def name(func):
    """ returns the name of a func"""
    return func.__name__


def region_potentials(residue_pot_df):
    """ Calculates the potentials for regions
    TODO: refactor this function to make cleaner.
    Args:
        residue_pot_df(pd.DataFrame): annotated DataFrame
        with the residue potentials.
    Returns:
        vals(dict): dict with values
    """

    def calculate_total_potential(region, chain, charge='pos'):
        """ Calculates the potential for a region
        Args:
            region(str|list[str]): the region to calculate
            chain(str|list[str]): the chain to calculate
            charge(str): the charge to calculate
        Returns:
            float: the potential for the region
        """

        if isinstance(region, str):
            region = [region]

        if isinstance(chain, str):
            chain = [chain]

        if charge == 'pos':
            df = residue_pot_df.query('total_pot > 0')
        elif charge == 'neg':
            df = residue_pot_df.query('total_pot < 0')
        elif charge == 'net':
            df = residue_pot_df
        else:
            raise ValueError('charge must be pos, neg or net')
        return df.loc[df.FV_region.isin(region) &
                      df.FV_chain.isin(chain)]['total_pot'].sum()

    cdrs = ['CDR1', 'CDR2', 'CDR3']
    frameworks = ['FR1', 'FR2', 'FR3', 'FR4']
    all_regions = cdrs + frameworks
    chains = ['H', 'L']

    charges = ['pos', 'neg', 'net']

    # set short name for calculate_total_potential
    calc_pot = calculate_total_potential

    vals = {f'{chain}{region}_APBS_{chrg}': calc_pot(region, chain, chrg)
            for chain in chains for region in all_regions
            for chrg in charges}

    vals.update({f'{chain}CDR_APBS_{chrg}': calc_pot(cdrs, chain, chrg) for
                 chain in chains for chrg in charges})

    vals.update({f'{chain}FR_APBS_{chrg}': calc_pot(frameworks, chain, chrg)
                 for chain in chains for chrg in charges})

    vals.update({f'{chain}C_APBS_{chrg}': calc_pot(all_regions, chain, chrg)
                 for chain in chains for chrg in charges})

    vals.update({f'TOTAL_CDR_APBS_{chrg}': calc_pot(cdrs, chains, chrg)
                 for chrg in charges})

    vals.update({f'TOTAL_FR_APBS_{chrg}': calc_pot(frameworks, chains, chrg)
                 for chrg in charges})

    vals.update({f'TOTAL_APBS_{chrg}': calc_pot(all_regions, chains, chrg)
                 for chrg in charges})

    return vals


def calculate_descriptors(residue_pot_df, antibody_name='', features=None):
    """Calculates features from resdiue_potential dataframe
    Args:
        residue_pot_df (pd.DataFrame)
        antibody_name (str): the name of the antibody.
        features(list): list of funcs for getting features.
    Returns
        features_df (pd.DataFrame)
    """
    if not features:
        features = [region_potentials]

    feat_dict = {}

    for func in features:
        feat_dict.update(func(residue_pot_df))

    features_df = pd.DataFrame.from_dict(feat_dict, orient='index').transpose()
    features_df.index = [antibody_name]
    return features_df


def descriptor_pipeline(light_chain_seq, heavy_chain_seq, residue_pot_file,
                        antibody_name='', features=None):
    """ Generates the surface descriptors for an antibody given
        a residue_potential file.
    Args:
        light_chain_seq (str): aa sequence
        heavy_chain_seq (str): aa sequence
        residue_pot_file (str|Path): path to residue_pot_file
        antibody_name(str): the name of the antibody.
    Returns:
        pd.DataFrame
    """
    residue_pot_df = pd.read_csv(residue_pot_file)
    sequences = {'L': light_chain_seq, 'H': heavy_chain_seq}

    annotated_residue_pot_df = annotate_residues_with_fv_regions(
        sequences, residue_pot_df, scheme='kabat')
    annotated_residue_pot_df.to_csv(
        residue_pot_file.with_suffix('.annotated.csv'))
    descriptors = calculate_descriptors(
        annotated_residue_pot_df, antibody_name, features)
    return descriptors


if __name__ == '__main__':
    from pathlib import Path

    lc = """DIELTQSPASLSASVGETVTITCQASENIYSYLAWHQQKQGKSPQLLVYNAKTLAGGVSSRFSGSG
            SGTHFSLKIKSLQPEDFGIYYCQHHYGILPTFGGGTKLEIK"""

    lc = ''.join(lc.split())

    hc = """QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNTRYTHKFKG
            KATLTADKSSSTAYMQLSSLASEDSGVYYCARGEGNYAWFAYWGQGTTVTVSS"""
    hc = ''.join(hc.split())

    input_path = Path('/Users/ChristopherRivera/Projects/developability')
    input_path = input_path / 'developability/data/pdbs/test'
    residue_pot_file = input_path/'residue_potential.csv'
    descriptors = descriptor_pipeline(lc, hc, residue_pot_file, features=None)
