# calculates descriptors from csv file
import pandas as pd
from abnumber import Chain
from Bio.Data.IUPACData import protein_letters_3to1


def map3to1(residues):
    """ uses dict to map proteins 3 letter code  to 1 letter code
    Args: 
        residues(list): list of 1 code residues
    Returns
        three_code(list)
    """

    #TODO: Handle noncanonical residues

    return [protein_letters_3to1.get(aa.capitalize(), '') for aa in residues]


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
                            
    regions_df= pd.DataFrame(dict(Residue=residues, 
                             Region = region_list, 
                             Residue_number=range(1, len(region_list)+1)
                            )
                       )
    if chain.chain_type=="H": 
        
        chain_type = "H" 
    else: 
        chain_type ="L"
    
    regions_df['Chain']=chain_type
    return regions_df


def map_region_to_ab_seq(light_chain_seq,heavy_chain_seq, residue_pot_df, scheme='kabat'):
    """Uses Abnumber to annoate residues of sequence and maps to residue potential dataframe
    Args:
        light_chain_seq(str): the light chain sequence
        heavy_chain_seq(str): the heavy chain sequence
        residue_pot_df(pd.DataFrame): the data frame with summed potential per residue. 
        scheme(str): the CDR annotation scheme to use
    Returns: 
        pd.DataFrame
    """

    # Use Abnumber to identify CDR regions for both light and heavy chain and merge into a single dataframe 
    light_chain = Chain(light_chain_seq, scheme=scheme)
    heavy_chain = Chain(heavy_chain_seq, scheme=scheme)
    
    light_regions = chain_to_df(light_chain)
    heavy_regions = chain_to_df(heavy_chain)
    
    both_regions = pd.concat([ light_regions,heavy_regions])
    both_regions['Residue_number']=range(1, len(both_regions)+1)

    # combine with potential data
    one_code = map3to1(residue_pot_df['Residue_name'])  # annotate 3 code with 1 code for aa
    residue_pot_df.insert(2,'Residue' , one_code)
    residue_pot_df = residue_pot_df.merge(both_regions, 
                                          how = 'left', 
                                          on=['Residue', 'Residue_number']
                                          )
    return residue_pot_df

def name(func): 
    """ returns the name of a func"""
    return func.__name__


def region_potentials(residue_pot_df): 
    """ Calculates the potentials for regions
    TODO: refactor this function to make cleaner. 
    Args: 
        residue_pot_df(pd.DataFrame): The data frame with potentials
    Returns: 
        vals(dict): dict with values

    """
    
    cdrs = ['CDR1', 'CDR2', 'CDR3', 'FR1', 'FR2', 'FR3', 'FR4', 'FR5']
    chains = ['H', 'L']

    # calculate for cdrs
    pos_df =  residue_pot_df.loc[residue_pot_df['total_pot']>=0]
    neg_df =  residue_pot_df.loc[residue_pot_df['total_pot']<0]
    
    vals = {f'{chain}{cdr}_APBS_pos':pos_df.query("Region==@cdr & Chain==@chain")['total_pot'].sum() 
            for cdr in cdrs for chain in chains}
    
    vals.update({f'{chain}{cdr}_APBS_neg':neg_df.query("Region==@cdr & Chain==@chain")['total_pot'].sum() 
            for cdr in cdrs for chain in chains}
               )
    
    #calculate net potential per cdr
    vals.update( {f'{chain}{cdr}_APBS_net':residue_pot_df.query("Region==@cdr & Chain==@chain")['total_pot'].sum() 
            for cdr in cdrs for chain in chains})
    
    # total positive and negative charge
    vals['CDR_APBS_pos']= pos_df.loc[pos_df.Region.isin(cdrs)]['total_pot'].sum()
    vals['CDR_APBS_neg']= neg_df.loc[neg_df.Region.isin(cdrs)]['total_pot'].sum()
    vals['CDR_APBS_net']= residue_pot_df.loc[residue_pot_df.Region.isin(cdrs)]['total_pot'].sum()
    
    return vals


def calculate_descriptors(residue_pot_df, antibody_name='', features = None): 
    """Calculates features from resdiue_potential dataframe
    Args: 
        residue_pot_df (pd.DataFrame)
        antibody_name (str): the name of the antibody
        features(list): list of funcs for getting features. 
    Returns
        features_df (pd.DataFrame)
    """
    if not features: 
        features= [region_potentials]

    feat_dict = {}
    
    for func in features:
         feat_dict.update(func(residue_pot_df))
    
    features_df = pd.DataFrame.from_dict(feat_dict, orient='index').transpose()
    features_df.index = [antibody_name]
    return features_df

def descriptor_pipeline(light_chain_seq, heavy_chain_seq, residue_pot_file, antibody_name='',features=None):
    """ generates the surface descriptors for an antibody given a residue_potential file
    Args:
        light_chain_seq (str): aa sequence
        heavy_chain_seq (str): aa sequence
        residue_pot_file (str|Path): path to residue_pot_file
        antibody_name(str): the name of the antibody. 
    Returns: 
        pd.DataFrame
    """
    residue_pot_df = pd.read_csv(residue_pot_file)
    annotated_residue_pot_df= map_region_to_ab_seq(light_chain_seq,heavy_chain_seq, residue_pot_df)
    descriptors = calculate_descriptors(annotated_residue_pot_df, antibody_name, features) 
    return descriptors

if __name__ == '__main__':
    from pathlib import Path


    lc = """DIELTQSPASLSASVGETVTITCQASENIYSYLAWHQQKQGKSPQLLVYNAKTLAGGVSSRFSGSGSGTHFSLKIKSLQPEDFGIYYCQHHYGILPTFGGGTKLEIK"""
    hc = """QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNTRYTHKFKGKATLTADKSSSTAYMQLSSLASEDSGVYYCARGEGNYAWFAYWGQGTTVTVSS"""

    input_path = Path('/Users/ChristopherRivera/Projects/developability/developability/data/pdbs/test')
    residue_pot_file = input_path/'residue_potential.csv'
    descriptors = descriptor_pipeline(lc, hc, residue_pot_file, features=None)

