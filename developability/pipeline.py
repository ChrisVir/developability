# pipeline for generating features

from developability.energy_minimization import EnergyMinimizer
from developability.electrostatics import APBS
from developability.surface import SurfacePotential
from developability.descriptors import descriptor_pipeline
from developability.utils import extract_sequence_from_pdb, clean_logs


def run_processing_pipeline(input_pdb, nanoshaper_options=None):
    """Runs pipeline on individual antibody to extract features. 
    Args:
        input_pdb (_type_): _description_
        nanoshaper_options(dict)
    Returns: 
        descriptors (pd.DataFrame): one row dataframe. 
    """
    
    # minimize the energy 
    minimizer=EnergyMinimizer(input_pdb)
    minimized_pdb = minimizer.minimize_energy()

    # compute electrostatics
    apbs = APBS(minimized_pdb )
    pqr_file, dx_file = apbs.calculate_potential()

    # compute the surface
    surface = SurfacePotential(pqr_file, dx_file, nanoshaper_options=None)
    residue_pot_file = surface.calculate_surface_potential()

    # get the sequence from the input pdb
    sequences = extract_sequence_from_pdb(input_pdb)
    light_chain_seq = sequences['L']
    heavy_chain_seq = sequences['H']

    # Calculate the descriptors
    antibody_name = input_pdb.name.split('.')[0]
    descriptors = descriptor_pipeline(light_chain_seq, heavy_chain_seq,residue_pot_file, antibody_name)
    
    # clean out logs in execution location
    clean_logs()
    
    return descriptors
