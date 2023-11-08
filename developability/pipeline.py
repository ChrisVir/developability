# pipeline for generating features

from developability.energy_minimization import EnergyMinimizer
from developability.electrostatics import APBS
from developability.surface import SurfacePotential
from developability.descriptors import descriptor_pipeline
from developability.utils import extract_sequence_from_pdb, clean_logs, determine_chain_type


def run_processing_pipeline(input_pdb, output_path=None, nanoshaper_options=None,clean=True):
    """Runs pipeline on individual antibody to extract features. 
    Args:
        input_pdb (path|str): path to input_pdb
        output_path(path|str): path to output
        nanoshaper_options(dict): options for nanoshaper. 
        clean(bool): whether to clean out logs in execution location.
    Returns: 
        descriptors (pd.DataFrame): one row dataframe. 
    """
    
    # minimize the energy 
    minimizer=EnergyMinimizer(input_pdb, output_path = output_path)
    minimized_pdb = minimizer.minimize_energy()

    # compute electrostatics
    apbs = APBS(minimized_pdb )
    pqr_file, dx_file = apbs.calculate_potential()

    # compute the surface
    surface = SurfacePotential(pqr_file, dx_file, nanoshaper_options=nanoshaper_options)
    residue_pot_file = surface.calculate_surface_potential()

    # get the sequence from the input pdb
    sequences = extract_sequence_from_pdb(input_pdb)
    light_chain_seq , heavy_chain_seq = determine_chain_type(sequences)

    # Calculate the descriptors
    antibody_name = input_pdb.name.split('.')[0]
    descriptors = descriptor_pipeline(light_chain_seq, heavy_chain_seq,residue_pot_file, antibody_name)
    
    # clean out logs in execution location
    if clean:
        clean_logs()
    
    return descriptors




