"""Main module."""
import click
import pandas as pd

from pathlib import Path
from developability.mutator import Mutator
from developability.electrostatics import APBS
from developability.surface import SurfacePotential
from developability.pdb_tools import extract_sequence_from_pdb
from developability.descriptors import descriptor_pipeline


@click.command()
@click.argument('pdb')
@click.argument('mutations')
@click.option('--input_dir', default=None, help='directory with input files')
@click.option('--output_dir', default=None, help='directory for output files')
def mutate_antibody(pdb, mutations, input_dir, output_dir):
    """Mutate an antibody."""

    if input_dir:
        pdb = Path(input_dir) / pdb
        mutations = Path(input_dir) / mutations
    else:
        pdb = Path(pdb)
        mutations = Path(mutations)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

    mutator = Mutator(pdb, mutations, output_path=output_dir)
    mutator.preprocess_parent_antibody()
    mutator.generate_mutants()


@click.command(help='Computes Electrostatics with APBS')
@click.argument('pdb_dir')
@click.argument('mutations_path')
@click.option('--output_path', default=None, help='directory for output files')
def mutate_multiple_antibodies(pdb_dir, mutations_path, output_path):
    """Mutate multiple antibodies."""
    for pdb in Path(pdb_dir).glob('*.pdb'):
        pdb_id = pdb.stem
        mutations = Path(mutations_path) / f'{pdb_id}_mutations.csv'
        mutate_antibody(pdb, mutations, output_dir=output_path)


@click.command()
@click.argument('input_pdb')
@click.option('--output_path', default='output', help='directory for outputs')
def compute_electrostatics(input_pdb, output_path):
    apbs = APBS(input_pdb, output_path)
    return apbs.calculate_potential()


@click.command()
@click.argument('input_pqr')
@click.argument('input_dx')
@click.option('--output_dir', default=None, help='directory for outputs')
def calculate_surface_potential(input_pqr, input_dx, output_dir=None):
    """Calculates a surface mesh using Nanoshaper and calculates potential
    at surface"""
    sp = SurfacePotential(input_pqr, input_dx, output_dir, multivalue_path="")
    return sp.calculate_surface_potential()


@click.command()
@click.argument('residue_potential_file')
@click.argument('antibody_pdb')
@click.option('--name', default=None, help='name for the output')
def calculate_electrostatic_features(residue_potential_file, antibody_pdb,
                                     name=None):
    """Uses residue potential to calculate electrostatic features for model
    Args:
        residue_potential_file(str|path): path to file with residue potential
        antibody_pdb(str|Path): File for antibody (only FV)
        name(str|optional):
    """

    seqs = extract_sequence_from_pdb(antibody_pdb)
    light_chain_seq = seqs['L']
    heavy_chain_seq = seqs['H']

    if not name:
        name = Path(antibody_pdb).stem

    descriptors = descriptor_pipeline(light_chain_seq,
                                      heavy_chain_seq,
                                      residue_potential_file,
                                      antibody_name=name)

    # Need to convert
    df = pd.DataFrame(descriptors)
    df.to_csv('electrostatics_descriptors.csv')


if __name__ == '__main__':
    pass
