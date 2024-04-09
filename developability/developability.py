"""Main module."""
import click
from pathlib import Path
from developability.mutator import Mutator
from developability.electrostatics import APBS


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


@click.command()
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
@click.optionan('--output_path', default=None, help='directory for outputs')
def compute_electrostatics(input_pdb, output_path):
    apbs = APBS(input_pdb, output_path)
    return apbs.calculate_potential()


if __name__ == '__main__':
    pass
