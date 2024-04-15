from biobb_amber.pdb4amber.pdb4amber_run import pdb4amber_run
from biobb_amber.leap.leap_gen_top import leap_gen_top
from biobb_amber.sander.sander_mdrun import sander_mdrun
from biobb_amber.process.process_minout import process_minout
from biobb_amber.ambpdb.amber_to_pdb import amber_to_pdb
import pandas as pd

from pathlib import Path


def run_pdb4amber(input_pdb_path, output_pdb_path, properties):
    """ runs pdb4amber to prep pdb to run in amber using biobb_amber
    Args:
        input_pdb_path(str|path): path to the input file
        output_pdb_path(str|path|None): path to output file
        properties(dict|None): configuration for running pdb4amber
    Returns: 
        None
    """

    if not properties:
        properties = {'remove_hydrogens': True}

    pdb4amber_run(input_pdb_path=input_pdb_path,
                  output_pdb_path=output_pdb_path,
                  properties=properties
                  )


def run_leap_gen_top(input_pdb_path, output_pdb_path,
                     output_top_path, output_crd_path,
                     properties=None):
    """ runs amber tools Leap using biobb_amber. The LEAP program does....
    Args: 
        input_pdb_path(str|path): The input pdb path
        output_pdb_path(str|path): The output pdb path
        output_top_path(str|path): Path to the output Amber topology file
        output_crd_path(str|path): Path to the output Amber coordinate file
    Returns: 
        None
    """
    if not properties:
        properties = {"forcefield": ["protein.ff19SB"]}

    leap_gen_top(input_pdb_path=str(input_pdb_path),
                 output_pdb_path=output_pdb_path,
                 output_top_path=output_top_path,
                 output_crd_path=output_crd_path,
                 properties=properties)


def extract_energy_path(sander_log, output_path):
    """ Uses biobb_amber minout to extract energy from sander output. 
    Args: 
        sander_log(str): Path to sander log file. 
    Returns: 
        None
    """
    process_minout(input_log_path=sander_log,
                   output_dat_path=output_path,
                   properties={"terms": ['ENERGY']}
                   )


def plot_energy_path(dat_path, max_enery=1000, figsize=(4, 3)):
    """"plots the energy path using matplotlib"""
    file = Path(dat_path)
    df = pd.read_csv(file, delim_whitespace=True,
                     header=None,
                     names=['Iteration', 'Energy']
                     )
    df = df.query('Energy<=@max_energy')
    ax = df.plot(x="Iteration", figsize=(4, 3))
    # TODO add savepath


def run_sander(input_top, input_crd, input_ref, output_traj, output_rst, output_log,
               properties=None):
    """Executes the Sander molecular dynamics algorithm via biobb_amber for energy minmization. 
    Args: 
        input_top(path|str): Amber topology file from Leap
        input_crd(path|str): Amber coordinate file from Leap
        input_ref(path|str): Amber referecnce file (TODO look up)
        output_traj(path|str): Output of trajectory file
        output_rst(path|str): Output of rst
        output_log(path|str): output logs
        properties(dict): dict of parameters for running sander
    Returns: 
        None
    """
    if not properties:
        properties = {
            'simulation_type': "min_vacuo",
            "mdin": {
                'maxcyc': 500,
                'ntpr': 500,
                'ntr': 1,
                'restraintmask': '\":*&!@H=\"',
                'restraint_wt': 50.0
            }
        }

    sander_mdrun(input_top_path=input_top,
                 input_crd_path=input_crd,
                 input_ref_path=input_ref,
                 output_traj_path=output_traj,
                 output_rst_path=output_rst,
                 output_log_path=output_log,
                 properties=properties)


def convert_amber_to_pdb(input_top, input_crd, output_pdb):
    """"Uses biobb_amber amber to pdb to convert topology and coordinate file to pdb
    Args:
        input_top(str): input topology file
        input_crd(str): input coordinate file
        output_pbd(str): output pdb
    Returns:
        None
    """
    return amber_to_pdb(input_top_path=input_top,
                        input_crd_path=input_crd,
                        output_pdb_path=output_pdb
                        )


class EnergyMinimizer(object):
    def __init__(self,
                 input_pdb,
                 output_pdb=None,
                 output_path=None,
                 amber_properties=None,
                 leap_properties=None,
                 sander_properties=None,
                 ):
        """ Class for runninng energy minimization. Given a PDB, the class first 
        uses the pdb4amber program to prepare the pdb, then runs leap and finally relaxes
        the structure using sander. 
        Args: 
            input_pdb(str|Path): path to the initial pdb
            output_pdb(str|Path): path to save final results
            path(str|Path|None): path to hold intermediate files and output
            amber_properties(dict): 
            leap_properties(dict)
            sander_proeperties
        """

        self.input_pdb = input_pdb
        self.name = Path(input_pdb).name.split('.')[0]

        if not output_path:
            output_path = Path(self.input_pdb).parent / f'{self.name}_outputs'

        if not output_path.exists():
            output_path.mkdir()
        self.output_path = output_path

        # output_pdb is the final output.
        if not output_pdb:
            output_pdb = self.output_path / f'{self.name}.energy_minimized.pdb'
        self.output_pdb = output_pdb

        # pdb4amber settings
        if not amber_properties:
            amber_properties = {'remove_hydrogens': True}
        self.amber_properties = amber_properties
        self.pdb4amber_pdb = self.output_path/'pdb4amber.pdb'

        # leap settings
        if not leap_properties:
            leap_properties = {"forcefield": ["protein.ff19SB"]}
        self.leap_properties = leap_properties
        self.leap_pdb = str(self.output_path/'leap.pdb')
        self.leap_top = str(self.output_path/'leap.top')  # topology
        self.leap_crd = str(self.output_path/'leap.crd')  # coordniate

        # md_sander settings TODO: consult with Kevin
        if not sander_properties:
            sander_properties = {
                'simulation_type': "min_vacuo",
                "mdin": {
                    'maxcyc': 500,
                    'ntpr': 5,
                    'ntr': 1,
                    'restraintmask': '\":*&!@H=\"',
                    'restraint_wt': 50.0
                }
            }

        self.sander_properties = sander_properties
        self.output_traj = str(self.output_path/'sander.x')
        self.output_rst = str(self.output_path/'sander.rst')
        self.sander_log = str(self.output_path/'sander.log')

    def minimize_energy(self, verbose=True):
        """Executes all the steps for the energy minimization. 
        These include 
        1) prepping the pdb file for sander, 
        2) running Leap to further prepare pdb for sander and 
        calculate coordinate and topology files, and
        3) running energy minimization with sander. 
        Args: 
            verbose(bool): if True, print out logs
        """

        if verbose:
            print(f'Starting pdb4amber with {self.input_pdb}. \n')

        run_pdb4amber(input_pdb_path=self.input_pdb,
                      output_pdb_path=self.pdb4amber_pdb,
                      properties=self.amber_properties
                      )

        if verbose:
            print('Finished pdb4amber. \n')
            print(f'Starting Leap with {self.pdb4amber_pdb}. \n')

        run_leap_gen_top(input_pdb_path=self.pdb4amber_pdb,
                         output_pdb_path=self.leap_pdb,
                         output_top_path=self.leap_top,
                         output_crd_path=self.leap_crd,
                         properties=self.leap_properties
                         )

        if verbose:
            print('Finished Leap. \n')
            print(f'Starting Sander with {self.leap_pdb}. \n')

        run_sander(input_top=self.leap_top,
                   input_crd=self.leap_crd,
                   input_ref=self.leap_crd,
                   output_traj=self.output_traj,
                   output_rst=self.output_rst,
                   output_log=self.sander_log,
                   properties=self.sander_properties
                   )

        if verbose:
            print('Finished Sander. \n')
            print('Converting Sander results to pdb. \n')

        convert_amber_to_pdb(self.leap_top, self.output_rst, self.output_pdb)

        if verbose:
            print('Finished conversion \n')

        return self.output_pdb


if __name__ == '__main__':

    input_pdb = Path().cwd().parent / 'Notebooks/abagovomab_forChris.pdb'
    minimizer = EnergyMinimizer(input_pdb)

    minimizer.minimize_energy()


# TODO Write tests
# TODO: add logger maybe
# TODO: add code to extract the graphs.
# TODO: clean up logs. (consider actually adding more paths. )
