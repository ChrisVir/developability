# code for computing electrostatics via APBS
from sarge import capture_both
from pathlib import Path
from pdb2pqr import psize, inputgen

from .utils import bytes_to_str


def run_pdb2pqr(input_pdb, output_path, options=None, save_log=False):
    """Runs PDB2PQR from commandline to create pqr file from pdb
    for running APBS
    Parameters:
        input_pdb(str|Path): The pdb to convert to pqr.
        save_path(str|Path): dir for saving.
        options(dict|None): dictionary with options
    Returns:
        output_pqr(Path): path to the output pqr file.
    """
    # convert to Path objects incase needed.
    input_pdb = Path(input_pdb)
    save_prefix = Path(output_path) / input_pdb.name
    output_pqr = save_prefix.with_suffix('.pqr')

    if not options:
        options = {'ff': 'AMBER',
                   'with-ph': 7.0,
                   'titration-state-method': 'propka',
                   'drop-water': '',
                   'include-header': ''
                   }

    # construct the command to be executed.
    cmd = ['pdb2pqr']
    for option, value in options.items():
        if value:
            cmd.append(f'--{option}={value}')
        else:
            cmd.append(f'--{option}')

    cmd = ' '.join(cmd)
    cmd = f'{cmd} {input_pdb} {output_pqr}'

    res = capture_both(cmd).stderr
    res = bytes_to_str(res)

    if save_log:
        ((output_path / 'pdb2pqr.log')
         .write_text(res)
         )
    return output_pqr


def run_apbs(input_ini, output_file=None, save_log=True):
    """Wrapper to execute APBS to calculate electrostatics. Requires .ini file
    and pqr file.
    Args:
        input_ini(str|Path): path to ini file for configuring APBS run.
        output_file(str|Path|None): where to save logs.
        save_log(bool): If true save log.
    Returns:
        res(str): stderr
    """
    if not output_file:
        output_file = Path(input_ini).with_suffix('.pqr.log')

    cmd = f'apbs --output-file={output_file} {input_ini}'

    stderr = capture_both(cmd).stderr
    stderr = bytes_to_str(stderr)
    return stderr


def __update_elecs__(input_, options):
    """helper function to update elecs of inputgen.Input object for
    controlling APBS calculations.
    Args:
        input_(inputgen.Input):
        options(dict): dict with option (key, value) pairs.
    returns:
        None (modifies in place)
    """

    elec1 = input_.elecs[0]
    for attribute, value in options.items():
        setattr(elec1, attribute, value)


def generate_apbs_ini(pqr, output_path, options=None, method='mg-auto'):
    """ Generates an apbs ini file to execute apbs
    This is a modified version of the dump_apbs method from pdb2pqr at
    https://github.com/Electrostatics/pdb2pqr/blob/master/pdb2pqr/io.py
    Args:
        pqr(str|Path): Path to the pqr file.
        output_path(str|Path): path to output i
        options(dict): dict of options to set for elec portion of ini file.
        method(str): method used for apbs.
    Returns:
        input_(inputgen.Input)
    """

    if not options:
        # default to set dielectric to 16 as per decriptor paper.
        options = {'pdie': 16.0000}

    # determine the optimal size of the grid
    size = psize.Psize()
    size.parse_input(pqr)
    size.run_psize(pqr)

    # set up the object for outputing the ini
    input_ = inputgen.Input(pqr, size, method, 0, potdx=True)
    input_.print_input_files(output_path)
    __update_elecs__(input_, options)

    # update the pqrname to the path (the program removes this)
    setattr(input_, 'pqrname', input_.pqrpath)

    # write out the ini
    input_.print_input_files(output_path)

    return input_


class APBS(object):

    def __init__(self,
                 input_pdb,
                 output_path=None,
                 pdb2pqr_options=None,
                 apbs_options=None):
        """Class for running APBS part of pipeline.
        This class' calculate_potential method does the following:
        1. Calls pdb2pqr command line to convert the pdb to a pqr file.
            It uses Propka for pka calculations to set the ionization state.
        2.  Generates an ini (input) file for executing APBS.
        3. Runs APBS using the input file.
        Args:
            input_pdb(str|path): path to the pdb
            output_path(str|Path): directory for output
            pdb2pqr_options(dict|None): dict with options for
                                        pb2pqr file.
            apbs_options(dict|None): dict for input parameter file
                to APBS commandline. (see APBS for defaults)
        Returns:
            None
        """

        self.input_pdb = Path(input_pdb)

        if not output_path:
            output_path = self.input_pdb.parent
        
        self.output_path = output_path

        if not output_path.exists():
            output_path.mkdir()

        # set the names of intermediate files based on the input pqdb
        self.pqr_file = output_path / self.input_pdb.with_suffix('.pqr').name
        self.ini_file = output_path / self.input_pdb.with_suffix('.ini').name
        self.dx_file = output_path/self.input_pdb.with_suffix('.pqr.dx').name

        if not pdb2pqr_options:
            pdb2pqr_options = {'ff': 'AMBER',
                               'with-ph': 7.0,
                               'titration-state-method': 'propka',
                               'drop-water': '',
                               'include-header': ''
                               }
        self.pdb2pqr_options = pdb2pqr_options
        if not apbs_options:
            apbs_options = {'pdie': 16.0000}
        self.apbs_options = apbs_options

    def calculate_potential(self, verbose=True):
        """run pipeline to calcuate surface potential via APBS
        Args:
            verbose(bool): if verbose, print out status
        Returns:
            self.pqr_file(Path): path to the pqr file.
            self.dx_file(Path): path to the dx file.
        """
        print(f'Starting pdb2pqr on {self.input_pdb.name}\n')
        _ = run_pdb2pqr(self.input_pdb, self.output_path, self.pdb2pqr_options)

        print('Generating APBS ini file.\n')
        _ = generate_apbs_ini(self.pqr_file, self.ini_file,
                              options=self.apbs_options)

        print('Running APBS. \n')
        print(run_apbs(self.ini_file))

        return self.pqr_file, self.dx_file


if __name__ == '__main__':
    pass
   