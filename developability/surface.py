from pathlib import Path
from importlib.resources import files

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from sarge import capture_both

from .utils import bytes_to_str
from .input_output import (pqr_to_xyzr, 
                          save_surface_xyz_to_csv, 
                          read_pqr,
                          save_surface_xyz_atoms_to_csv
                          )

MULTIVIEW_PATH = Path('/Users/ChristopherRivera/bioprograms/APBS-3.4.1.Darwin/share/apbs/tools/bin')

def generate_prm_file(xyzr_filename='', output_path=None, options=None): 
    """ generates a prm (parameter) file for running nanoshaper
    Args: 
        xyzr_filename(str|Path): the path to the input xyzr file 
        output_path(str|Path): The path to save the prm file to. 
        options(dict): options allow update parameters for prm file. 
    Returns: 
        rendered_prm(str): the rendered prm as a str. 
    """
    
    # there are many options in nanoshaper, however, I elect to keep these ones.
    # These are the only options one can update for now, without updating the template.prm. 

    base_options = {'surface': 'ses', 
                    'grid_perfil': 80.0,
                    'number_thread': 16, 
                    'vertex_atom_info': 'true', 
                    'surface_file_name': 'triangulatedSurf.off',
                    'xyzr_filename': xyzr_filename
                    }

    if options: 
        base_options.update(options)

    # use jinja to render a crm file from template
    template_path = files('developability') # for now keeping the template in same dir
    environment = Environment(loader = FileSystemLoader(template_path))
    template = environment.get_template('template.prm')
    rendered_prm =template.render(base_options)
    
    # consider removing below. 
    if not output_path: 
        output_path = Path().cwd()

    output_path = Path(output_path) / 'conf.prm'
    output_path.write_text(rendered_prm)

    return rendered_prm

    
def run_nanoshaper(prm_file, save_log=True, working_directory=None): 
    """Runs nanoshaper from command line using the input prm file. 
    Args: 
        prm_file(str|Path): Path to the prm file
        save_log(bool): if true save log file to same location as prm
        working_directory(None|Path): The working directory to run command from. 
                                     Note that Nanoshaper saves the files to wd. 

    Returns: 
        log(str): the output log. 
    """

    if not working_directory: 
        working_directory = Path(prm_file).parent
    
    cmd = f"NanoShaper {prm_file}"
    
    log = capture_both(cmd, cwd = working_directory).stdout
    log = bytes_to_str(log)
    
    if save_log:  
        output = Path(prm_file).parent/ 'NanoShaper.log'
        output.write_text(log)

    return log


def run_multivalue(coordinates, dx, multiview_path = None, output = None): 
    """Executes multivalue from command line to obtain potential for desired coordinates from dx file
    Note that multivalues is insalled with the APBS package. 

    Args: 
        coordinates(Path/str): path to coordinate csv file
        dx(Path/str): path to dx file with electrostatics from ABS
        mulitiview_path(Path/str): path to directory with multiview binary
    Returns: 
        None
    """

    # remove this in the future. 
    if not multiview_path:  
        multivalue_path = MULTIVIEW_PATH 
    
    multivalue = multivalue_path/'multivalue'

    if not output:
        output = 'potential_coordinates.csv'

    cmd = f"{multivalue} {coordinates} {dx} {output}"
    print(cmd)

    capture_both(cmd)

    return None


def sum_atom_potential(potential_coordinates_file=None, atom_coordinates_file=None): 
    """ Sums up the potentials per atom.
    Args: 
        potential_coordinates_file (str|Path): path to file with potential and coordinates.
        atom_coordinates_file (str|Path): path to file with atom and coordinates
    Returns: 
    
    """

    def num_positive(x): 
        return (x>=0).sum()

    def percent_positive(x): 
        return (x>=0).mean()
        
    pot_df = pd.read_csv(potential_coordinates_file, 
                         header = None, 
                         names = ['x','y','z', 'pot']
                         )
    atoms_df = pd.read_csv(atom_coordinates_file)
    
    pot_atoms_df = (pot_df.merge(atoms_df)
                    .sort_values('atom')
                    .reset_index(drop=True)
                   )
    #Nanoshaper uses numbering starting at zero. 
    pot_atoms_df['atom']+=1
    
    res =  (pot_atoms_df.groupby('atom')['pot']
            .agg([len, 'sum', 'mean',num_positive, percent_positive ])
            .reset_index())
    
    names_map = {'len':'number', 'sum':'total_pot', 'mean':'mean_pot', 'atom':'Atom_number'}

    return res.rename(names_map, axis = 1)


def map_potential_to_atoms(pqr_file, potential_coordinates_file=None, atom_coordinates_file=None, output_dir=None): 
    """Map Electric Surface Potential to atoms on PQR file. Saves output as csv and returns df.
    Args: 
        pqr(Path/str): path to pqr file
        potential_coordinates_file (Path/str): path to the coordinate file with potential
        atom_coordinates_file(Path/str): Path to to the coordinate with atoms.
        output_dir (str|Path): path to save output file.  
    Returns: 
        atom_pot_df(pd.DataFrame)
    """
    pqr_df = read_pqr(pqr_file)
    atom_pot_df = sum_atom_potential(potential_coordinates_file, atom_coordinates_file)

    atom_pot_df = pqr_df.merge(atom_pot_df, how='left')
    
    # TODO: consider renaming this. 
    file_name = output_dir/'atom_potential.csv'
    atom_pot_df.to_csv(file_name)
    return atom_pot_df


def sum_potential_over_residues(atom_pot_df, output_dir): 
    """ Given a dataframe with the potentials summed per atoms, sums up over residues
    Args: 
        atom_pot_df(pd.DataFrame): dataframe with potential and atoms. 
        output_dir(str|Path): path to save output

    Returns: 
        residue_pot_df(pd.DataFrame)
    """
    ##DRY (I will refactor this later in actual repo). 
    
    def num_positive(x): 
        return (x>=0).sum()

    def percent_positive(x): 
        return (x>=0).mean()
    
    residue_pot_df = (atom_pot_df.groupby(['Residue_number','Residue_name'])
           ['total_pot']
            .agg([len, 'sum', 'mean',num_positive, percent_positive ])
            .reset_index()
            .rename(
                {'len':'number_atoms', 
                 'sum':'total_pot', 
                 'mean':'mean_pot', 
                 'atom':'Atom_number'}
                 , axis = 1)
          )
    filename = output_dir/'residue_potential.csv'
    residue_pot_df.to_csv(filename)
    return residue_pot_df

  

class SurfacePotential(object):

    def __init__(self, 
                 input_pqr, 
                 input_dx, 
                 output_dir=None,
                 nanoshaper_options=None 
                 ):
        """Given an pqr file and a dx with potential calculated by 

        Args:
            input_pqr (str|Path): path to the pqr file
            input_dx (str|Path): path to the dx file
            output_dir (None|str|Path): path to save outputs and intermediate files
            nanoshaper_options (dict): options for configuring nanoshaper
        """

        self.input_pqr = Path(input_pqr)
        self.input_dx = Path(input_dx)

        if not output_dir: 
            output_dir = Path(input_pqr).parent
        self.output_dir = output_dir

        if not output_dir.exists(): 
            output_dir.mkdir()

        self.nanoshaper_options = nanoshaper_options
        # generate the requried files from above and save to desired output_dir
        self.xyzr_file, self.prm_file = self._generate_input_files_()
        
        # these are the expected names of gnerated files 
        self.surface_file = self.output_dir/'triangulatedSurf.off'
        self.coordinates_file = self.output_dir/'triangulatedSurf.csv'
        self.atom_coordinates_file = self.output_dir/'triangulatedSurf.atom.csv'
        self.potential_coordinates_file = self.output_dir/'potential_coordinates.csv'
        self.atom_potential_file = self.output_dir/'atom_potential.csv'
        self.residue_potential_file = self.output_dir/'residue_potential.csv'

    def _generate_input_files_(self): 
        """Generates the input files such as the prm file and xyzr file and returns their paths. 
        Returns: 
            xyzr_file, prm_file: tuple[Paths]
        """
        
        # generate the xyzr file
        xyzr_name = self.input_pqr.with_suffix('.xyzr').name
        xyzr_file =  self.output_dir / xyzr_name
        _ = pqr_to_xyzr(self.input_pqr, xyzr_file )

        # generate the the prm file
        prm_file = self.output_dir / 'conf.prm'
        generate_prm_file(xyzr_filename=xyzr_file, 
                          output_path=self.output_dir, 
                          options=self.nanoshaper_options)
        return xyzr_file, prm_file
    

    def calculate_surface_potential(self, verbose = True): 
        """"calculates the surface potential. 
        Creates 3 files of interest:
        1) The surface off file with the surface coordinates
        2)
        3) a csv file with potentials mapped to the coordinates  
        Args: 
            verbose(bool): if True, print out status
        Returns: 
            residue_potential_file(Path): path to the results file
            
        """

        print(f'Starting Nanoshaper on {self.xyzr_file.name} file. \n')
        _= run_nanoshaper(self.prm_file, self.output_dir)
        
        print('Finished Nanoshaper, now creating coordinates.csv.\n')
        save_surface_xyz_to_csv(self.surface_file)
        save_surface_xyz_atoms_to_csv(self.surface_file)

        print('Finished extracting xyz and atoms.\n')
        print('Now extracting pertinent potential values using multivalue.\n')
        
        run_multivalue(self.coordinates_file, 
                       self.input_dx, 
                       output = self.potential_coordinates_file 
                       )
        
        print('Finished extracting potential values.')
        print('Now mapping potential to individual atoms in  pqr_file.\n')
        atom_pot_df = map_potential_to_atoms(pqr_file =self.input_pqr, 
                                         potential_coordinates_file=self.potential_coordinates_file, 
                                         atom_coordinates_file=self.atom_coordinates_file, 
                                         output_dir= self.output_dir
                                         )
        
        print('Finished mapping potentials to atoms.')
        print('Now aggregating potentials over residues.\n')
        results =sum_potential_over_residues(atom_pot_df, self.output_dir)
        
        print('Finished...')
        return self.residue_potential_file
         
if __name__ == '__main__': 
    output = Path('/Users/ChristopherRivera/Projects/developability/developability/data/pdbs/test')
    input_pqr = output/'abagovomab_forChris.energy_minimized.pqr'
    input_dx = output/'abagovomab_forChris.energy_minimized.pqr.dx'
    
    
    sp = SurfacePotential(input_pqr, input_dx, output)
    results = sp.calculate_surface_potential()
