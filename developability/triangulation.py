import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
from developability.input_output import read_potential_file

class TriangulatedSurface: 
    """
    A class for representing a triangulated surface in 3D space.

    Attributes:
        file_type (str): The file type of the surface data.
        num_vertices (int): The number of vertices in the surface.
        num_faces (int): The number of faces in the surface.
        num_edges (int): The number of edges in the surface.
        vertices (pandas.DataFrame): A DataFrame containing the x, y, and z coordinates of each vertex.
        faces (pandas.DataFrame): A DataFrame containing the indices of the vertices that make up each face.
        face_areas (numpy.ndarray): An array containing the area of each face.
        total_area (float): The total area of the surface.

    Methods:
        get_vertices_for_face(face): Returns the indices of the vertices that make up a given face.
        get_vertex_cordinates(vertex): Returns the x, y, and z coordinates of a given vertex.
        compute_face_area(face): Computes the area of a given face.
    """

    def __init__(self, fname): 
        """
        Initializes a TriangulatedSurface object from a file.

        Args:
            fname (str): The name of the OFF file containing the surface data.
        """

        self.text = [line for line in fname.read_text().split('\n') if not line.startswith('#') and line]
        self.file_type = self.text[0]
        self.num_vertices, self.num_faces, self.num_edges = list(map(int, self.text[1].split()))
        self.vertices = self._parse_vertices_()
        self.faces = self._parse_faces_()
        self.face_areas = self._face_areas_
        self.total_area = self.face_areas.sum()
        self.face_dict, self.vertices_dict= self.__generate_face_vertices_dicts__
                         
        # for memoization
        self.sides = {}

    def _parse_lines_(self, start, end, num_type=float): 
        """
        Parses a range of lines from the surface data file.

        Args:
            start (int): The index of the first line to parse.
            end (int): The index of the last line to parse.
            num_type (type): The type of the numbers in the lines.

        Returns:
            list: A list of lists containing the parsed data.
        """
        lines = [ list(map(num_type, line.split(' '))) for line in self.text[start:end]]
        return lines
    

    def _parse_vertices_(self): 
        """
        Parses the vertices from the surface data file.

        Returns:
            pandas.DataFrame: A DataFrame containing the x, y, and z coordinates of each vertex.
        """
        vertices = self._parse_lines_(2, self.num_vertices+2)
        if len(vertices[0]) == 4: 
            cols = ['x', 'y', 'z', self.file_type[-1]]
        else:
            cols = ['x', 'y', 'z']
            
        vertices = pd.DataFrame(vertices, columns=cols)
        return vertices
    

    def _parse_faces_(self): 
        """
        Parses the faces from the surface data file.

        Returns:
            pandas.DataFrame: A DataFrame containing the indices of the vertices that make up each face.
        """
        faces = self._parse_lines_(self.num_vertices+2, self.num_vertices+self.num_faces+2, int)
        
        num_vertices = faces[0][0]
        cols = ['num_vertices'] + [f'v{i}' for i in range(1,num_vertices+1) ]
        return (pd.DataFrame(faces, columns=cols)
                .reset_index()
                .rename(columns={'index':'face'})
                )
        

    def get_vertices_for_face(self, face):
        """
        Returns the indices of the vertices that make up a given face.

        Args:
            face (int): The index of the face.

        Returns:
            tuple: A tuple containing the indices of the vertices that make up the face.
        """ 
        face = self.faces.iloc[face]
        vertices = face.v1, face.v2, face.v3
        return sorted(vertices)
    

    def get_vertex_cordinates(self, vertex): 
        """
        Returns the x, y, and z coordinates of a given vertex.

        Args:
            vertex (int): The index of the vertex.

        Returns:
            numpy.ndarray: An array containing the x, y, and z coordinates of the vertex.
        """

        vertex = self.vertices.iloc[vertex]
        return np.array([vertex.x, vertex.y, vertex.z])


    def compute_face_area(self, face): 
        """
        Computes the area of a given face.

        Args:
            face (int): The index of the face.

        Returns:
            float: The area of the face.
        """
        v1, v2, v3 = self.get_vertices_for_face(face)
        v1 = self.get_vertex_cordinates(v1)
        v2 = self.get_vertex_cordinates(v2)
        v3 = self.get_vertex_cordinates(v3)
        triangle = Triangle(v1, v2, v3)
        return triangle.area
    

    @property
    def __generate_face_vertices_dicts__(self):
        """ generates a dictionary of faces and vertices and a dictionary of vertices and faces
        """
        face_dict = {}
        verticies_dict = {}

        vertices = self.faces[['v1', 'v2', 'v3']].values
    
        for i in range(len(vertices)): 
            face_dict[i] = vertices[i].tolist()
            for vertex in vertices[i]: 
                 verticies_dict.setdefault(vertex, []).append(i)

        return face_dict, verticies_dict
    

    def __share_two_vertices__(self, f1, f2): 
        """Determines if two surface faces share at least two vertices (ie a side)"""

        return len(set(self.face_dict[f1]).intersection(self.face_dict[f2]))>=2


    def __generate_face_face_edges__(self, share_two=True): 
        """ generates a list of edges for a given face
        """
        face_dict = self.face_dict
        vertices_dict = self.vertices_dict

        edges = set()
        for f1 in face_dict: 
            for v in face_dict[f1]: 
                for f2 in vertices_dict[v]: 
                    if f2 != f1:
                        if share_two and self.__share_two_vertices__(f1,f2): 
                            edge = sorted((f1, f2))
                            edges.add(tuple(edge))
                        elif not share_two:
                            edge = sorted((f1, f2))
                            edges.add(tuple(edge))

        return list(edges)

    
    def generate_face_face_graph(self, share_two=True): 
        """Generate a graph of face-face edges.        
        Returns:
            g (nx.Graph): graph of face-face edges
        """
        edges = self.__generate_face_face_edges__(share_two)
        g = nx.from_edgelist(edges)

        # add the face area for each node. 
        add_data_from_dataframe(g, self.face_areas, 'area')

        return g

    
    @property
    def _face_areas_(self): 
        """
            Computes the area of each face in the surface.

        Returns:
            numpy.ndarray: An array containing the area of each face.
        """
        faces = self.faces
        p1 = self.vertices.loc[faces['v1'], ['x', 'y', 'z']].values
        p2 = self.vertices.loc[faces['v2'], ['x', 'y', 'z']].values
        p3 = self.vertices.loc[faces['v3'], ['x', 'y', 'z']].values

        s1 = p1-p2
        s2 = p1-p3

        return pd.DataFrame(np.linalg.norm(np.cross(s1, s2), axis = 1) / 2, columns=['area'])
        

class Point: 
    def __init__(self, x,y,z): 
        """3D Point

        Args:
            x (float): 
            y (float):
            z (float):
        """
        self.x = x
        self.y = y
        self.z = z
        self.vector = np.array([x,y,z])

    def __repr__(self):
        return f'Point({self.x}, {self.y}, {self.z})'
    
    def __str__(self):
        return f'Point({self.x}, {self.y}, {self.z})'
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def dist(self, other): 
        return np.linalg.norm(self.vector - other.vector) 
    
    def length(self): 
        return np.linalg.norm(self.vector)


class Triangle:
    def __init__(self, p1, p2, p3) -> None:
        if not isinstance(p1, Point):
            p1 = Point(*p1)

        if not isinstance(p2, Point):
            p2 = Point(*p2)
        
        if not isinstance(p3, Point):
            p3 = Point(*p3)

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3 

        self.s1 = p1-p2
        self.s2 = p1-p3
        

    def __repr__(self) -> str:
        return f'Triangle({self.p1}, {self.p2}, {self.p3})'
    
    @property
    def area(self) -> float:
        """ area of triangle

        Returns:
            float: area of triangle
        """
        return np.linalg.norm(np.cross(self.s1.vector, self.s2.vector)) / 2
    
############################################################################################################
# Functions for graph traversal                     
############################################################################################################
    
def get_nodes(g, data=True): 
    """gets the node"""
    return list(g.nodes(data=data))


def add_data_from_dataframe(g, df, cols): 
    """Add data from a dataframe to a graph. 
    """
    if isinstance(cols, str): 
        cols = [cols]

    nodes = get_nodes(g)

    for node in nodes: 
        for col in cols: 
            node[1][col] = df.loc[node[0], col]



def aggregate_feature_over_vertices(faces_dict, features, columns,aggfunc=np.mean): 
    """Aggregate a feature over vertices for each face. 
    Args:
        faces_dict (dict): dictionary of faces and vertices
        features (pd.DataFrame): DataFrame of features for each vertex
        columns (list): list of columns to aggregate
        aggfunc (function): function to aggregate features
    Returns:
        agg (pd.DataFrame): DataFrame of aggregated features for each face
    """
    agg = {}

    feature_dicts = {col:features[col].to_dict() for col in columns}

    for face, vertices in faces_dict.items(): 
        agg.setdefault(face, [])
        for col in columns: 
            agg[face].append(aggfunc([feature_dicts[col][v] for v in vertices]))

    agg = pd.DataFrame(agg).T
    agg.columns = columns
    return agg


def break_edges(g, attribute): 
    """Break edges in a graph based on nodes not having same value for attribute. 
    Args: 
        g (nx.Graph): graph to break
        attribute (str): attribute to break on
    Returns:
        g (nx.Graph): graph with edges broken
    """
    h = g.copy()

    for edge in h.edges: 
        if h.nodes[edge[0]][attribute] != h.nodes[edge[1]][attribute]: 
            h.remove_edge(*edge)

    return h
    

def get_connected_components(g, attribute='charge'):
    """Get connected components of a graph that have same attribute
    Args: 
        g (nx.Graph): graph to search
        attribute (str): attribute to search for
    Returns:
        components (list): list of connected components
    """
    h = break_edges(g, attribute)
    components = list(nx.connected_components(h))    
    return components


def aggregate_features_over_components(g, components, attributes, aggfunc=np.sum): 
    """Aggregate components of a graph.
    Args: 
        g (nx.Graph): graph to search
        attribute (str): attribute to search for
    Returns:
        components (list): list of connected components
    """

    agg = {}

    for i, component in enumerate(components): 
        agg.setdefault(i, {})
        agg[i]['size'] = len(component)
        for attribute in attributes: 
            agg[i][attribute] = aggfunc([g.nodes[n][attribute] for n in component])

    return pd.DataFrame(agg).T



###########

def compute_surface_potential_patches(path, share_two=True, save_intermediate_results=True): 
    """"Computes the surface potential patches for a given antibody in path
    
    """

    surface_off = path/'triangulatedSurf.off'
    potential_file = path/'potential_coordinates.csv'

    # compute the surface
    ts = TriangulatedSurface(surface_off)

    
    #generate the graph 
    g  = ts.generate_face_face_graph(share_two)

    ##add potential 
    potential_df = read_potential_file(potential_file)
    potential_df['charge_sign'] = np.sign(potential_df['potential'])
    aggregated_potential = aggregate_feature_over_vertices(ts.face_dict, potential_df, ['potential', 'charge_sign'])
    add_data_from_dataframe(g,aggregated_potential, ['charge_sign', 'potential'] )
    

    # break the graph into subcomponents and get the desired measurements
    components = get_connected_components(g, attribute='charge_sign')

    df = aggregate_features_over_components(g, components, attributes=['area', 'charge_sign', 'potential'], 
                                            aggfunc=np.sum)
    df = df.sort_values('size', ascending=False)
    df['charge_density'] = np.abs(df['potential']/df['area'])
    
    if save_intermediate_results: 
        fname = path/'surface_patch_potential.csv'
        df.to_csv(fname)
    
    features = dict(
        max_size = df['size'].max(),
        max_area = df['area'].max(),
        average_area = df['area'].mean(), 
        area_std = df['area'].std(), 
        max_potential = df['potential'].max(),
        min_potential = df['potential'].min(),
        average_potential = df['potential'].mean(),
        potential_std = df['potential'].std(),
        number_patches =len(df),
        number_positive_patches = len(df.loc[df['charge_sign']==1]),
        number_negative_patches = len(df.loc[df['charge_sign']==-1]),
        max_charge_density = df['charge_density'].max(),
        average_charge_density = df['charge_density'].mean(),
        charge_density_std = df['charge_density'].std()
    )

    features = pd.DataFrame(features, index = [path.name.split('_')[0]])
    
    return features, df