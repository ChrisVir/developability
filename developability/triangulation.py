import numpy as np
import pandas as pd
from pathlib import Path



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
        return pd.DataFrame(faces, columns=cols)

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

        return np.linalg.norm(np.cross(s1, s2), axis = 1) / 2
        

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