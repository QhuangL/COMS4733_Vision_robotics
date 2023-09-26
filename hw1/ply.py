import numpy as np
import os


class Ply(object):
    """Class to represent a ply in memory, read plys, and write plys.
    """

    def __init__(self, ply_path=None, triangles=None, points=None, normals=None, colors=None):
        """Initialize the in memory ply representation.

        Args:
            ply_path (str, optional): Path to .ply file to read (note only
                supports text mode, not binary mode). Defaults to None.
            triangles (numpy.array [k, 3], optional): each row is a list of point indices used to
                render triangles. Defaults to None.
            points (numpy.array [n, 3], optional): each row represents a 3D point. Defaults to None.
            normals (numpy.array [n, 3], optional): each row represents the normal vector for the
                corresponding 3D point. Defaults to None.
            colors (numpy.array [n, 3], optional): each row represents the color of the
                corresponding 3D point. Defaults to None.
        """
        super().__init__()
        # TODO: If ply path is None, load in triangles, point, normals, colors.
        #       else load ply from file. If ply_path is specified AND other inputs
        #       are specified as well, ignore other inputs.
        # TODO: If normals are not None make sure that there are equal number of points and normals.
        # TODO: If colors are not None make sure that there are equal number of colors and normals.
        self.triangles = triangles
        self.points = points
        self.normals = normals
        self.colors = colors
        if ply_path is not None:
            Ply.read(self, ply_path)
        else:
            if self.normals is not None:
                if len(self.points) != len(self.normals):
                    raise Exception('points != normals')
            if self.colors is not None:
                if len(self.colors) != len(self.normals):
                    raise Exception('colors != normals')
            # Ply.write(self, ply_path)

    def write(self, ply_path):
        """Write mesh, point cloud, or oriented point cloud to ply file.

        Args:
            ply_path (str): Output ply path.
        """
        # TODO: Write header depending on existance of normals, colors, and triangles.
        # TODO: Write points.
        # TODO: Write normals if they exist.
        # TODO: Write colors if they exist.
        # TODO: Write face list if needed.

        ply_file = open(ply_path, 'w')
        para = self.points

        ply_file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
''' % (len(self.points)))
        if self.normals is not None:
            para = np.hstack((para, self.normals))
            ply_file.write('''property float nx
property float ny
property float nz
''')
        if self.colors is not None:
            ply_file.write('''property uchar red
property uchar green
property uchar blue
''')
        if self.triangles is not None:
            ply_file.write('''element face %s
property list uchar int vertex_index
''' % (len(self.triangles)))
        ply_file.write('''end_header
''')

        for i in range(para.shape[0]):
            values = [str(value) for value in para[i]]
            ply_file.write(' '.join(values))
            if self.colors is not None:
                ply_file.write(' ')
                ply_file.write(' '.join(str(value) for value in self.colors[i])+ '\n')
            else:
                ply_file.write('\n')
        if self.triangles is not None:
            tri = np.full((len(self.triangles), 1), 3)
            f_tri = np.hstack((tri, self.triangles))
            for i in range(f_tri.shape[0]):
                values = [str(value) for value in f_tri[i]]
                ply_file.write(' '.join(values) + '\n')

        ply_file.close()

        with open(ply_path, "r") as cur_file:
            for line in cur_file:
                print(line)

        print("Write .ply file Done.")

    def read(self, ply_path):
        """Read a ply into memory.
        Args:
            ply_path (str): ply to read in.
        """
        # TODO: Read in ply.
        points = []
        normals = []
        colors = []
        triangles = []

        with open(ply_path, "r") as file:
            reading_points = False
            reading_normals = False
            reading_colors = False
            reading_triangles = False
            for line in file:
                if "element vertex" in line:
                    reading_points = True
                    continue
                if "property float nx" in line:
                    reading_normals = True
                    continue
                else:
                    self.normals = None
                if "property uchar red" in line:
                    reading_colors = True
                    continue
                else:
                    self.colors = None
                if "element face" in line:
                    reading_triangles = True
                    continue
                else:
                    self.triangles = None
                if "end_header" in line:
                    continue
                if line.split('.')[0].isdigit() or line.split()[0].isdigit():
                    if len(line.split()) != 4:
                        if reading_points:
                            point = [float(x) for x in line.split()[:3]]
                            points.append(point)
                            if reading_normals:
                                normal = [float(x) for x in line.split()[3:6]]
                                normals.append(normal)

                                if reading_colors:
                                    color = [int(x) for x in line.split()[6:9]]
                                    colors.append(color)

                            else:
                                if reading_colors:
                                    color = [int(x) for x in line.split()[3:6]]
                                    colors.append(color)


                    else:
                        if reading_triangles:
                            triangle = [int(x) for x in line.split()[1:]]
                            triangles.append(triangle)
                else:
                    continue

            # if reading_normals:
            self.normals = np.array(normals)
            # if reading_colors:
            self.colors = np.array(colors)
            # if reading_triangles:
            self.triangles = np.array(triangles)
            self.points = np.array(points)
            print(self.points)
            print(self.normals)
            print(self.colors)
            print(self.triangles)


