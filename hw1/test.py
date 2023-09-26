import numpy as np
from ply import Ply
import os


path1 = '/Users/qianhuangli/Desktop/Spring/aspect of robotics/hw1/hw1/data/point_sample.ply'
path2 = '/Users/qianhuangli/Desktop/Spring/aspect of robotics/hw1/hw1/data/triangle_sample.ply'

path3 = '/Users/qianhuangli/Desktop/Spring/aspect of robotics/hw1/hw1/data/sample.ply'

# points = np.array([[0.,0.,1.], [0.,1.,0.], [1.,0.,0.]])
#
# normals = np.array([[1.,0.,0.],[1.,0.,0.],[1.,0.,0.]])
# colors =np.array([[0,0,155],[0,0,155],[0,0,155]])
# triangles = np.array([[2,1,0]])
#
# point = Ply(points=points, normals=normals, colors=colors, triangles=triangles)
#
# point.write(path3)


points = np.array([[0.,0.,1.],[0.,1.,0.],[1.,0.,0.]] )
normals = np.array([[1.,0.,0.],[1.,0.,0.],[1.,0.,0.]])
colors =np.array([[0,0,155],[0,0,155],[0,0,155]])

point = Ply(points=points, normals=normals, colors=colors,)
point.write(path3)
# Point = Ply(ply_path= path1)





