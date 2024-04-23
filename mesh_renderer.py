import numpy as np


file = r"C:\Users\flori\EasyMocap\data\wildtrack\mesh-bg\FirstPoly.obj"
import pywavefront
scene = pywavefront.Wavefront(file, collect_faces=True)
scene.parse()
print(len(scene.vertices))

for name, material in scene.materials.items():
    print("Material name: " + name)
    print("Diffuse: " + str(material.diffuse))

for mesh in scene.mesh_list:
    print("Mesh name: " + mesh.name)
    print(mesh.faces)
    print("Triangle count: {}".format(len(mesh.faces)))

from pywavefront import visualization
visualization.draw(scene)
        