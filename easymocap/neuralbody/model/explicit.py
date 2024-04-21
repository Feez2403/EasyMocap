from .base import Base
from .embedder import get_embedder
import torch.nn.functional as F
import torch
import torch.nn as nn
import pywavefront
from tqdm import tqdm

class CheckBoard(Base):
    def __init__(self, 
        color=[0.5, 0.5, 0.5], # color
        latent={}, # latent code
        sample_args=None
        ) -> None:
        super().__init__(sample_args=sample_args)
        self.color = color
        # set the embed

    def calculate_density_color(self, wpts, viewdir, latents={}, **kwargs):
        # Linear mode
        # wpts: (..., 3)
        # return: (..., 3) for rgb, (..., 1) for occupancy
        alpha = torch.ones(*wpts.shape[:-1], 1, device=wpts.device) * 1.0
        #blue
        rgb = torch.ones(*wpts.shape[:-1], 3, device=wpts.device) * torch.tensor(self.color, device=wpts.device)
        #make a checkerboard pattern
        #if x and y are both even or both odd, then the point is colored 
        #if x and y are different, then the point is black
        x = wpts[..., 0]
        y = wpts[..., 1]
        x = torch.floor(x)
        y = torch.floor(y)
        x = x.int()
        y = y.int()
        x = x % 2
        y = y % 2
        mask = (x == y)
        mask = mask.unsqueeze(-1)
        rgb = rgb * mask.float()
        
        outputs = {
            'occupancy': alpha,
            'rgb': rgb,
            'raw_rgb': rgb,
            'raw_alpha': alpha,
        }
        return outputs
    
    
class Mesh(Base):
    def __init__(self, 
        obj_filepath,
        color=[0.5, 0.5, 0.5], # color
        latent={}, # latent code
        sample_args=None
        ) -> None:
        super().__init__(sample_args=sample_args)
        #self.color = color
        # set the embed
        print('load mesh', obj_filepath)
        self.mesh = pywavefront.Wavefront(obj_filepath, collect_faces=True) # load the mesh
        self.mesh.parse()
        #print('mesh_list', self.mesh.mesh_list)
        #print('vertices', self.mesh.vertices)
        #for name, mesh in self.mesh.meshes.items():
        #    print('name', name)
        #    print('mesh', mesh)
        #    print('faces', mesh.faces)
    
    def calculate_density_color(self, wpts, viewdir):
        # Linear mode
        # wpts: (..., 3)
        # return: (..., 3) for rgb, (..., 1) for occupancy
        alpha = torch.ones(*wpts.shape[:-1], 1, device=wpts.device) * 1.0
        
        x = wpts[..., 0]
        y = wpts[..., 1]
        z = wpts[..., 2]
        #find the triangle that the point is in
        def get_triangle(x, y, z):
            for _, mesh in self.mesh.meshes.items():
                for idx, triangle in enumerate(mesh.faces):
                    trixyz1 = self.mesh.vertices[triangle[0]]
                    trixyz2 = self.mesh.vertices[triangle[1]]
                    trixyz3 = self.mesh.vertices[triangle[2]]
                    #print('triangle', trixyz1, trixyz2, trixyz3)
                    #find the normal of the triangle
                    #cross product of two vectors
                    if x >= trixyz1[0] and x <= trixyz2[0] \
                        and y >= trixyz1[1] and y <= trixyz3[1] \
                        and z >= trixyz1[2] and z <= trixyz2[2]:
                        return idx, [trixyz1, trixyz2, trixyz3]
            return None, None
        
        normals = torch.zeros(*wpts.shape[:-1], 3, device=wpts.device)
        #(B,1,3)
        for i in tqdm(range(wpts.shape[0])):
            print('i', i)
            print('x', x[i], 'y', y[i], 'z', z[i])
            idx, triangle = get_triangle(x[i], y[i], z[i])
            print('triangle', triangle)
            if idx is None:
                continue
            v1 = torch.tensor(triangle[1], device=wpts.device) - torch.tensor(triangle[0], device=wpts.device)
            v2 = torch.tensor(triangle[2], device=wpts.device) - torch.tensor(triangle[0], device=wpts.device)
            normal = torch.cross(v1, v2) 
            normal = normal / torch.norm(normal)
            normals[i,0,:] = normal
        
    

        #blue
        rgb = normals
        outputs = {
            'occupancy': alpha,
            'rgb': rgb,
            'raw_rgb': rgb,
            'raw_alpha': alpha,
        }
        return outputs
        