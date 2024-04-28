import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from easymocap.mytools.camera_utils import write_camera, read_camera
import pywavefront
from tqdm import tqdm


class RasterCompose():
    def __init__(self):
            
        root = r"data\wildtrack"
        file = r"data\wildtrack\mesh-bg\mountains.obj"

        self.scene = pywavefront.Wavefront(file, collect_faces=True)
        self.scene.parse()

        camera_id = "1"
        self.cams = read_camera(root + "/intri.yml", root + "/extri.yml", ["1", "2", "3", "4", "5", "6", "7"])

        #P = np.dot(cams[camera_id]["K"], np.hstack((cams[camera_id]["R"], cams[camera_id]["T"])))
#
#
        #image_path = root + f"/images/{camera_id}/000000.jpg"
        #image = cv2.imread(image_path)
#
        #R = cams[camera_id]["R"]
        #T = cams[camera_id]["T"]
        #cam_pos = -np.dot(R.T, T)
#
        #z_buffer = np.ones((image.shape[0], image.shape[1])) * np.inf
        #image = np.zeros_like(image)

    def edge_function(self, v0, v1, v2):
        return (v1[...,0] - v0[...,0]) * (v2[...,1] - v0[...,1]) - (v1[...,1] - v0[...,1]) * (v2[...,0] - v0[...,0])

    def clamp_to_int(self, x, a, b):
        return int(np.maximum(a, np.minimum(b, x)))

    def rasterize_triangle(self,vertices, dist, W, H, z_buffer, rgb, img):
        # rasterize triangle
        min_x = self.clamp_to_int(np.min(vertices[:, 0]), 0, H-1)
        min_y = self.clamp_to_int(np.min(vertices[:, 1]), 0, W-1)

        max_x = self.clamp_to_int(np.max(vertices[:, 0]+1), 0, H-1)
        max_y = self.clamp_to_int(np.max(vertices[:, 1]+1), 0, W-1)
        
        if min_x == max_x or min_y == max_y:
            return None
        v1, v2, v3 = vertices
        A = self.edge_function(v1, v2, v3)
        if A < 0:
            return None
        
        xs, ys = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
        pts = np.dstack((xs, ys))
        #print(f"[{min_x}:{max_x}, {min_y}:{max_y}]")
        w0 = self.edge_function(v2, v3, pts)
        w1 = self.edge_function(v3, v1, pts)
        w2 = self.edge_function(v1, v2, pts)
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        mask = pts[mask]
        dist_mask = dist < z_buffer[mask[:,1], mask[:,0]]
        mask = mask[dist_mask]
        z_buffer[mask[:,1], mask[:,0]] = dist

        img[mask[:,1], mask[:,0]] = rgb
        
    def render(self, K, R, T, H, W, z_buffer=None, image=None):
        
        P = np.dot(K, np.hstack((R, T)))
        if image is None:
            image = np.zeros((H, W, 3))
        if z_buffer is None:
            z_buffer = np.ones((H, W)) * np.inf
        cam_pos = -np.dot(R.T, T)
        
        for mesh in self.scene.mesh_list:
            print("Mesh name: " + mesh.name)
            #rgb = scene.materials[mesh.material].diffuse
            
            for face in tqdm(mesh.faces):
                
                vertices = [self.scene.vertices[i] for i in face]
                vertices = np.array(vertices)
                invert_yz = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
                scale_10 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) *1.4
                translate = np.array([0, 0, 0])
                vertices = np.dot(np.dot(invert_yz, scale_10), vertices.T).T + translate
                normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                normal = normal / np.linalg.norm(normal)
                
                rgb = np.abs(normal)
                
                
                vertices_cam = (np.dot(R, vertices.T) + T).T
                if np.any(vertices_cam[:, 2] <= 0):
                    continue
                v2d = np.dot(K, vertices_cam.T)
                
                
                v2d = v2d[:2] / v2d[2]
                v2d = v2d.T # (x, y)
                
                dist = np.linalg.norm(np.mean(vertices, axis=1) - cam_pos)
                
                if dist < 2:
                    continue
                self.rasterize_triangle(v2d, dist, image.shape[0], image.shape[1], z_buffer, rgb, image)
                
        return image , z_buffer

    def render_camera(self, camera_id):
        K = self.cams[camera_id]["K"]
        R = self.cams[camera_id]["R"]
        T = self.cams[camera_id]["T"]
        H, W = 1080, 1920
        image ,_ = self.render(K, R, T, H, W)
        return image

def create_center_radius(center, radius=10., up='y', ranges=[0, 360, 36], angle_x=0, **kwargs):
    center = np.array(center).reshape(1, 3)
    thetas = np.deg2rad(np.linspace(*ranges))
    st = np.sin(thetas)
    ct = np.cos(thetas)
    zero = np.zeros_like(st)
    Rotx = cv2.Rodrigues(np.deg2rad(angle_x) * np.array([1., 0., 0.]))[0]
    if up == 'z':
        center = np.stack([radius*ct, radius*st, zero], axis=1) + center
        R = np.stack([-st, ct, zero, zero, zero, zero-1, -ct, -st, zero], axis=-1)
    elif up == 'y':
        center = np.stack([radius*ct, zero, radius*st, ], axis=1) + center
        R = np.stack([
            +st,  zero,  -ct,
            zero, zero-1, zero, 
            -ct,  zero, -st], axis=-1)
    R = R.reshape(-1, 3, 3)
    R = np.einsum('ab,fbc->fac', Rotx, R)
    center = center.reshape(-1, 3, 1)
    T = - R @ center
    RT = np.dstack([R, T])
    return RT

if __name__ == "__main__":
    
    rc = RasterCompose()
   #K = rc.cams["1"]["K"]
   #RT = create_center_radius([0, 0, 2], 10., up='y', ranges=[0, 360, 36], angle_x=10)
   #print(RT.shape)
   #for i in range(36):
   #    
   #    image = rc.render(K, RT[i, :3, :3], RT[i, :3, 3].reshape(3, 1), 1080, 1920)
   #    cv2.imwrite(f"image_{i:03d}.png", image)

    image = rc.render_camera("1")
    plt.imshow(image)
    plt.show()


