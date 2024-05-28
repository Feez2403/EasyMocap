import numpy as np
import cv2
import torch.nn as nn
import torch
import time
import json
from ..model.base import augment_z_vals, concat
from ..brdf.renderer import gen_light_xyz

from .render_relight import RelightModule

_time_ = 0
def tic():
    global _time_
    _time_ = time.time()

def toc(name):
    global _time_
    print('{:15s}: {:.1f}'.format(name, 1000*(time.time() - _time_)))
    _time_ = time.time()


def raw2acc(raw):
    alpha = raw[..., -1]
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]
    acc_map = torch.sum(weights, -1)
    return acc_map

def raw2outputs(outputs, z_vals, rays_d, bkgd=None):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        acc: [num_rays, num_samples along ray, 1]. Prediction from model.
        feature: [num_rays, num_samples along ray, N]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        feat_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    #if 'occupancy' in outputs.keys():
    #    alpha = outputs['occupancy'][..., 0]
    #elif 'density' in outputs.keys():
    #    dists = z_vals[..., 1:] - z_vals[..., :-1] # [N_rays, N_samples-1]
    #    
    #    #add a large value to the end of dists
    #    dists = torch.cat([dists,torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists)],-1)  # [N_rays, N_samples]
#
    #    
    #    dists = dists * torch.norm(rays_d, dim=-1)
    #    noise = 0.
    #    # alpha = raw2alpha(raw[..., -1] + noise, dists)  # [N_rays, N_samples]
    #    alpha = 1 - torch.exp(-dists*torch.relu(outputs['density'][..., 0] + noise)) # (N_rays, N_samples_)
    #else:
    #    raise NotImplementedError
    #///////////////////////////////////////////////////////////////////////////////////////////
    #dists = z_vals[..., 1:] - z_vals[..., :-1] # [N_rays, N_samples-1]
    #
    ##add a large value to the end of dists
    #dists = torch.cat([dists,torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists)],-1)  # [N_rays, N_samples]
    #
    #dists = dists * torch.norm(rays_d, dim=-1)
    #noise = 0.
    ## alpha = raw2alpha(raw[..., -1] + noise, dists)  # [N_rays, N_samples]
    #
    ##alpha = 1 - torch.exp(-dists*torch.relu(outputs['density'][..., 0] + noise)) # (N_rays, N_samples_)
    #alpha = 1 - torch.exp(-dists*torch.relu(outputs['occupancy'][..., 0] + noise)) # (N_rays, N_samples_)
    
    alpha = outputs['occupancy'][..., 0]
    
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]
    acc_map = torch.sum(weights, -1)
    # ATTN: here depth must /||ray_d||
    depth_map = torch.sum(weights * z_vals, -1)/(1e-10 + acc_map)/torch.norm(rays_d, dim=-1).squeeze() # [N_rays]
    
    # OPTION 1 : weighted average of the depth from the raw output 
    #depth_raw = torch.sum(weights * z_vals, -1)/torch.norm(rays_d, dim=-1).squeeze() # [N_rays]
    depth_raw = depth_map
    
    # OPTION 2 : first value of z_vals above the threshold
    #threshold = 0.5
    #msk = torch.gt(alpha, threshold)
    #
    #indices = torch.argmax(msk, dim=-1)
    #depth_raw = torch.where(msk.any(dim=-1), z_vals[torch.arange(z_vals.shape[0]), indices], torch.zeros_like(z_vals[:, 0])) / torch.norm(rays_d, dim=-1).squeeze()
    
     
    
    results = {
        'acc_map': acc_map, # [N_rays]
        'depth_map': depth_map, # [N_rays]
        'depth_raw': depth_raw, # [N_rays]
    }
    for key, val in outputs.items():
        if key == 'occupancy':
            continue
        results[key+'_map'] = torch.sum(weights[..., None] * val, -2)  # [N_rays, 3]

    return results


class BaseRenderer(nn.Module):
    def __init__(self, net, chunk, white_bkgd, use_occupancy, N_samples, split,
        
        return_raw=False, return_extra=False, use_canonical=False):
        super().__init__()
        self.net = net
        self.chunk = chunk
        self.white_bkgd = white_bkgd
        self.use_occupancy = use_occupancy
        self.N_samples = 64
        self.split = split
        self.return_extra = return_extra
        self.use_canonical = use_canonical
        if use_canonical:
            self.net.use_canonical = use_canonical
        
        #print("RenderWrapper: ", net.models.keys())
        self.relight = RelightModule(net)

    def compose(self, retlist, mask=None, bkgd=None):
        res = {}
        for key in retlist[0].keys():
            val = torch.cat([r[key] for r in retlist])
            if mask is not None and val.shape[0] != mask.shape[0]:
                val_ = torch.zeros((mask.shape[0], *val.shape[1:]), device=val.device, dtype=val.dtype)
                if key == 'rgb_map': # consider the background
                    if bkgd is None:
                        import ipdb; ipdb.set_trace()
                    elif bkgd is not None and bkgd.shape[0] > 1:
                        val_[~mask] = bkgd[~mask]
                    else:
                        val_[~mask] = bkgd[0]
                val_[mask] = val
                val = val_.unsqueeze(0)
            else:
                val = val.unsqueeze(0)
            res[key] = val
        return res

    def batch_forward(self, batch, viewdir, start, end, bkgd):
        ray_o = batch['ray_o'][0, start:end, None]
        ray_d = batch['ray_d'][0, start:end, None]
        #print(f"ray_o.shape: {ray_o.shape}")
        #print(f"ray_d.shape: {ray_d.shape}")
        n_pixel = ray_o.shape[0]
        
        
        viewdirs = batch['viewdirs'][0, start:end, None].expand(-1, 1, -1)
        keys_all = self.net.keys.copy()
        object_keys = [d[0] for d in batch['meta']['object_keys']]
        
        mapkeys = {}
        operation = {}
        keys_all = object_keys
        for key in object_keys:
            mapkeys[key] = key.split('_@')[0]
            if '_@' in key:
                params = json.loads(key.split('_@')[1].replace("'", '"'))
                operation[key] = params
                
        # print('render keys: ', keys_all)
        ret_all = []
        for key in object_keys:
            #print(f"key: {key}")
            if '@' in key:
                model = self.net.model(mapkeys[key])
                model.current = key
            else:
                model = self.net.model(key)
                # Set the key manually here, because it will not be overwritten automatically in non-share mode.
                model.current = key
            
            mask = batch[key + '_mask'][0]
            start_ = mask[:start].sum()
            end_ = mask[:end].sum()
            near, far = [batch[key+'_'+nearfar][0, start_:end_][:, None] for nearfar in ['near', 'far']]
            mask = mask[start:end]
            #print(f"mask.shape: {mask.shape}")
            if mask.sum() < 1:
                # print('Skip {} [{}, {}]'.format(key, start, end))
                continue
            #print(f"ray_o.shape: {ray_o[mask].shape}")
            #print(f"ray_d.shape: {ray_d[mask].shape}")
            #print(f"near.shape: {near.shape}")
            #print(f"far.shape: {far.shape}")
            z_vals, pts, raw_output = model.calculate_density_from_ray(
                ray_o[mask], ray_d[mask], near, far, self.split)
            
            #raw_output = {'occupancy': alpha,'raw_alpha': raw_alpha}
            #print(f"z_vals.shape: {z_vals.shape}")
            raw_output['z_vals'] = z_vals[..., 0]
            
            instance_ = torch.zeros((*pts.shape[:-1], len(keys_all)), 
                        dtype=pts.dtype, device=pts.device)
            instance_[..., keys_all.index(key)] = 1.
            raw_output['instance'] = instance_
                    
            #density = raw_output['density']
            if self.split == 'train' :
                with torch.no_grad():
                    #for param in self.net.model(key).parameters():
                    #    if param.requires_grad:
                    #        print(f"param: {param}")
                    density = raw_output['occupancy']
                    normal = -torch.autograd.grad(density, pts, torch.ones_like(density), retain_graph=True)[0]
                    raw_output['normal'] = normal
                    #print(f"raw_output.keys: {raw_output.keys()}") # "occupancy", "raw_alpha", "density", "normal
                    #print(f"raw_output['normal'].shape: {raw_output['normal'].shape}")
                    #print(f"raw_output['density'].shape: {raw_output['density'].shape}")
                    
                    
                    #print(f"raw_output['z_vals'].shape: {raw_output['z_vals'].shape}")
                    # add instance
                    
                    #print(f"raw_output['instance'].shape: {raw_output['instance'].shape}")
                    
                    
            raw_padding = {}
            for key_out, val in raw_output.items():
                if len(val.shape) == 1: # for traj
                    raw_padding[key_out] = val
                    continue
                padding = torch.zeros([mask.shape[0], *val.shape[1:]], dtype=val.dtype, device=val.device)
                padding[mask] = val
                raw_padding[key_out] = padding
            ret_all.append(raw_padding)
           
        if len(ret_all) == 0:
            # Completion 0
            occupancy = torch.zeros([ray_d.shape[0], 1, 1], device=ray_d.device)#todo
            color = torch.zeros([ray_d.shape[0], 1, 3], device=ray_d.device)#todo
            instance = torch.zeros([ray_d.shape[0], 1, len(object_keys)], device=ray_d.device)#todo
            z_vals_blank = torch.zeros([ray_d.shape[0], 1], device=ray_d.device)#todo
            blank_output = {'occupancy': occupancy, 'rgb': color, 'instance': instance,#todo
                'raw_alpha': occupancy}#todo
            blank_output['raw_rgb'] = blank_output['rgb']#todo
            ret = raw2outputs(blank_output, z_vals_blank, ray_d, bkgd)#todo
            ret['acc_map'] = torch.zeros([ray_d.shape[0]], device=ray_d.device)#todo
            ret['depth_raw'] = torch.zeros([ray_d.shape[0]], device=ray_d.device)#todo
            ret['surf'] = torch.zeros([n_pixel, 3], device=ray_d.device)#todo
            
            if self.split == 'train':
                ret['normal_map'] = torch.zeros([ray_d.shape[0], 3], device=ray_d.device)#todo
                ret['lvis_hit'] = torch.zeros([ray_d.shape[0], 1], device=ray_d.device)#todo
            print("ret_all == 0")#todo
            return ret
        raw_concat = concat(ret_all, dim=1, unsqueeze=False)
        z_vals = raw_concat.pop('z_vals')
        z_vals_sorted, indices = torch.sort(z_vals, dim=-1)
        # toc('sort')
        ind_0 = torch.zeros_like(indices, device=indices.device)
        ind_0 = ind_0 + torch.arange(0, indices.shape[0], device=indices.device).reshape(-1, 1)
        raw_sorted = {}
        for key, val in raw_concat.items():
            val_sorted = val[ind_0, indices]
            raw_sorted[key] = val_sorted
            #print(f"raw_sorted[{key}].shape: {raw_sorted[key].shape}")
        
        
        ret = raw2outputs(raw_sorted, z_vals_sorted, ray_d, bkgd)
        
        #for key, val in ret.items():
        #    print(f"ret[{key}].shape: {ret[key].shape}")
        
        acc = ret['acc_map'] # (N_rays)
        depth = ret['depth_raw'] # (N_rays)
        #print (f"acc.shape: {acc.shape}")
        #print (f"depth.shape: {depth.shape}")
        #print (f"ray_d.shape: {ray_d.shape}")
        #print (f"ray_o.shape: {ray_o.shape}")   
        surf = ray_o + ray_d * depth.reshape(-1,1,1) # (N_rays,1, 3) xyz of the surface
        if self.split == 'train' :
            with torch.no_grad():
                ret['normal_map'] = torch.nn.functional.normalize(ret['normal_map'], p=2, dim=-1) # (N_rays, 3) normal of the surface
                normal = ret['normal_map']
        
                light_xyz, _ = gen_light_xyz(8,16)
                
                light_xyz = torch.from_numpy(light_xyz).float().to(ray_d.device).reshape(1,-1,3) # (1, N_lights, 3)
                n_lights = light_xyz.shape[1]
                #print(f"n_lights: {n_lights}")
                lvis_hit = torch.zeros(surf.shape[0],n_lights,device=surf.device)
                lpix_chunk = 64
                for i in range(0,n_lights,lpix_chunk):
                    end_i = min(n_lights,i+lpix_chunk)
                    lxyz_chunk = light_xyz[:,i:end_i] # (1, lpix_chunk, 3)
                    #print(f"lxyz_chunk.shape: {lxyz_chunk.shape}")
                    #print(f"surf.shape: {surf.shape}")
                    surf2light = lxyz_chunk - surf # (N_rays, lpix_chunk, 3)
                    surf2light = torch.nn.functional.normalize(surf2light, p=2, dim=-1)
                    surf2lightflat = surf2light.reshape(-1,3)
                    lcos = torch.einsum('ijk,ik->ij', surf2light, normal) # (N_rays, lpix_chunk)
                    
                    # for each pixel, whether each light is in front of the surface
                    front_lit = lcos > 0 # (N_rays, lpix_chunk) 
                    front_lit = front_lit.reshape(-1)
                    if torch.sum(front_lit) == 0:
                        continue
                    surfrep = surf.repeat(1,surf2light.shape[1], 1)
                    surfflat = surfrep.reshape(-1,3)
                    front_surf = surfflat[front_lit, :].reshape(-1,1,3) # ray origin ( N_rays * lpix_chunk,1, 3)
                    front_surf2light = surf2lightflat[front_lit, :].reshape(-1,1,3) # ray direction ( N_rays * lpix_chunk,1, 3)
                    lvis_far = torch.ones(front_surf.shape[:2],device=surf.device) * 0.5 # lvis_far = 20 (N_rays * lpix_chunk,1)
                    lvis_near = torch.ones(front_surf.shape[:2],device=surf.device) * 0.0 # lvis_near = 0.0 (N_rays * lpix_chunk,1)
                    
                    #ret_all = []
                    #for key in object_keys:
                    #    if key in ["ground", "background"]:
                    #        continue
                    #    if '@' in key:
                    #        model = self.net.model(mapkeys[key])
                    #        model.current = key
                    #    else:
                    #        model = self.net.model(key)
                    #        model.current = key
                    #    
                    #    
                    #    #print(f"key: {key}")
                    #    lvis_z, lvis_pts, lvis_raw_output = model.calculate_density_from_ray(
                    #        front_surf, front_surf2light, lvis_near, lvis_far, self.split)
                    #    #print(f"lvis_z.shape: {lvis_z.shape}")
                    #    #print(f"lvis_pts.shape: {lvis_pts.shape}")
                    #    #print(f"lvis_raw_output.keys: {lvis_raw_output.keys()}")
                    #    lvis_raw_output['z_vals'] = lvis_z[..., 0]
                    #                        
                    #    ret_all.append(lvis_raw_output)
                    #
                    #if len(ret_all) == 0:
                    #    print("ret_all_light == 0")
                    #    continue
                    #
                    #raw_concat = concat(ret_all, dim=1, unsqueeze=False)
                    #z_vals = raw_concat.pop('z_vals')
                    #z_vals_sorted, indices = torch.sort(z_vals, dim=-1)
                    #
                    #ind_0 = torch.zeros_like(indices, device=indices.device)
                    #ind_0 = ind_0 + torch.arange(0, indices.shape[0], device=indices.device).reshape(-1, 1)
                    #raw_sorted = {}
                    #for key, val in raw_concat.items():
                    #    val_sorted = val[ind_0, indices]
                    #    raw_sorted[key] = val_sorted
                    #    #print(f"raw_sorted[{key}].shape: {raw_sorted[key].shape}")
                    #
                    #lvis_outputs = raw2outputs(raw_sorted, z_vals_sorted, front_surf2light, bkgd)
    #
                    #lvis_acc = lvis_outputs['acc_map']
                    
                    
                    lvis_z, lvis_pts, lvis_raw_output = model.calculate_density_from_ray(
                            front_surf, front_surf2light, lvis_near, lvis_far, self.split)
                    lvis_outputs = raw2outputs(lvis_raw_output, lvis_z[..., 0], front_surf2light, bkgd)
                    lvis_acc = lvis_outputs['acc_map']
                    
                    tmp = torch.zeros(lvis_hit.shape, dtype=bool)
                    front_lit = front_lit.reshape(n_pixel, lpix_chunk)
                    tmp[:, i:end_i] = front_lit
                    lvis_hit[tmp] = 1 - lvis_acc
                ret['lvis_hit'] = lvis_hit
             
        ret['surf'] = surf.reshape(n_pixel, 3)
        
        
        
        
        # toc('render')
        return ret

    def forward_multi(self, batch, bkgd):
        keys = [d[0] for d in batch['meta']['keys']]
        # prepare each model
        latent_features = {}
        res_cache = {}
        for key in self.net.keys:
            model = self.net.model(key)
            model.clear_cache()
        for key in keys:
           #if '@' in key:
           #    key0 = key.split('_@')[0]
           #    model = self.net.model(key0)
           #    model.current = key
           #else:
            model = self.net.model(key)
            model.before(batch, key)
            
            # save the latent features 
            if "human" in key : 
                latent_features[key] = model.sparse_feature[key]
            elif "background" in key:
                latent_features[key] = model.cache['embed']
            elif "ground" in key:
                latent_features[key] = model.cache['embed']
            
            if key in model.cache.keys():
                res_cache[key+'_cache'] = model.cache[key]
        viewdir = batch['viewdirs'][0].unsqueeze(1)
        retlist = []
        for bn in range(0, viewdir.shape[0], self.chunk):
            #print(f"bn: {bn} / {viewdir.shape[0]}")
            start, end = bn, min(bn + self.chunk, viewdir.shape[0])
            ret = self.batch_forward(batch, viewdir, start, end, bkgd)
            if ret is not None:
                retlist.append(ret)
            
        
        #print(f"rgb.shape: {rgb.shape}")
            
        res = self.compose(retlist)
        
        #if batch["meta"]["step"] % 10 == 0:
        #    coords = batch['coord']
#
        #    import matplotlib.pyplot as plt
        #    rgb = res['normal_map']
        #    rgb = torch.abs(rgb)
        #    rgb = rgb[0].detach().cpu().numpy()
        #    coords = coords[0].cpu().numpy()
        #    blank = np.zeros((1080,1920,3)) 
        #    print(f"coords.shape: {coords.shape}")
        #    print(f"rgb.shape: {rgb.shape}")
        #    
        #    blank[coords[:,0],coords[:,1]] = rgb
        #    
        #    plt.imshow(blank)
        #    plt.show()
        #
        #blank = np.zeros((1080,1920))
        #blank[coords[:,0],coords[:,1]] = res['acc_map'][0].detach().cpu().numpy()
        #plt.imshow(blank)
        #plt.show()
        #
        #blank = np.zeros((1080,1920))
        #blank[coords[:,0],coords[:,1]] = res['depth_raw'][0].detach().cpu().numpy()
        #print("max depth: ", np.max(res['depth_raw'][0].detach().cpu().numpy()))
        #print("min depth: ", np.min(res['depth_raw'][0].detach().cpu().numpy()))
        #plt.imshow(blank)
        #plt.show()
        #
        #print("res['lvis_hit'].shape: ", res['lvis_hit'].shape)
        #lvis_hit_sum = torch.sum(res['lvis_hit'], dim=-1)
        #blank = np.zeros((1080,1920))
        #blank[coords[:,0],coords[:,1]] = lvis_hit_sum.detach().cpu().numpy()
        #plt.imshow(blank)
        #plt.show()
        
        
        #blank = np.zeros((1080,1920))
        #blank[coords[:,0],coords[:,1]] = res['depth_raw'][0].detach().cpu().numpy() 
        #plt.imshow(blank)
        #plt.show()
        
        
        #surf = res['surf'][0].detach().cpu().numpy()
        #ray_o = batch['ray_o'][0].cpu().numpy()
        #print (f"surf.shape: {surf.shape}")
        #print (f"ray_o.shape: {ray_o.shape}")
        #dist = np.linalg.norm(surf - ray_o, axis=-1)
        #dist = dist / np.max(dist)
        #
        #blank = np.zeros((1080,1920))
        #blank[coords[:,0],coords[:,1]] = dist
        #
        #plt.imshow(blank)
        #plt.show()
        
        
        # add cache
        res.update(res_cache)
        res['keys'] = keys
        res['latent_features'] = latent_features
        return res

    def forward(self, batch):
        #print("batch.keys(): ", batch.keys())
        #print("batch['meta'].keys(): ", batch['meta'].keys())
        keys = [d[0] for d in batch['meta']['keys']]
        rand_bkgd = None
        device = batch['rgb'].device
        #print(batch.keys())
        #print(batch['rgb'].shape)
        if self.split == 'train':
            rand_bkgd = torch.rand(3, device=device).reshape(1, 1, 3)
        else:
            if self.white_bkgd:
                rand_bkgd = torch.ones(3, device=device).reshape(1, 1, 3)
        #if len(keys) == 1:
        #    results = self.forward_single(batch, rand_bkgd)
        #else:
        
        results = self.forward_multi(batch, rand_bkgd)

        # set the random background in target
        if self.split == 'train':
            idx = torch.nonzero(batch['rgb'][0, :, 0] < 0)
            if rand_bkgd is not None:
                batch['rgb'][0, idx] = rand_bkgd
            else:
                batch['rgb'][0, idx] = 0.
        
        out = self.relight(batch, results, mode = self.split)
        if self.split != 'train' :
            out["keys"] = keys
            out["acc_map"] = results["acc_map"]
            out["depth_map"] = results["depth_map"]
            out["instance_map"] = results["instance_map"]
        print("render_base : out.keys(): ", out.keys())
        return out

    def compute_loss(self, batch, ret, **loss_kwargs):
        loss = self.relight.compute_loss(batch, ret, **loss_kwargs)
        return loss
