
from os.path import basename
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..brdf.microfacet import Microfacet
from ..model.embedder import Embedder
from ..brdf.renderer import gen_light_xyz

from collections import OrderedDict

from .img_utils import one_hot_img, func_linear2srgb, safe_acos, read_hdr
import os
import os.path as osp
import numpy as np
import copy
import cv2

import matplotlib.pyplot as plt
class RelightModule(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.mlp_chunk= 1024
        self.mlp_width= 256
        self.mlp_depth= 4
        self.mlp_skip_at= 1
        self.xyz_scale= 1.0
        self.n_freqs_xyz= 10
        self.n_freqs_ldir= 4
        self.n_freqs_vdir= 4
        self.n_freqs= 2
        self.light_h= 8
        self.NLights = self.light_h*self.light_h*2
        self.xyz_jitter_std= 0.01
        self.smooth_use_l1= True
        self.olat_inten= 50.
        self.ambient_inten= 0.
        self.normal_loss_weight= 1.
        self.lvis_loss_weight= 1.
        self.normal_smooth_weight= 0.01
        self.lvis_smooth_weight= 0.5
        self.albedo_slope= 0.77
        self.albedo_bias= 0.03
        self.pred_brdf= True
        self.use_xyz= True
        self.albedo_smooth_weight= 0.005
        self.brdf_smooth_weight= 0.00001
        self.specular= True
        self.light_tv_weight= 0.000005
        self.learned_brdf_scale= 1.
        self.light_init_max= 1
        self.light_achro_weight= 0.
        self.achro_light= True
        self.linear2srgb= True
        self.normalize_z= False
        self.fresnel_f0= 0.04
        self.fix_light= True
        # Use Gaussian light
        self.gaussian_light= True
        
        self.lvis_far= 0.5
        self.lvis_near= 0.0
        self.perturb_light= 0.0
        #self.albedo_sparsity= 0.0005
        self.albedo_sparsity= 0.0005
        
        self.z_dim = 1
        self.shape_mode = "scratch"
        self.normalize_brdf_z = False
        self.use_shape_sup = True
        
        self.predict_normals = True
        
        self.predict_normals_lvis_only = False
        
        if self.predict_normals_lvis_only:
            assert self.predict_normals
        
        #self.train_relight= True
        
        self.net_dict = nn.ModuleDict()
        self.embedder = self._init_embedder()
        for key in net.models.keys():
            #print("RelightModule: ", key)
            if key != 'background':
                self.net_dict[key] = self._init_net()
                print("RelightModule: ", key + " nparams : " + str(np.sum([p.numel() for p in self.net_dict[key].parameters()])))
        #self.net = self._init_net()

        self.net = net
        
        light_xyz, _ = self._gen_lights()
        self.light_xyz = light_xyz.reshape(1,-1,3)
        
        
        light_h = self.light_h
        self.light_res = (light_h, 2*light_h)
        lxyz, lareas = self._gen_lights()
        self.lxyz, self.lareas = lxyz, lareas
        maxv = self.light_init_max
        
        if self.gaussian_light:
            #gaussian_light
            light_pos = torch.rand(2) * torch.Tensor(self.light_res).float() # xy position of the gaussian
            light_intensity = torch.rand(1) * 3 # intensity of the gaussian
            light_sigma = torch.rand(1) * 0.5 + 0.5 # sigma of the gaussian
            light_ambient = torch.rand(1) * 0.5  # ambient light, constant term

            self.pos = torch.nn.Parameter(light_pos.cuda())
            self.intens = torch.nn.Parameter(light_intensity.cuda())
            self.sigma = torch.nn.Parameter(light_sigma.cuda())
            self.ambient = torch.nn.Parameter(light_ambient.cuda())
               
            
        else :
            if self.achro_light:
                light = torch.abs(torch.randn(self.light_res + (1,))*maxv)
            else:
                light = torch.abs(torch.randn(self.light_res + (3,))*maxv)
            self._light = nn.parameter.Parameter(light)
            
        
        
        # Novel lighting conditions for relighting at test time:
        # (1) OLAT
        self.do_relight_olat = True
        
        novel_olat = OrderedDict()
        light_shape = self.light_res + (3,)
        olat_inten = self.olat_inten
        ambient_inten = self.ambient_inten
        ambient = ambient_inten*torch.ones(light_shape, device=torch.device('cuda:0'))
        olats = [ 10,  22,  37,  63,  64,  84,  99, 121] # light index, randomly selected
        idx = -1
        for i in range(self.light_res[0]):
            for j in range(self.light_res[1]):
                idx += 1
                if idx in olats:
                    one_hot = one_hot_img(*ambient.shape, i, j)
                    envmap = olat_inten * one_hot + ambient
                    #plt.imshow(arr)
                    #plt.show()
                    novel_olat['%04d-%04d' % (i, j)] = envmap
        self.novel_olat = novel_olat
        # (2) Light probes
        self.do_relight_probes = True
        
        novel_probes = OrderedDict()
        for path in sorted(os.listdir('light-probes/')):
            if '.hdr' in path:
                name = basename(path)[:-len('.hdr')]
                arr = read_hdr(osp.join('light-probes/', path))
                arr = cv2.resize(arr, (self.light_res[1], self.light_res[0]), interpolation=cv2.INTER_LINEAR)
                #plt.imshow(arr)
                #plt.show()
                tensor = torch.from_numpy(arr).cuda()
                novel_probes[name] = tensor
        self.novel_probes = novel_probes
    
    def _init_embedder(self):
        kwargs = {
            'input_dims': 3,
            'include_input': True,
            'max_freq_log2': self.n_freqs_xyz - 1,
            'num_freqs': self.n_freqs_xyz,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos]
        }
        embedder_xyz = Embedder(**kwargs)
        kwargs['max_freq_log2'] = self.n_freqs_ldir - 1
        kwargs['num_freqs'] = self.n_freqs_ldir
        embedder_ldir = Embedder(**kwargs)
        kwargs['max_freq_log2'] = self.n_freqs_vdir - 1
        kwargs['num_freqs'] = self.n_freqs_vdir
        embedder_vdir = Embedder(**kwargs)
        embedder = {
            'xyz': embedder_xyz, 'ldir': embedder_ldir, 'vdir': embedder_vdir
        }
        return embedder
        
    def _init_net(self, feature_dim = 256):
        mlp_width = self.mlp_width
        mlp_depth = self.mlp_depth
        mlp_skip_at = self.mlp_skip_at
        net = nn.ModuleDict()
        
        net['latent_fc'] = nn.Linear(256, 256)
        if self.predict_normals:
        # Normals
            net['normal_mlp'] = MLP(
                feature_dim + 2*self.n_freqs_xyz*3+3, [mlp_width]*mlp_depth, act=['relu']*mlp_depth, skip_at=[mlp_skip_at]
            )
            net['normal_out'] = MLP(
                mlp_width, [3], act=None
            )
            
        net['lvis_mlp'] = MLP(
            feature_dim + 2*self.n_freqs_xyz*3+3+2*self.n_freqs_ldir*3+3, [mlp_width]*mlp_depth, act=['relu']*mlp_depth, skip_at=[mlp_skip_at]
        )
        net['lvis_out'] = MLP(
            mlp_width, [1], act=['sigmoid']
        )
        if self.use_xyz:
            net['albedo_mlp'] = MLP(
                feature_dim + 2*self.n_freqs_xyz*3+3, [mlp_width]*mlp_depth, act=['relu']*mlp_depth, skip_at=[mlp_skip_at]
            )
        else:
            net['albedo_mlp'] = MLP(
                feature_dim, [mlp_width]*mlp_depth, act=['relu']*mlp_depth, skip_at=[mlp_skip_at]
            )
        net['albedo_out'] = MLP(
            mlp_width, [3], act=['sigmoid']
        )
        # brdf
        if self.pred_brdf:
            if self.use_xyz:
                net['brdf_z_mlp'] = MLP(
                    feature_dim + 2*self.n_freqs_xyz*3+3, [mlp_width]*mlp_depth, act=['relu']*mlp_depth, skip_at=[mlp_skip_at]
                )
            else:
                net['brdf_z_mlp'] = MLP(
                    feature_dim, [mlp_width]*mlp_depth, act=['relu']*mlp_depth, skip_at=[mlp_skip_at]
                )
            net['brdf_z_out'] = MLP(mlp_width, [self.z_dim], act=['sigmoid']) # [0, 1]
        return net
    
    
    def _eval_brdf_at(self, pts2l, pts2c, normal, albedo, brdf_prop):
        """Fixed to microfacet (GGX).
        """
        
        rough = brdf_prop
        fresnel_f0 = self.fresnel_f0
        microfacet = Microfacet(f0=fresnel_f0)
        brdf = microfacet(pts2l, pts2c, normal, albedo=albedo, rough=rough)
        return brdf # NxLx3
                 
    def _gen_lights(self):
        light_h = int(self.light_h)
        light_w = int(light_h * 2)
        lxyz, lareas = gen_light_xyz(light_h, light_w)
        lxyz = torch.from_numpy(lxyz).float().cuda()
        lareas = torch.from_numpy(lareas).float().cuda()
        return lxyz, lareas
    
    def _get_ldir(self, pts):
        bs, n_lights, _ = self.light_xyz.shape
        _, n_rays, _ = pts.shape
        surf2light = self.light_xyz.reshape(1, -1, 3) - pts.reshape(n_rays, 3)[:, None, :]
        surf2light = torch.nn.functional.normalize(surf2light, p=2, dim=-1, eps=1e-7)
        return surf2light
    
    @staticmethod
    def _get_vdir(cam_loc, pts):
        surf2cam = cam_loc - pts
        surf2cam = torch.nn.functional.normalize(surf2cam, p=2, dim=-1, eps=1e-7)
        return surf2cam.reshape(-1, 3) # Nx3
    
    #def forward(self, batch):
    #    #id_, hw, alpha, xyz, normal, lvis, light_map = batch
    #    
    #    acc_map         = batch['acc_map']      # (1, nray)
    #    depth_map       = batch['depth_map']    # (1, nray)
    #    density_map     = batch['density_map']  # (1, nray, 1)
    #    normal          = batch['normal_map']   # (1, nray, 3)
    #    instance_map    = batch['instance_map'] # (1, nray, nkeys)
    #    surf            = batch['surf']         # (1, nray, 3)
    #    lvis            = batch['lvis_hit']     # (1, nray, 512)
    #    keys            = batch['keys'] # list of keys
    #    sparse_features = batch['sparse_feature'] # dict of features per person
#
    #    xyz = surf
    #    
    #    surf2light = self._get_ldir(xyz)
#
    #    normal_pred = self._pred_normal_at(xyz)
    #    normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1, eps=1e-7)
#
    #    lvis_pred = self._pred_lvis_at(xyz, surf2light)
    #    # ------ Loss
    #    pred = {'normal': normal_pred, 'lvis': lvis_pred}
    #    gt = {'normal': normal.reshape(-1, 3), 'lvis': lvis.reshape(-1, 512), 'alpha': alpha}
    #    loss_kwargs = {}
    #    # ------ To visualize
    #    to_vis = {'id': id_, 'hw': hw}
    #    for k, v in pred.items():
    #        to_vis['pred_' + k] = v
    #    for k, v in gt.items():
    #        to_vis['gt_' + k] = v
#
    #    return pred, gt, loss_kwargs, to_vis
    
    
    @staticmethod
    def chunk_apply(func, x, dim, n_chunk):
        n = x.shape[0]
        ret = torch.zeros((n, dim)).to(x)
        for i in range(0, n, n_chunk):
            end_i = min(n, i + n_chunk)
            x_chunk = x[i:end_i]
            ret_chunk = func(x_chunk)
            ret[i:end_i] = ret_chunk
        return ret

    def _pred_normal_at(self, net, pts, features):
        eps = 1e-6
        mlp = net['normal_mlp']
        out = net['normal_out']
        scaled_pts = self.xyz_scale * pts
        
        def chunk_func(input):
            surf_embed = self.embedder["xyz"].embed(input).float()
            return surf_embed
        surf_embed = self.chunk_apply(chunk_func, scaled_pts.reshape(-1, 3), 63, self.mlp_chunk)
        #surf_embed = self.embedder['xyz'].embed(scaled_pts.reshape(-1, 3)).float()
        
        surf_embed = torch.cat([features, surf_embed], dim=-1)

        def chunk_func(surf):
            normals = out(mlp(surf))
            return normals
        
        normal = self.chunk_apply(chunk_func, surf_embed, 3, self.mlp_chunk)
        normal = normal + eps
        return normal

    def _pred_lvis_at(self, net, pts, surf2light, features=None):
        mlp = net['lvis_mlp']
        out = net['lvis_out']
        scaled_pts = self.xyz_scale * pts
        n_lights = surf2light.shape[1]
        surf2light_flat = surf2light.reshape(-1, 3) #NLx3
        surf_rep = scaled_pts.reshape(-1, 3)[:, None, :].repeat(1, n_lights, 1)
        surf_flat = surf_rep.reshape(-1, 3)
        feat = features[:, None, :].repeat(1, n_lights, 1).reshape(-1, 256)
        lvis_flat = torch.zeros(surf_flat.shape[0], 1).to(surf_flat)
        #for chunk in range(0, surf_flat.shape[0], self.mlp_chunk):
        #    print("chunk: ", chunk)
        #    surf_flat_chunk = surf_flat[chunk:chunk+self.mlp_chunk]
        #    surf2light_flat_chunk = surf2light_flat[chunk:chunk+self.mlp_chunk]
        #    feat_chunk = feat[chunk:chunk+self.mlp_chunk]
        #    
        #    mlp_in_chunk = torch.cat([feat_chunk, surf_flat_chunk, surf2light_flat_chunk], dim=-1)
        #    lvis_flat_chunk = out(mlp(mlp_in_chunk))
        #    lvis_flat[chunk:chunk+self.mlp_chunk] = lvis_flat_chunk
        #lvis = lvis_flat.reshape(scaled_pts.shape[1], n_lights)
        #return lvis
    
        surf_embed = self.embedder['xyz'].embed(surf_flat).float()
        
        surf2light_embed = self.embedder['ldir'].embed(surf2light_flat).float()
        
        mlp_input = torch.cat([feat, surf_embed, surf2light_embed], dim=-1)
        
        def chunk_func(input):
            lvis = out(mlp(input))
            return lvis
        lvis_flat = self.chunk_apply(chunk_func, mlp_input, 1, self.mlp_chunk)
        lvis = lvis_flat.reshape(scaled_pts.shape[1], n_lights)
        return lvis

    #def compute_loss(self, pred, gt, **kwargs):
    #    normal_loss_weight = self.normal_loss_weight
    #    lvis_loss_weight = self.lvis_loss_weight
#
    #    normal_pred, normal_gt = pred['normal'], gt['normal']
    #    lvis_pred, lvis_gt = pred['lvis'], gt['lvis']
#
    #    alpha_map = gt['alpha']
    #    mse = nn.MSELoss()
    #    normal_loss = mse(normal_gt, normal_pred)
    #    lvis_loss = mse(lvis_gt, lvis_pred)
    #    loss = normal_loss * normal_loss_weight + lvis_loss * lvis_loss_weight
    #    return loss, normal_loss, lvis_loss

    def light(self):
        # No negative light
        if self.fix_light:
            if self.gaussian_light:
                
                positions = torch.meshgrid(torch.arange(self.light_res[0]).cuda(), torch.arange(self.light_res[1]).cuda())
                xs, ys = positions
                xs = xs.float()
                ys = ys.float()
                
                dy = torch.abs(ys - self.pos[1])
                dy = torch.min(dy, self.light_res[1] - dy)
                
                distance = torch.sqrt((xs - self.pos[0])**2 + dy**2)
                gaussian = torch.exp(-distance / F.relu(self.sigma))
                light = F.relu(self.intens) * gaussian + F.relu(self.ambient)
                
                assert light.all() >= 0
                return light.unsqueeze(-1).repeat(1, 1, 3)
                
            else:
                return torch.ones(self.light_res + (3,)).to(self._light) + self._light.repeat(1,1,3)
        else:
            if self.perturb_light > 0:
                light_noise = torch.normal(mean=0, std=self._light.max().item() / self.perturb_light, size=self.light_res + (3,)).to(self._light)
            else:
                light_noise = 0.
            if self.achro_light:
                return torch.clip(self._light.repeat(1, 1, 3), min=0., max=1e6) + light_noise
            else:
                return torch.clip(self._light, min=0., max=1e6) + light_noise


    def forward(self, batch, result, mode='train', relight_olat=True, relight_probes=True, albedo_scale=None, albedo_override=None, brdf_z_override=None):
        
        xyz_jitter_std = self.xyz_jitter_std
        #id_, hw, rayo, _, rgb, alpha, xyz, normal, lvis, raw_batch = batch
        #print ("batch: ", batch.keys())
        
        alpha_map           = result['acc_map']      # (1, nray)
        depth_map       = result['depth_map']    # (1, nray)
        density_map     = result['density_map']  # (1, nray, 1)
        instance_map    = result['instance_map'] # (1, nray, nkeys)
        surf_map             = result['surf']         # (nray, 3)
        keys            = result['keys'] # list of keys
        latent_features = result['latent_features'] # dict of features per person
        #gt
        normal_gt = result['normal_map']   # (1, nray, 3)
        if mode == 'train':
            normal_map          = result['normal_map']   # (1, nray, 3)
            lvis_hit_map            = result['lvis_hit']     # (1, nray, nlights)
                    
        ray_o = batch['ray_o']
        
        
        
        
        ret_mask = torch.zeros_like(alpha_map, dtype=torch.bool) # (1, nray)
        
        pred_rgb = torch.zeros_like(surf_map)
        pred_normal = torch.zeros_like(surf_map)
        pred_lvis = torch.zeros((surf_map.shape[0], surf_map.shape[1], self.NLights), device=torch.device('cuda:0'))
        pred_albedo = torch.zeros_like(surf_map)
        pred_brdf = torch.zeros_like(surf_map)
        
        pred_normal_jitter = torch.zeros_like(pred_normal)
        pred_lvis_jitter = torch.zeros_like(pred_lvis)
        pred_albedo_jitter = torch.zeros_like(surf_map)
        pred_brdf_jitter = torch.zeros_like(surf_map)
        
        if self.do_relight_probes:
            pred_rgb_probes = torch.zeros((surf_map.shape[0], surf_map.shape[1], len(self.novel_probes), 3), device=torch.device('cuda:0'))
        if self.do_relight_olat:
            pred_olat = torch.zeros((surf_map.shape[0], surf_map.shape[1], len(self.novel_olat), 3), device=torch.device('cuda:0'))
        
        
        argmax_instances = torch.argmax(instance_map, dim=-1)
        for i, key in enumerate(keys):
            #print(i, " key: ", key)
            sparse_features = latent_features[key]
            
            if "human" in key:
                sp_input = sparse_features["sp_input"]
                feature_volume = sparse_features["feature_volume"]
                latent_time = sparse_features["latent_time"]
                
            elif "background" in key:
                #we don't want to relight the background
                continue
            elif "ground" in key:
                sparse_features = {"latent_time":sparse_features}
            
            mask = (argmax_instances == i) & (alpha_map > 0.5)
            #print("mask: ", mask.shape)
            if mask.sum() == 0:
                print("mask.sum() == 0 for key: ", key)
                continue
            
            # ///////////////////////
            """
            coords = batch['coord']
            import matplotlib.pyplot as plt
            blank = np.zeros((1080,1920)) 
            coords = coords[0].cpu().numpy()
            
            
            values = mask
            values = values[0].detach().cpu().numpy()
                       
            
            blank[coords[:,0],coords[:,1]] = values
            
            plt.imshow(blank)
            plt.show()
            """
            #////////////////////////
            
            
            
            #print("key: ", key)
            
            # Nx3 to 1xNx3
            xyz = surf_map[mask].reshape(1, -1, 3)
            rayo = ray_o[mask]#.reshape(1, -1, 3)
            
            ret_mask[mask] = True
            
            if self.training and mode == 'train' and xyz_jitter_std > 0.:
                xyz_noise = torch.normal(mean=0, std=xyz_jitter_std, size=xyz.shape).to(xyz)
                xyz_jittered = xyz + xyz_noise
            else:
                xyz_noise = None
            
            
            # get the feature volume from the pretrained model
            with torch.no_grad():
                nerf_net = self.net.model(key) 
                features , valid_mask = nerf_net.get_feature(xyz, sparse_features)
                #print('valid_mask: ', valid_mask.sum()/valid_mask.numel())
                if xyz_noise is not None:
                    features_jitter, valid_mask_jitter  = nerf_net.get_feature(xyz_jittered, sparse_features)
                    
            net = self.net_dict[key]
            
            
            features = net['latent_fc'](features).reshape(-1, 256)
            if xyz_noise is not None:
                features_jitter = net['latent_fc'](features_jitter).reshape(-1, 256)
            

            surf2light = self._get_ldir(xyz)
            
            surf2cam = self._get_vdir(rayo.float(), xyz)
            
            if self.predict_normals:
                normal_pred = self._pred_normal_at(net, xyz, features)
                if xyz_noise is None:
                    normal_jitter = None
                else:
                    normal_jitter = self._pred_normal_at(net, xyz_jittered, features_jitter)
                normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1, eps=1e-7)
                if normal_jitter is not None:
                    normal_jitter = torch.nn.functional.normalize(normal_jitter, p=2, dim=-1, eps=1e-7)
            else:
                
                normal_pred = normal_gt[mask]
                normal_jitter = normal_pred

            lvis_pred = torch.zeros((xyz.shape[1], self.NLights), device=torch.device('cuda:0'))
            
            # For demo, we chunk to avoid Out of Memory error
            chunk_size = 16384
            for chunk in range(0, xyz.shape[1], chunk_size):
                chunk_end = min(xyz.shape[1], chunk + chunk_size)                
                
                xyz_chunk = xyz[:, chunk:chunk_end]
                surf2light_chunk = surf2light[chunk:chunk_end]
                features_chunk = features[chunk:chunk_end]
                lvis_pred_chunk = self._pred_lvis_at(net, xyz_chunk, surf2light_chunk, features_chunk)
                lvis_pred[chunk:chunk_end] = lvis_pred_chunk
                
            #lvis_pred = self._pred_lvis_at(net, xyz, surf2light, features)
            
            # Jitter only for training, no need to chunk
            if xyz_noise is None:
                lvis_jitter = None
            else:
                lvis_jitter = self._pred_lvis_at(net, xyz_jittered, surf2light, features_jitter)

            # albedo
            
            if not self.predict_normals_lvis_only:
                albedo = self._pred_albedo_at(net, xyz, features)
                if xyz_noise is None:
                    albedo_jitter = None
                else:
                    albedo_jitter = self._pred_albedo_at(net, xyz_jittered, features_jitter)

                if True :
                    brdf_prop = self._pred_brdf_at(net, xyz, features)
                    if xyz_noise is None:
                        brdf_prop_jitter = None
                    else:
                        brdf_prop_jitter = self._pred_brdf_at(net, xyz_jittered, features_jitter)
                    if self.normalize_brdf_z:
                        brdf_prop = torch.nn.functional.normalize(brdf_prop, p=2, dim=-1, eps=1e-7)
                        if brdf_prop_jitter is not None:
                            brdf_prop_jitter = torch.nn.functional.normalize(brdf_prop_jitter, p=2, dim=-1, eps=1e-7)
                    brdf = self._eval_brdf_at(surf2light, surf2cam, normal_pred, albedo, brdf_prop) # NxLx3
                else:
                    brdf_prop = torch.zeros_like(albedo)
                    brdf_prop_jitter = torch.zeros_like(albedo)
                    brdf = albedo[ :, None, :].repeat(1, self.NLights, 1) / np.pi # NxLx3 Default Lambertian BRDF
                    ##print("eval_brdf_at")
            
            
                #print("_render")
                # rendering
                rgb_pred, rgb_olat, rgb_probes, hdr = self._render(lvis_pred, brdf, surf2light, normal_pred, 
                                                                relight_olat=self.do_relight_olat and mode != 'train', 
                                                                relight_probes=self.do_relight_probes and mode != 'train')

            
            
            pred_normal[mask] = normal_pred
            pred_lvis[mask] = lvis_pred
            if not self.predict_normals_lvis_only:
                pred_rgb[mask] = rgb_pred
                pred_albedo[mask] = albedo
                pred_brdf[mask] = brdf_prop
            
            if xyz_noise is not None:
                pred_normal_jitter[mask] = normal_jitter
                pred_lvis_jitter[mask] = lvis_jitter
                if not self.predict_normals_lvis_only:
                    pred_albedo_jitter[mask] = albedo_jitter
                    pred_brdf_jitter[mask] = brdf_prop_jitter
            
            if not self.predict_normals_lvis_only:
                #if rgb_olat is not None:
                #    pred['rgb_olat'] = rgb_olat
                if rgb_probes is not None:
                    pred_rgb_probes[mask] = rgb_probes 
                if rgb_olat is not None:
                    pred_olat[mask] = rgb_olat
        
        
        if self.predict_normals_lvis_only:
                    
            pred_normal = pred_normal[ret_mask]
            pred_lvis = pred_lvis[ret_mask]
            if xyz_noise is not None:
                pred_normal_jitter = pred_normal_jitter[ret_mask]
                pred_lvis_jitter = pred_lvis_jitter[ret_mask]
                
            normal = normal_gt[ret_mask]
            lvis = lvis_hit_map[ret_mask]
            
            #####################################
            #coords = batch['coord']
            #import matplotlib.pyplot as plt
            #blank = np.zeros((1080,1920,3)) 
            #coords = coords[ret_mask].detach().cpu().numpy()
            #
            #
            #values = (normal - pred_normal) 
            #values = values.detach().cpu().numpy()
            #print("values: ", values.shape)
            #values = np.linalg.norm(values, axis=-1)**2
            ##map 0 to green, 1 to red
            #values = values / values.max()
            #values = np.stack([values, 1-values, np.zeros_like(values)], axis=-1)
            #blank[coords[:,0],coords[:,1]] = values
            #
            #plt.imshow(blank)
            #plt.show()
            #####################################
            
            pred = {'normal': pred_normal, 'lvis': pred_lvis}
            gt = {'normal': normal, 'lvis': lvis, 'alpha': alpha_map[ret_mask]}
            loss_kwargs = {'mode': mode, 'normal_jitter': pred_normal_jitter, 'lvis_jitter': pred_lvis_jitter}
            return pred, gt, loss_kwargs
            
        
        if mode != 'train':
            light = self.light()
            #print("light.shape: ", light.shape)
            #print("light min, max, mean: ", light.min(), light.max(), light.mean())
            #plt.imshow(light.detach().cpu().numpy())
            #plt.show()
            
            
            ret_mask = ret_mask.unsqueeze(-1)
            pred_rgb = torch.where(ret_mask, pred_rgb, torch.zeros_like(pred_rgb))
            pred_normal = torch.where(ret_mask, pred_normal, torch.zeros_like(pred_normal))
            pred_lvis = torch.where(ret_mask, pred_lvis, torch.zeros_like(pred_lvis))
            pred_albedo = torch.where(ret_mask, pred_albedo, torch.zeros_like(pred_albedo))
            pred_brdf =  torch.where(ret_mask, pred_brdf, torch.zeros_like(pred_brdf))
            if rgb_probes is not None:                
                pred_rgb_probes = torch.where(ret_mask[..., None], pred_rgb_probes, torch.zeros_like(pred_rgb_probes))
            
            if rgb_olat is not None:
                pred_olat = torch.where(ret_mask[..., None], pred_olat, torch.zeros_like(pred_olat))
        
            return {'rgb_map': pred_rgb[None,:], 'normal_map': pred_normal[None,:], 'lvis_map': pred_lvis[None,:], 
                    'albedo_map': pred_albedo[None,:], 'brdf_map': pred_brdf[None,:], 
                    'rgb_probes': pred_rgb_probes[None,:], 'olat': pred_olat[None,:]}
        
        
        
        
            
        
        
        pred_rgb = pred_rgb[ret_mask]
        pred_normal = pred_normal[ret_mask]
        pred_lvis = pred_lvis[ret_mask]
        pred_albedo = pred_albedo[ret_mask]
        pred_brdf = pred_brdf[ret_mask]
        
        if xyz_noise is not None:
            pred_normal_jitter = pred_normal_jitter[ret_mask]
            pred_lvis_jitter = pred_lvis_jitter[ret_mask]
            pred_albedo_jitter = pred_albedo_jitter[ret_mask]
            pred_brdf_jitter = pred_brdf_jitter[ret_mask]
        
        rgb =batch['rgb']
        alpha = result['acc_map']
        normal = result['normal_map']
        lvis = result['lvis_hit']
        
        rgb = rgb[ret_mask]
        alpha = alpha[ret_mask]
        normal = normal[ret_mask]
        lvis = lvis[ret_mask]
        
        
        
        pred = {'rgb': pred_rgb, 'normal': pred_normal, 'lvis': pred_lvis, 'albedo': pred_albedo, 'brdf': pred_brdf}
        gt = {'rgb': rgb, 'normal': normal, 'lvis': lvis, 'alpha': alpha}
        loss_kwargs = {
            'mode': mode, 'normal_jitter': pred_normal_jitter,
            'lvis_jitter': pred_lvis_jitter, 'brdf_prop_jitter': pred_brdf_jitter,
            'albedo_jitter': pred_albedo_jitter}
            # ------ To visualize
        #to_vis = {'id': id_, 'hw': hw}
        #for k, v in pred.items():
        #    to_vis['pred_' + k] = v
        #for k, v in gt.items():
        #    to_vis['gt_' + k] = v
        return pred, gt, loss_kwargs#, to_vis

    def _pred_albedo_at(self, net, pts, features=None):
        albedo_scale = self.albedo_slope
        albedo_bias = self.albedo_bias
        mlp = net['albedo_mlp']
        out = net['albedo_out']
        embedder = self.embedder['xyz']
        scaled_pts = self.xyz_scale * pts
        surf_embed = embedder.embed(scaled_pts.reshape(-1, 3)).float()
        if features is not None:
            if self.use_xyz:
                surf_embed = torch.cat([features, surf_embed], dim=-1)
            else:
                surf_embed = features

        def chunk_func(surf):
            albedo = out(mlp(surf))
            return albedo
        
        albedo = self.chunk_apply(chunk_func, surf_embed, 3, self.mlp_chunk)
        albedo = albedo_scale * albedo + albedo_bias # [bias, scale + bias]
        return albedo # Nx3

    def _pred_brdf_at(self, net, pts, features=None):
        mlp = net['brdf_z_mlp']
        out = net['brdf_z_out']
        embedder = self.embedder['xyz']
        scaled_pts = self.xyz_scale * pts
        surf_embed = embedder.embed(scaled_pts.reshape(-1, 3)).float()
        if features is not None:
            if self.use_xyz:
                surf_embed = torch.cat([features, surf_embed], dim=-1)
            else:
                surf_embed = features

        def chunk_func(surf):
            brdf_z = out(mlp(surf))
            return brdf_z

        brdf_z = self.chunk_apply(chunk_func, surf_embed, self.z_dim, self.mlp_chunk)
        return brdf_z # NxZ

    def _render(
            self, light_vis, brdf, surf2light, normal,
            relight_olat=False, relight_probes=False):
        linear2srgb = self.linear2srgb
        light = self.light()

        lcos = torch.einsum('ijk,ik->ij', surf2light, normal)
        areas = self.lareas.reshape(1, -1, 1)
        front_lit = lcos > 0
        lvis = front_lit * light_vis

        hdr = None
        # hdr_contrib = brdf * lcos[:, :, None] * areas
        # hdr = torch.sum(hdr_contrib, dim=1)

        def integrate(light):
            light_flat = light.reshape(-1, 3)
            light = lvis[:, :, None] * light_flat[None, :, :] # NxLx3
            light_pix_contrib = brdf * light * lcos[:, :, None] * areas # NxLx3
            rgb = torch.sum(light_pix_contrib, dim=1) #Nx3
            rgb = torch.clip(rgb, 0., 1.)
            if linear2srgb:
                rgb = func_linear2srgb(rgb)
            return rgb
        
        rgb = integrate(light)
        # print('light', light)
        rgb_olat = None
        if relight_olat:
            rgb_olat = []
            for _, light in self.novel_olat.items():
                rgb_relit = integrate(light)
                rgb_olat.append(rgb_relit)
            rgb_olat = torch.cat([x[:, None, :] for x in rgb_olat], dim=1)

        rgb_probes = None
        if relight_probes:
            rgb_probes = []
            for _, light in self.novel_probes.items():
                rgb_relit = integrate(0.35*light)
                #rgb_relit = integrate(0.25*light + 0.1*self.light())

                rgb_probes.append(rgb_relit)
            rgb_probes = torch.cat([x[:, None, :] for x in rgb_probes], dim=1)
        return rgb, rgb_olat, rgb_probes, hdr # Nx3
    
    def compute_loss(self, pred, gt, **kwargs):
        if self.predict_normals_lvis_only:
            
            normal_pred = pred['normal']
            lvis_pred = pred['lvis']
            normal_gt = gt['normal']
            lvis_gt = gt['lvis']
            normal_jitter = kwargs.pop('normal_jitter')
            lvis_jitter = kwargs.pop('lvis_jitter')
            
            cos_sim_loss = lambda x, y: torch.mean(1 - F.cosine_similarity(x, y, dim=-1))
            mse = nn.MSELoss()
            smooth_loss = nn.L1Loss() if self.smooth_use_l1 else nn.MSELoss()
            loss = 0
            
            normal_loss = cos_sim_loss(normal_gt, normal_pred) * self.normal_loss_weight
            loss += normal_loss 
            
            lvis_loss = mse(lvis_gt, lvis_pred) * self.lvis_loss_weight
            loss += lvis_loss
            
            normal_smooth_loss = smooth_loss(normal_pred, normal_jitter) * self.normal_smooth_weight
            loss += normal_smooth_loss
            
            lvis_smooth_loss = smooth_loss(lvis_pred, lvis_jitter) * self.lvis_smooth_weight
            loss += lvis_smooth_loss
            
            return loss, normal_loss, lvis_loss, normal_smooth_loss, lvis_smooth_loss, 0, 0, 0, 0
            
            
        
            
        
        normal_loss_weight = self.normal_loss_weight
        lvis_loss_weight = self.lvis_loss_weight
        smooth_use_l1 = self.smooth_use_l1
        light_tv_weight = self.light_tv_weight
        light_achro_weight = self.light_achro_weight
        smooth_loss = nn.L1Loss() if smooth_use_l1 else nn.MSELoss()
        mode = kwargs.pop('mode')
        normal_jitter = kwargs.pop('normal_jitter')
        lvis_jitter = kwargs.pop('lvis_jitter')
        albedo_jitter = kwargs.pop('albedo_jitter')
        brdf_prop_jitter = kwargs.pop('brdf_prop_jitter')

        alpha, rgb_gt = gt['alpha'], gt['rgb']
        rgb_pred = pred['rgb']
        normal_pred, normal_gt = pred['normal'], gt['normal']
        lvis_pred, lvis_gt = pred['lvis'], gt['lvis']
        albedo_pred = pred['albedo']
        brdf_prop_pred = pred.get('brdf', None)
        #hdr = pred['hdr']

        # RGB recon. loss is always here
        loss = 0
        mse = nn.MSELoss()
        rgb_loss = mse(rgb_gt.reshape(-1, 3), rgb_pred) * 10.
        loss = loss + rgb_loss # N
        # If validation, just MSE -- return immediately
        if mode == 'vali':
            return loss
        # If we modify the geometry
        normal_loss = torch.zeros(1, device=torch.device('cuda:0'))
        lvis_loss = torch.zeros(1, device=torch.device('cuda:0'))
        normal_smooth_loss = torch.zeros(1, device=torch.device('cuda:0'))
        lvis_smooth_loss = torch.zeros(1, device=torch.device('cuda:0'))
        albedo_smooth_loss = torch.zeros(1, device=torch.device('cuda:0'))
        brdf_smooth_loss = torch.zeros(1, device=torch.device('cuda:0'))
        albedo_entropy = torch.zeros(1, device=torch.device('cuda:0'))
        if self.shape_mode in ('scratch', 'finetune'):
            # Predicted values should be close to initial values
            normal_loss = mse(normal_gt.reshape(-1, 3), normal_pred) # N
            lvis_loss = mse(lvis_gt.reshape(-1, lvis_pred.shape[-1]), lvis_pred) # N
            if self.shape_mode in ('scratch'):
                assert self.use_shape_sup
            if self.use_shape_sup:
                loss += normal_loss_weight * normal_loss
                loss += lvis_loss_weight * lvis_loss
            # Predicted values should be smooth
            if normal_jitter is not None:
                normal_smooth_loss = smooth_loss(normal_pred, normal_jitter) # N
                loss += self.normal_smooth_weight * normal_smooth_loss
            if lvis_jitter is not None:
                lvis_smooth_loss = smooth_loss(lvis_pred, lvis_jitter) # N
                loss += self.lvis_smooth_weight * lvis_smooth_loss
        # Albedo should be smooth
        if albedo_jitter is not None:
            albedo_smooth_loss = smooth_loss(albedo_pred, albedo_jitter) # N
            loss += self.albedo_smooth_weight * albedo_smooth_loss
        # BRDF property should be smooth
        if brdf_prop_jitter is not None:
            brdf_smooth_loss = smooth_loss(brdf_prop_pred, brdf_prop_jitter) # N
            loss += self.brdf_smooth_weight * brdf_smooth_loss

        # Light should be smooth
        #if mode == 'train':
            #light = self.light()
            ## Spatial TV penalty
            #if light_tv_weight > 0:
            #    dx = light - torch.roll(light, 1, 1)
            #    dy = light - torch.roll(light, 1, 0)
            #    tv = torch.sum(dx ** 2 + dy ** 2)
            #    loss += light_tv_weight * tv
            ## Cross-channel TV penalty
            #if light_achro_weight > 0:
            #    dc = light - torch.roll(light, 1, 2)
            #    tv = torch.sum(dc ** 2)
            #    loss += light_achro_weight * tv

        
        if self.albedo_sparsity > 0:
            albedo_entropy = 0
            for i in range(3):
                channel = albedo_pred[..., i]
                hist = GaussianHistogram(15, 0., 1., sigma=torch.var(channel))
                h = hist(channel)
                if h.sum() > 1e-6:
                    h = h.div(h.sum()) + 1e-6
                else:
                    h = torch.ones_like(h).to(h)
                albedo_entropy += torch.sum(-h*torch.log(h))
            loss += self.albedo_sparsity * albedo_entropy
        else :
            albedo_entropy = torch.zeros(1, device=torch.device('cuda:0'))
            

        return loss, normal_loss_weight*normal_loss, lvis_loss_weight*lvis_loss, self.normal_smooth_weight*normal_smooth_loss, self.lvis_smooth_weight*lvis_smooth_loss, self.albedo_smooth_weight*albedo_smooth_loss, self.brdf_smooth_weight*brdf_smooth_loss, rgb_loss, albedo_entropy

class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device=torch.device('cuda:0')).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, widths, act=None, skip_at=None):
        super(MLP, self).__init__()
        depth = len(widths)
        self.input_dim = input_dim

        if act is None:
            act = [None] * depth
        assert len(act) == depth
        self.layers = nn.ModuleList()
        self.activ = None
        prev_w = self.input_dim
        i = 0
        for w, a in zip(widths, act):
            if isinstance(a, str):
                if a == 'relu':
                    self.activ = nn.ReLU()
                elif a == 'softplus':
                    self.activ = nn.Softplus()
                elif a == 'sigmoid':
                    self.activ = nn.Sigmoid()
                else:
                    raise NotImplementedError
            layer = nn.Linear(prev_w, w)
            prev_w = w
            if skip_at and i in skip_at:
                prev_w += input_dim
            self.layers.append(layer)
            i += 1
        self.skip_at = skip_at

    def forward(self, x):
        x_ = x + 0
        for i, layer in enumerate(self.layers):
            # print(i)
            # print(x_.shape)
            if self.activ:
                y = self.activ(layer(x_))
            else:
                y = layer(x_)
            if self.skip_at and i in self.skip_at:
                y = torch.cat((y, x), dim=-1)
            x_ = y
            # print(y.shape)
        return y
