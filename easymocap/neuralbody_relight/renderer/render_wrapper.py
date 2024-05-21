'''
  @ Date: 2021-09-05 20:24:16
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-09-05 21:25:08
  @ FilePath: /EasyMocap/easymocap/neuralbody/renderer/render_wrapper.py
'''
import torch
import torch.nn as nn
from ...config import load_object
from .render_relight import RelightModule

# train_renderer
class RenderWrapper(nn.Module):
    def __init__(self, net, renderer_module, renderer_args, loss, loss_reg={}):
        super().__init__()
        renderer_args = dict(renderer_args)
        renderer_args['net'] = net
        self.renderer = load_object(renderer_module, renderer_args)
        self.weights = {key:val['weight'] for key, val in loss.items()}
        self.weights.update({key:val['weight'] for key, val in loss_reg.items()})
        loss = {key:load_object(val.module, val.args) for key, val in loss.items()}
        loss_reg = {key:load_object(val.module, val.args) for key, val in loss_reg.items()}
        self.loss = nn.ModuleDict(loss)
        self.loss_reg = nn.ModuleDict(loss_reg)
        

    def forward(self, batch):
        pred, gt, loss_kwargs = self.renderer(batch)
        #print(ret.keys())
        
        loss_sum, normal_loss, lvis_loss, normal_smooth_loss, lvis_smooth_loss, albedo_smooth_loss, brdf_smooth_loss, rgb_loss, albedo_entropy = self.renderer.compute_loss(pred, gt, **loss_kwargs)
        
        scalar_stats = {
            'loss': loss_sum,
            'normal_loss': normal_loss,
            'lvis_loss': lvis_loss,
            'normal_smooth_loss': normal_smooth_loss,
            'lvis_smooth_loss': lvis_smooth_loss,
            'albedo_smooth_loss': albedo_smooth_loss,
            'brdf_smooth_loss': brdf_smooth_loss,
            'rgb_loss': rgb_loss,
            'albedo_entropy': albedo_entropy,
        }
        
        image_stats = {}

        return pred, loss_sum, scalar_stats, image_stats