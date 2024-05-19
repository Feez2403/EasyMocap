import torch
import torch.nn as nn
from .nerf import Nerf, EmbedMLP, MultiLinear
from os.path import join
from ...mytools.file_utils import read_json
import numpy as np

def create_dynamic_embedding(mode, embed):
    if mode == 'dense':
        embedding = nn.Embedding(embed.shape[0], embed.shape[1])
    elif mode == 'mlp':
        if 'D' not in embed.keys():
            embedding = EmbedMLP(
                input_ch=1,
                multi_res=32,
                W=128,
                D=2,
                bounds=embed.shape[0],
                output_ch=embed.shape[1])
        else:
            embedding = EmbedMLP(
                input_ch=1,
                multi_res=32,
                W=embed.W,
                D=embed.D,
                bounds=embed.shape[0],
                output_ch=embed.shape[1])
    else:
        raise NotImplementedError
    return embedding

class NeRFT(Nerf):
    def __init__(self, embed, nerf):
        nerf['latent'] = {'time': embed.shape[1]}
        super().__init__(**nerf)
        self.mode = embed.mode
        self.embedding = create_dynamic_embedding(self.mode, embed)
        self.cache = {}
    
    def clear_cache(self):
        self.cache = {}

    def before(self, batch, name):
        data = super().before(batch, name)
        nf, nv = batch['meta']['time'][0], batch['meta']['nview'][0]
        if 'frame' in name:
            nf = nf + batch[name+'_frame'] - batch['meta']['nframe']
        self.cache['embed'] = self.embedding(nf)
        return data

    def calculate_density(self, wpts, **kwargs):
        latents = {'time': self.cache['embed']}
        return super().calculate_density(wpts, latents, **kwargs)
    
