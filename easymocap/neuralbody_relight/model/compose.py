from ...config.baseconfig import load_object
import torch
import torch.nn as nn
from copy import deepcopy

class ComposedModel(nn.Module):
    def __init__(self, models) -> None:
        super().__init__()
        models = deepcopy(models)
        for key in ['human', 'ball']:
            if 'all' + key in models.keys():
                pids = models['all'+key].pop('pids')
                for pid in pids:
                    models['{}_{}'.format(key, pid)] = deepcopy(models['all'+key])
                    if 'pid' in models['{}_{}'.format(key, pid)].network_args.keys():
                        models['{}_{}'.format(key, pid)].network_args.pid = pid
                models.pop('all'+key)
        if 'allkeys' in models.keys():
            object_keys = models['allkeys'].pop('keys')
            for key in object_keys:
                models[key] = deepcopy(models['allkeys'])
            models.pop('allkeys')
        modules = {}
        for key, val in models.items():
            model = load_object(val['network_module'], val['network_args'])
            print('[model] {:15s}: {:4.1f}M'.format(key, sum([m.numel() for m in model.parameters()])/1000000))
            modules[key] = model
        self.models = nn.ModuleDict(modules)
        self.keys = list(self.models.keys())
        self.is_share = False
    
    def model(self, name):
        model = self.models[name]
        model.current = name
        return model

    def forward(self, pts):
        raise NotImplementedError

