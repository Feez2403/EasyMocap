# Training code based on PyTorch-Lightning
import os

from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim.optimizer import Optimizer
# To run on windows, need to set PL_TORCH_DISTRIBUTED_BACKEND=gloo
# on linux with NCCL just remove this line
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
from os.path import join
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from easymocap.mytools.debug_utils import myerror
import torch
from easymocap.config import load_object, Config
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from easymocap.neuralbody_relight.renderer.render_wrapper import RenderWrapper
from easymocap.neuralbody_relight.model.compose import ComposedModel
from easymocap.neuralbody_relight.dataset.mvbase import BaseDataset

# https://github.com/Project-MONAI/MONAI/issues/701
#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

#trained_checkpoint = join("neuralbody","wildtrack_remapped","model","last.ckpt")
trained_checkpoint = join("neuralbody","soccer1_6_125","model","last.ckpt")

class plwrapper(pl.LightningModule):
    def __init__(self, cfg, mode='train'):
        super().__init__()
        # load model
        self.cfg = cfg
        self.network = ComposedModel(**cfg.network_args)
        load_ckpt(self.network, trained_checkpoint)
        trainer_args = dict(cfg.trainer_args)
        trainer_args['net'] = self.network
        self.train_renderer = RenderWrapper(**trainer_args)
        if mode == 'train' or mode == 'trainvis':
            self.train_dataset =  BaseDataset(**cfg.data_train_args)
        # self.val_dataset = load_object(cfg.data_val_module, cfg.data_val_args)
        else:
            if mode + '_renderer_module' in cfg.keys():
                module, args = cfg[mode+'_renderer_module'], cfg[mode+'_renderer_args']
            else:
                module, args = cfg.renderer_module, cfg.renderer_args
            self.test_renderer = load_object(module, args, net=self.network)
        if mode + '_visualizer_module' in cfg.keys():
            module, args = cfg[mode+'_visualizer_module'], cfg[mode+'_visualizer_args']
        else:
            module, args = cfg.visualizer_module, cfg.visualizer_args
        self.visualizer = load_object(module, args)
        
        #for param in self.network.parameters():
            #param.requires_grad = False
            
        #for param in self.train_renderer.renderer.net.parameters():
            #param.requires_grad = False
        #print("weight human0: ", self.network.state_dict())

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        self.network.train()
        self.test_renderer.train()
        self.test_renderer.relight.net.train()
        torch.set_grad_enabled(True)
        batch['step'] = self.trainer.global_step
        batch['meta']['step'] = self.trainer.global_step
        output = self.test_renderer(batch)
        self.visualizer(output, batch)
        return 0

    def training_step(self, batch, batch_idx):
        batch['step'] = self.trainer.global_step
        batch['meta']['step'] = self.trainer.global_step
        # training_step defines the train loop. It is independent of forward
        output, loss, loss_stats, image_stats = self.train_renderer(batch)
        for key, val in loss_stats.items():
            self.log(key, val, batch_size=1)
        return loss

    def train_dataloader(self):
        from easymocap.neuralbody.trainer.dataloader import make_data_sampler, make_batch_data_sampler, make_collator, worker_init_fn
        shuffle = True
        is_distributed = len(self.cfg.gpus) > 1
        is_train = True
        sampler = make_data_sampler(self.cfg, self.train_dataset, shuffle, is_distributed, is_train)
        batch_size = self.cfg.train.batch_size
        drop_last = False
        max_iter = self.cfg.train.ep_iter

        self.batch_sampler = make_batch_data_sampler(self.cfg, sampler, batch_size,
                                            drop_last, max_iter, is_train)
        num_workers = self.cfg.train.num_workers
        collator = make_collator(self.cfg, is_train)
        data_loader = torch.utils.data.DataLoader(self.train_dataset,
                                              batch_sampler=self.batch_sampler,
                                              num_workers=num_workers,
                                              collate_fn=collator,
                                              worker_init_fn=worker_init_fn)
        return data_loader
    
    #def optimizer_step(
    #    self,
    #    *args,**kwargs
    #):
    #    for param in self.network.parameters():            
    #        param.grad = None
    #        
    #    super().optimizer_step(*args, **kwargs)

    def configure_optimizers(self):
        from easymocap.neuralbody.trainer.optimizer import Optimizer
        from easymocap.neuralbody.trainer.lr_sheduler import Scheduler, set_lr_scheduler
        optimizer = Optimizer(self.train_renderer, cfg.optimizer)
        scheduler = Scheduler(cfg.scheduler, optimizer)
        return [optimizer], [scheduler]
    
    def on_train_epoch_end(self):
        if len(cfg.gpus) > 1:
            self.batch_sampler.sampler.set_epoch(self.current_epoch)

def train(cfg):
    model = plwrapper(cfg)
    if cfg.resume and os.path.exists(trained_checkpoint):
        print('Resume multi-neuralbody from {}'.format(join(cfg.trained_model_dir, 'last.ckpt')))
        trained_no_relight_dir = trained_checkpoint
        resume_trainer_from_checkpoint = join(cfg.trained_model_dir, 'last.ckpt')
        ckpt_epoch = load_ckpt(model.network, trained_no_relight_dir, model_name='network')
        
        for n, param in model.network.named_parameters():
            print(n)
            param.requires_grad = False
        model.train_renderer.renderer.net = model.network
        model.train_renderer.renderer.relight.net = model.network
        if os.path.exists(resume_trainer_from_checkpoint):
            print('Resume relight trainer from {}'.format(resume_trainer_from_checkpoint))
            ckpt_epoch = load_ckpt(model.train_renderer, resume_trainer_from_checkpoint, model_name='train_renderer')
            
        else:
            print('Resume relight trainer from scratch using pretrained features for the object features network')
            #load_ckpt(model.train_renderer.renderer.net, trained_no_relight_dir, model_name='network')
        
        
            
            
    else:
        raise NotImplementedError("Please provide a pretrained model to resume from")
        print('Start from scratch')
        resume_from_checkpoint = None
        if os.path.exists(cfg.recorder_args.log_dir):
            # os.removedirs(cfg.recorder_args.log_dir)
            pass
        os.makedirs(cfg.recorder_args.log_dir, exist_ok=True)
        print(cfg, file=open(join(cfg.recorder_args.log_dir, 'exp.yml'), 'w'))
        ckpt_epoch = -1
    logger = TensorBoardLogger(save_dir=cfg.recorder_args.log_dir, name=cfg.exp)
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        verbose=True,
        dirpath=cfg.trained_model_dir,
        every_n_epochs=5 if not args.debug else 1,
        save_last=True,
        save_top_k=-1,
        monitor='loss',
        filename="{epoch}")
    # Log true learning rate, serves as LR-Scheduler callback
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    extra_args = {
        # 'num_nodes': len(cfg.gpus),
        'accelerator': 'gpu', 
    }
    if len(cfg.gpus) > 0:
        extra_args['strategy'] = 'ddp'
        extra_args['replace_sampler_ddp'] = False
    trainer = pl.Trainer(
        gpus=len(cfg.gpus), 
        logger=logger,
        #resume_from_checkpoint=resume_from_checkpoint,
        callbacks=[ckpt_callback, lr_monitor],
        max_epochs=cfg.train.epoch,
        # profiler='simple',
        **extra_args
    )
    print("Current epoch: {}".format(ckpt_epoch))
    trainer.fit_loop.epoch_progress.current.completed = ckpt_epoch
    trainer.fit(model)

def load_ckpt(model, ckpt_path, model_name='network'):
    print('Load from {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    checkpoint_ = {}
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in []:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    model.load_state_dict(checkpoint_, strict=False)
    return epoch

def test(cfg):
    from glob import glob
    from os.path import join
    from tqdm import tqdm
    model = plwrapper(cfg, mode=cfg.split)

    ckptpath = join(cfg.trained_model_dir, 'last.ckpt')
    if os.path.exists(ckptpath):
        epoch = load_ckpt(model.network, ckptpath, model_name='network')
        load_ckpt(model.test_renderer, ckptpath, model_name='train_renderer.renderer')
        for param in model.network.parameters():
            param.requires_grad = False
        for param in model.train_renderer.renderer.net.parameters():
            param.requires_grad = False
        
        model.train_renderer.renderer.net = model.network
        model.train_renderer.renderer.relight.net = model.network
    else:
        myerror('{} not exists'.format(ckptpath))
        epoch = -1
    model.step = epoch * 1000      
            
    if cfg['output'] == 'none':
        vis_out_dir = join('neuralbody', cfg.exp, cfg.split + '_{}'.format(epoch))
    else:
        vis_out_dir = join('neuralbody', cfg.exp, cfg.output)
    model.visualizer.data_dir = vis_out_dir
    model.visualizer.subs = cfg.data_val_args.subs

    if cfg.split == 'test':
        dataset = load_object(cfg.data_val_module, cfg.data_val_args)
    elif cfg.split in ['demo']:
        dataset = load_object(cfg['data_{}_module'.format(cfg.split)], cfg['data_{}_args'.format(cfg.split)])
        import json
        print('Save cameras to {}'.format(vis_out_dir))
        os.makedirs(vis_out_dir, exist_ok=True)
        with open(join(vis_out_dir,'cameras.json'), "w") as f:
            ouput_dict = {}
            for i in range(len(dataset)):
                index = dataset.infos[i]['index']
                camera = dataset.infos[i]['camera']
                ouput_dict[index] = {v : camera[v].tolist() for v in camera}
            json.dump(ouput_dict, f, indent=4)                  
                

    dataloader = torch.utils.data.DataLoader(dataset, 
        batch_size=1, num_workers=cfg.test.num_workers)
    
    extra_args = {
        'accelerator': 'gpu', 
    }
    if len(cfg.gpus) > 1:
        extra_args['strategy'] = 'ddp'

    trainer = pl.Trainer(
        gpus=len(cfg.gpus), 
        max_epochs=cfg.train.epoch,
        **extra_args
    )
    preds = trainer.predict(model, dataloader, return_predictions=False)

def parse(args, cfg):
    from os.path import join
    cfg.recorder_args.local_rank = cfg.local_rank
    if not args.slurm:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    assert cfg.exp != "", "Please set the experiement name"
    
    cfg.relight = "relight" in cfg.exp
    print(cfg.relight, cfg.exp)
    
    cfg.trained_model_dir = join('neuralbody', cfg.exp, 'model')
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    
    cfg.recorder_args.log_dir = join('neuralbody', cfg.exp, 'record')
    os.makedirs(cfg.recorder_args.log_dir, exist_ok=True)
    exp = 'vis'
    if 'keyframe' in cfg.data_val_args:
        exp += '_{}'.format(cfg.data_val_args.keyframe)
    if 'pid' in cfg.data_val_args:
        exp += '_{}'.format(cfg.data_val_args.pid)
    if 'pids' in cfg.data_val_args:
        exp += '_{}'.format('+'.join(list(map(str, cfg.data_val_args.pids))))
    if cfg.split == 'eval':
        if 'camnf' not in cfg.visualizer_args.format:
            cfg.visualizer_args.format = 'camnf'
        cfg.visualizer_args.concat = 'none'
        cfg.visualizer_args['keys'] = list(cfg.visualizer_args['keys']) + ['rgb', 'instance_map'] + ['raw_depth']
        assert len(cfg.data_val_args.subs) > 0, cfg.data_val_args.subs
        cfg.visualizer_args['subs'] = cfg.data_val_args.subs

if __name__ == "__main__":
    usage = '''This is the training script for Neuralbody'''
    args, cfg = Config.load_args(usage=usage)
    parse(args, cfg)
    if cfg.fix_random:
        seed_everything(666)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if cfg.split == 'train':
        train(cfg)
    elif cfg.split in ['test', 'demo', 'eval', 'trainvis', 'canonical', 'novelposes']:
        test(cfg)
