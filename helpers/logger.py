"""Wrapper class for logging into the TensorBoard, comet.ml and wandb"""
__author__ = 'Atomicoo'
__all__ = ['Logger']

import os
from tensorboardX import SummaryWriter
try:
    import wandb
except ImportError:
    wandb = None


class Logger(object):

    def __init__(self, logdir, experiment, model_name, wandb_info=None):
        self.model_name = model_name
        self.project_name = "%s-%s" % (experiment, self.model_name)
        self.logdir = os.path.join(logdir, self.project_name)
        self.writer = SummaryWriter(log_dir=self.logdir)
        self.wandb = None if wandb_info is None else wandb
        if self.wandb and self.wandb.run is None:
            self.wandb.init(**wandb_info)

    def log_model(self, model):
        self.writer.add_graph(model)
        if self.wandb is not None:
            self.wandb.watch(model)

    def log_step(self, phase, step, scalar_dict, figure_dict=None):
        if phase == 'train':
            if step % 2 == 0:
                # self.writer.add_scalar('lr', get_lr(), step)
                # self.writer.add_scalar('%s-step/loss' % phase, loss, step)
                for key in sorted(scalar_dict):
                    self.writer.add_scalar(f"{phase}-step/{key}", scalar_dict[key], step)
                if self.wandb is not None:
                    self.wandb.log(scalar_dict)

            if step % 10 == 0:
                for key in sorted(figure_dict):
                    self.writer.add_figure(f"{self.model_name}/{key}", figure_dict[key], step)
                if self.wandb is not None:
                    self.wandb.log({k: self.wandb.Image(v) for k,v in figure_dict.items()})

    def log_epoch(self, phase, epoch, scalar_dict):
        for key in sorted(scalar_dict):
            self.writer.add_scalar(f"{phase}/{key}", scalar_dict[key], epoch)
        if self.wandb is not None:
            self.wandb.log(scalar_dict)
