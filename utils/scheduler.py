import math
import torch

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, after_scheduler):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * float(self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        if not self.finished:
            self.after_scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.finished = True
        # Fix: For ReduceLROnPlateau, get lrs from optimizer param_groups
        if hasattr(self.after_scheduler, 'get_last_lr'):
            return self.after_scheduler.get_last_lr()
        else:
            return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metrics=None):
        if self.last_epoch < self.warmup_epochs:
            super().step()
        else:
            if hasattr(self.after_scheduler, 'step'):
                if metrics is not None:
                    self.after_scheduler.step(metrics)
                else:
                    self.after_scheduler.step()
            self._step_count += 1