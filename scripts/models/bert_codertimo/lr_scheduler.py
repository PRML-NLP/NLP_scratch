import numpy as np
from torch import optim


class LearningRateScheduler:
    def __init__(self, optim: optim.Optimizer, d_model, warmup_step: int):
        self.optim = optim
        self.warmup_step = warmup_step
        self.current_step = 0
        self.init_lr = np.power(d_model, -0.5)

    def update_lr_and_step_optim(self):
        # change next learning rate value
        self.update_lr()
        # update the parameters with computed gradient
        self.optim.step()

    def initialize_optim(self):
        # initialize gradient tensor with zero
        self.optim.zero_grad()

    def get_lr_scaler(self):
        # get learning rate scaler with exponentially decreasing function after warm-up step
        return np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.warmup_step, -1.5) * self.current_step,
            ]
        )

    def update_lr(self):
        self.current_step += 1
        # get the new learning rate value with computed scaler
        lr = self.init_lr * self.get_lr_scaler()
        # replace previous learning rate to new learning rate in optim config setting
        for param_group in self.optim.param_groups:
            param_group["lr"] = lr
