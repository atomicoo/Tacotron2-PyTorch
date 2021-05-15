from torch import nn


class AverageMeter:
    """Keeps track of most recent, average, sum, and count of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        spec_target, gate_target = targets[0], targets[1]
        spec_target.requires_grad_(False)
        gate_target.requires_grad_(False)
        gate_target = gate_target.view(-1, 1)

        spec_out, spec_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        spec_loss = nn.MSELoss()(spec_out, spec_target) + \
                    nn.MSELoss()(spec_out_postnet, spec_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return spec_loss + gate_loss
