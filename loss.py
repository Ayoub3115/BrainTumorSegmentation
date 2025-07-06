import torch
import torch.nn as nn
class DiceLossBinary(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super(DiceLossBinary, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class ComboLossBinary(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        super(ComboLossBinary, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLossBinary()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets.float())
        return self.alpha * dice + self.beta * bce