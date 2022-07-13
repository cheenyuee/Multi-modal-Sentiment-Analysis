from torch import nn
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights


class ImgModule(nn.Module):
    def __init__(self, args):
        super(ImgModule, self).__init__()
        self.encoder = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x = self.net(x)
        # x = x.transpose(1, 2)
        # torch.Size([N, L, E])
        # N:bach大小, L:序列长度, E:特征维度
        x = self.encoder(x)
        return x
