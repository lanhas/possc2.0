import torch.nn as nn

class BpNet(nn.Module):
    def __init__(self, input_shape, output_shape) -> None:
        super(BpNet, self).__init__()

        def block(in_feat, out_feat, normalize=True, drop=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            if drop:
                layers.append(nn.Dropout(0.2))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            *block(input_shape, 256, drop=False),
            *block(256, 128, drop=False),
            nn.Linear(128, output_shape)
        )

    def forward(self, x):
        output = self.model(x)
        return output