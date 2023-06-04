import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MoCo(nn.Module):
    def __init__(
        self, dim=512, hidden_dim=2048, weight_file="web/model.pt", device="cuda"
    ):
        super(MoCo, self).__init__()
        # Create layers and train from scratch
        self.layers = timm.create_model("resnet34", num_classes=dim, pretrained=True)
        dim_mlp = self.layers.fc.weight.shape[1]
        self.layers.fc = nn.Sequential(
            nn.Linear(dim_mlp, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False),
        )
        self.layers.load_state_dict(torch.load(weight_file))
        # self.layers.fc = nn.Identity()
        self.to(device)

    def forward(self, x):
        return F.normalize(self.layers(x), dim=1)
