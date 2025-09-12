from base import *
from models.vgg16 import VGG16
from utils import interpret_new
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import nn
import torch

class MultiModel(nn.Module):
    def __init__(self, args):
        super(MultiModel, self).__init__()
        self.E = VGG16()
        self.D = MMDecoder(self.E.full_features,
                           out_channel=1,
                           out_size=(int(args['Isize']), int(args['Isize'])),
                           is_blip=False)


    def forward(self, image, z_text, text_pos, clip_model):
        z_image = self.E(image)
        image_224 = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=True)
        z = interpret_new(image_224.detach(), text_pos.detach(), clip_model, device).detach().clone().float()
        z = F.interpolate(z, size=(38,38), mode="bilinear", align_corners=True)
        min_low = z_image.min()
        max_low = z_image.max()
        a = 0.5
        power_scaled_z = torch.pow(z, a)
        z = (power_scaled_z - min_low) / (max_low - min_low) + min_low
        zrn = z * z_image
        mask, heatmap, afa = self.D(zrn, z_text)
        return mask, afa,heatmap

class MMDecoder(nn.Module):
    def __init__(self, full_features, out_channel, out_size, is_blip=False):
        super(MMDecoder, self).__init__()
        self.gap_fc = nn.Linear(512, 1, bias=False)
        self.relu = nn.ReLU(True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)
        if is_blip:
            self.bottleneck = BottleneckBlock(full_features[4], 256)
            self.up0 = UpBlock(256, full_features[3],
                               func='relu', drop=0).cuda()
            self.up1 = UpBlock(full_features[3], out_channel,
                               func='None', drop=0).cuda()
        else:
            self.bottleneck = BottleneckBlock(full_features[4], 512)
            self.up0 = UpBlock(512, full_features[3],
                               func='relu', drop=0).cuda()
            self.up1 = UpBlock(full_features[3], out_channel,
                               func='None', drop=0).cuda()
        self.out_size = out_size

    def forward(self, z, z_text):
        zz = self.bottleneck(z)
        afa = torch.sum(zz, dim=1, keepdim=True)
        afa = (afa - afa.min()) / (afa.max() - afa.min())
        zz_norm = zz / zz.norm(dim=1).unsqueeze(dim=1)
        attn_map = (zz_norm * z_text.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdims=True)
        zzr = zz * attn_map
        heatmap = torch.sum(zzr, dim=1, keepdim=True)
        zz = self.up0(zz)
        zz = self.up1(zz)
        zz = F.interpolate(zz, size=self.out_size, mode="bilinear", align_corners=True)
        pooled_feature = self.avg_pool(zzr).view(zzr.size(0), -1)
        fc1_out = self.relu(self.fc1(pooled_feature))
        counterfactual_prediction = F.softmax(self.fc2(fc1_out), dim=1)
        return F.sigmoid(zz),heatmap, counterfactual_prediction



