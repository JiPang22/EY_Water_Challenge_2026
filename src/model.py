import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 유틸리티 및 기본 블록 정의
def conv_bn_act(in_c, out_c, kernel_size=3, stride=1, padding=None, groups=1):
    if padding is None:
        padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
        nn.SiLU()
    )

class MV2Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expansion=2):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_c * expansion)
        self.use_res_connect = (self.stride == 1 and in_c == out_c)
        self.conv = nn.Sequential(
            conv_bn_act(in_c, hidden_dim, kernel_size=1),
            conv_bn_act(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        if self.use_res_connect: return x + self.conv(x)
        return self.conv(x)

class MobileViTBlock(nn.Module):
    def __init__(self, in_c, out_c, d_model, depth, kernel_size=3, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.local_rep = nn.Sequential(
            conv_bn_act(in_c, in_c, kernel_size=kernel_size),
            nn.Conv2d(in_c, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*2, dropout=0.1, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fusion = nn.Sequential(
            conv_bn_act(d_model + in_c, in_c, kernel_size=kernel_size),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        res = x
        x = self.local_rep(x)
        _, c, h, w = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(x.size(0), c, -1, self.patch_size**2).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, c)
        x = self.transformer(x)
        x = x.view(x.size(0), -1, self.patch_size**2, c).permute(0, 3, 1, 2).contiguous().view(x.size(0), c, h, w)
        x = torch.cat([res, x], dim=1)
        return self.fusion(x)

# 2. 이미지 특징 추출용 백본 (MobileViT_XXS 구조)
class MobileViT_Backbone(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.conv1 = conv_bn_act(in_channels, 16, stride=2)
        self.mv2_1 = MV2Block(16, 16, stride=1, expansion=2)
        self.mv2_2 = MV2Block(16, 24, stride=2, expansion=2)
        self.mv2_3 = MV2Block(24, 24, stride=1, expansion=2)
        self.mvit_1 = MobileViTBlock(24, 48, d_model=64, depth=2)
        self.mv2_4 = MV2Block(48, 64, stride=2, expansion=2)
        self.mvit_2 = MobileViTBlock(64, 80, d_model=80, depth=4)
        self.mv2_5 = MV2Block(80, 160, stride=2, expansion=2)
        self.mvit_3 = MobileViTBlock(160, 160, d_model=144, depth=3)
        self.conv_last = conv_bn_act(160, 320, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)   # 16
        x = self.mv2_1(x)
        x = self.mv2_2(x)   # 24
        x = self.mv2_3(x)
        x = self.mvit_1(x)  # 48
        x = self.mv2_4(x)   # 64
        x = self.mvit_2(x)  # 80
        x = self.mv2_5(x)   # 160
        x = self.mvit_3(x)
        x = self.conv_last(x) # 320
        return self.pool(x).view(-1, 320)

# 3. 수치 데이터 인코더
class TabularEncoder(nn.Module):
    def __init__(self, input_dim=9, output_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.SiLU()
        )

    def forward(self, x):
        return self.mlp(x)

# 4. 멀티모달 통합 모델 (Late Fusion)
class MobileViT_XXS_Multimodal(nn.Module):
    def __init__(self, image_channels=5, tabular_dim=9, output_dim=3, img_size=256):
        super().__init__()
        self.img_size = img_size
        self.backbone = MobileViT_Backbone(in_channels=image_channels)
        self.tab_encoder = TabularEncoder(input_dim=tabular_dim, output_dim=32)
        self.fusion_head = nn.Sequential(
            nn.Linear(320 + 32, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, img, tab):
        if img.size(2) != self.img_size or img.size(3) != self.img_size:
            img = F.interpolate(img, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        feat_img = self.backbone(img)
        feat_tab = self.tab_encoder(tab)
        combined = torch.cat((feat_img, feat_tab), dim=1)
        return self.fusion_head(combined)
