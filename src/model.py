import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------
# 1. 유틸리티 및 기본 블록
# --------------------------------------------------------

def conv_bn_act(in_c, out_c, kernel_size=3, stride=1, padding=None, groups=1):
    # 패딩이 명시되지 않았다면, 이미지 크기를 유지하도록 자동 설정 (k=3 -> p=1)
    if padding is None:
        padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_c),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).permute(0, 2, 1, 3), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# --------------------------------------------------------
# 2. MobileViT 핵심 블록
# --------------------------------------------------------

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size

        # Local Representation
        self.conv1 = conv_bn_act(channel, channel, kernel_size)
        self.conv2 = nn.Conv2d(channel, dim, 1, bias=False)

        # Global Representation (Transformer)
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        # Fusion
        self.conv3 = nn.Conv2d(dim, channel, 1, bias=False)
        self.conv4 = conv_bn_act(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local
        x = self.conv1(x)
        x = self.conv2(x)

        # Global (Unfold -> Transformer -> Fold)
        _, _, h, w = x.shape

        # Patching (256x256 보장 시 h,w는 항상 짝수라 안전함)
        x = x.reshape(x.shape[0], x.shape[1], h // self.ph, self.ph, w // self.pw, self.pw)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        x = self.transformer(x)

        # Un-patching
        x = x.reshape(x.shape[0], h // self.ph, w // self.pw, self.ph, self.pw, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(x.shape[0], -1, h, w)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((y, x), dim=1)
        x = self.conv4(x)
        return x

class MV2Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expand_ratio=4):
        super().__init__()
        hidden_dim = in_c * expand_ratio
        self.use_res_connect = stride == 1 and in_c == out_c

        layers = []
        if expand_ratio != 1:
            # kernel_size=1 -> padding=0 (자동)
            layers.append(conv_bn_act(in_c, hidden_dim, kernel_size=1))

        layers.extend([
            # Depthwise: kernel=3 -> padding=1 (자동)
            conv_bn_act(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

# --------------------------------------------------------
# 3. 전체 모델 (MobileViT-XXS)
# --------------------------------------------------------

class MobileViT_XXS(nn.Module):
    def __init__(self, input_channels=5, output_dim=3, img_size=256):
        super(MobileViT_XXS, self).__init__()

        # XXS Configuration
        dims = [64, 80, 96]
        channels = [16, 16, 24, 48, 64, 80]

        # [핵심] 256으로 설정하여 차원 에러 방지
        self.img_size = img_size

        # Backbone
        self.conv1 = conv_bn_act(input_channels, channels[0], stride=2)

        self.mv2_1 = MV2Block(channels[0], channels[1], stride=1)
        self.mv2_2 = nn.Sequential(
            MV2Block(channels[1], channels[2], stride=2),
            MV2Block(channels[2], channels[2], stride=1),
            MV2Block(channels[2], channels[2], stride=1)
        )

        self.mv2_3 = MV2Block(channels[2], channels[3], stride=2)

        self.mvit_1 = MobileViTBlock(dims[0], depth=2, channel=channels[3], kernel_size=3, patch_size=2, mlp_dim=dims[0]*2)

        self.mv2_4 = MV2Block(channels[3], channels[4], stride=2)

        self.mvit_2 = MobileViTBlock(dims[1], depth=4, channel=channels[4], kernel_size=3, patch_size=2, mlp_dim=dims[1]*2)

        self.mv2_5 = MV2Block(channels[4], channels[5], stride=2)

        self.mvit_3 = MobileViTBlock(dims[2], depth=3, channel=channels[5], kernel_size=3, patch_size=2, mlp_dim=dims[2]*2)

        self.conv_last = conv_bn_act(channels[5], 320, kernel_size=1)

        # Head (Regression)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(320, output_dim)
        )

    def forward(self, x):
        # [Resize] 32 -> 256
        if x.size(2) < self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        x = self.conv1(x)
        x = self.mv2_1(x)
        x = self.mv2_2(x)
        x = self.mv2_3(x)
        x = self.mvit_1(x)
        x = self.mv2_4(x)
        x = self.mvit_2(x)
        x = self.mv2_5(x)
        x = self.mvit_3(x)
        x = self.conv_last(x)

        x = self.pool(x).view(-1, 320)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 모델 초기화 (256 모드)
    model = MobileViT_XXS(input_channels=5, output_dim=3, img_size=256).to(device)
    dummy_input = torch.randn(2, 5, 32, 32).to(device)

    # 실행 테스트
    output = model(dummy_input)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"✅ MobileViT-XXS (256 Ver.) 구축 완료")
    print(f" - 총 파라미터 수: {total_params / 1e6:.2f}M")
    print(f" - 입력(Resize): {dummy_input.shape} -> 256x256")
    print(f" - 출력 크기: {output.shape} (기대값: [2, 3])")
