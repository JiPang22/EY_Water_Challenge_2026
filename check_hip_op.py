import torch
import torch.nn.functional as F

device = torch.device('cuda')
try:
    # 에러가 발생했던 interpolation 연산 테스트
    test_tensor = torch.randn(1, 5, 32, 32).to(device)
    output = F.interpolate(test_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    print("GPU Operation Success: ", output.shape)
except Exception as e:
    print("GPU Operation Failed: ", e)
