import torch
import torchvision

vgg16 = torchvision.models.vgg16(weights=None)
# 保存方式1
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2（官方推荐）
# 获取 vgg16 的状态并保存为字典，
torch.save(vgg16.state_dict(), "vgg16_method2.pth")