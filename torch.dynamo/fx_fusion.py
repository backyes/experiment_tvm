import torch
from torch.fx.experimental.optimization import fuse
from torchvision.models import resnet18

model = resnet18()
model.eval() # 必须在eval模型下fuse
print(model)
print("----")
fused_model = fuse(model)
print("fused_model")
print(fused_model)
# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer2): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer3): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer4): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=512, out_features=1000, bias=True)
# )
# ----
# fused_model
# GraphModule(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Module(
#     (0): Module(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     )
#     (1): Module(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     )
#   )
#   (layer2): Module(
#     (0): Module(
#       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (downsample): Module(
#         (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2))
#       )
#     )
#     (1): Module(
#       (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     )
#   )
#   (layer3): Module(
#     (0): Module(
#       (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (downsample): Module(
#         (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2))
#       )
#     )
#     (1): Module(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     )
#   )
#   (layer4): Module(
#     (0): Module(
#       (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (downsample): Module(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
#       )
#     )
#     (1): Module(
#       (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=512, out_features=1000, bias=True)
# )
# 
# 
# 
# def forward(self, x : torch.Tensor) -> torch.Tensor:
#     conv1 = self.conv1(x);  x = None
#     relu = self.relu(conv1);  conv1 = None
#     maxpool = self.maxpool(relu);  relu = None
#     layer1_0_conv1 = getattr(self.layer1, "0").conv1(maxpool)
#     layer1_0_relu = getattr(self.layer1, "0").relu(layer1_0_conv1);  layer1_0_conv1 = None
#     layer1_0_conv2 = getattr(self.layer1, "0").conv2(layer1_0_relu);  layer1_0_relu = None
#     add = layer1_0_conv2 + maxpool;  layer1_0_conv2 = maxpool = None
#     layer1_0_relu_1 = getattr(self.layer1, "0").relu(add);  add = None
#     layer1_1_conv1 = getattr(self.layer1, "1").conv1(layer1_0_relu_1)
#     layer1_1_relu = getattr(self.layer1, "1").relu(layer1_1_conv1);  layer1_1_conv1 = None
#     layer1_1_conv2 = getattr(self.layer1, "1").conv2(layer1_1_relu);  layer1_1_relu = None
#     add_1 = layer1_1_conv2 + layer1_0_relu_1;  layer1_1_conv2 = layer1_0_relu_1 = None
#     layer1_1_relu_1 = getattr(self.layer1, "1").relu(add_1);  add_1 = None
#     layer2_0_conv1 = getattr(self.layer2, "0").conv1(layer1_1_relu_1)
#     layer2_0_relu = getattr(self.layer2, "0").relu(layer2_0_conv1);  layer2_0_conv1 = None
#     layer2_0_conv2 = getattr(self.layer2, "0").conv2(layer2_0_relu);  layer2_0_relu = None
#     layer2_0_downsample_0 = getattr(getattr(self.layer2, "0").downsample, "0")(layer1_1_relu_1);  layer1_1_relu_1 = None
#     add_2 = layer2_0_conv2 + layer2_0_downsample_0;  layer2_0_conv2 = layer2_0_downsample_0 = None
#     layer2_0_relu_1 = getattr(self.layer2, "0").relu(add_2);  add_2 = None
#     layer2_1_conv1 = getattr(self.layer2, "1").conv1(layer2_0_relu_1)
#     layer2_1_relu = getattr(self.layer2, "1").relu(layer2_1_conv1);  layer2_1_conv1 = None
#     layer2_1_conv2 = getattr(self.layer2, "1").conv2(layer2_1_relu);  layer2_1_relu = None
#     add_3 = layer2_1_conv2 + layer2_0_relu_1;  layer2_1_conv2 = layer2_0_relu_1 = None
#     layer2_1_relu_1 = getattr(self.layer2, "1").relu(add_3);  add_3 = None
#     layer3_0_conv1 = getattr(self.layer3, "0").conv1(layer2_1_relu_1)
#     layer3_0_relu = getattr(self.layer3, "0").relu(layer3_0_conv1);  layer3_0_conv1 = None
#     layer3_0_conv2 = getattr(self.layer3, "0").conv2(layer3_0_relu);  layer3_0_relu = None
#     layer3_0_downsample_0 = getattr(getattr(self.layer3, "0").downsample, "0")(layer2_1_relu_1);  layer2_1_relu_1 = None
#     add_4 = layer3_0_conv2 + layer3_0_downsample_0;  layer3_0_conv2 = layer3_0_downsample_0 = None
#     layer3_0_relu_1 = getattr(self.layer3, "0").relu(add_4);  add_4 = None
#     layer3_1_conv1 = getattr(self.layer3, "1").conv1(layer3_0_relu_1)
#     layer3_1_relu = getattr(self.layer3, "1").relu(layer3_1_conv1);  layer3_1_conv1 = None
#     layer3_1_conv2 = getattr(self.layer3, "1").conv2(layer3_1_relu);  layer3_1_relu = None
#     add_5 = layer3_1_conv2 + layer3_0_relu_1;  layer3_1_conv2 = layer3_0_relu_1 = None
#     layer3_1_relu_1 = getattr(self.layer3, "1").relu(add_5);  add_5 = None
#     layer4_0_conv1 = getattr(self.layer4, "0").conv1(layer3_1_relu_1)
#     layer4_0_relu = getattr(self.layer4, "0").relu(layer4_0_conv1);  layer4_0_conv1 = None
#     layer4_0_conv2 = getattr(self.layer4, "0").conv2(layer4_0_relu);  layer4_0_relu = None
#     layer4_0_downsample_0 = getattr(getattr(self.layer4, "0").downsample, "0")(layer3_1_relu_1);  layer3_1_relu_1 = None
#     add_6 = layer4_0_conv2 + layer4_0_downsample_0;  layer4_0_conv2 = layer4_0_downsample_0 = None
#     layer4_0_relu_1 = getattr(self.layer4, "0").relu(add_6);  add_6 = None
#     layer4_1_conv1 = getattr(self.layer4, "1").conv1(layer4_0_relu_1)
#     layer4_1_relu = getattr(self.layer4, "1").relu(layer4_1_conv1);  layer4_1_conv1 = None
#     layer4_1_conv2 = getattr(self.layer4, "1").conv2(layer4_1_relu);  layer4_1_relu = None
#     add_7 = layer4_1_conv2 + layer4_0_relu_1;  layer4_1_conv2 = layer4_0_relu_1 = None
#     layer4_1_relu_1 = getattr(self.layer4, "1").relu(add_7);  add_7 = None
#     avgpool = self.avgpool(layer4_1_relu_1);  layer4_1_relu_1 = None
#     flatten = torch.flatten(avgpool, 1);  avgpool = None
#     fc = self.fc(flatten);  flatten = None
#     return fc
#     
# # To see more debug info, please use `graph_module.print_readable()`
