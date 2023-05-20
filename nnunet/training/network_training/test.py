import torch
import torch.nn as nn


data= torch.Tensor(64, 80, 3,3)
out_w, out_h = 1, 1
out_layer1 = nn.Sequential(
    nn.Conv2d(80, 32,kernel_size=3,groups=2),
    nn.LayerNorm([32,out_w,out_h]),
    nn.Dropout(0.1)
)
out = out_layer1(data)
print(out.shape)

data= torch.Tensor(64, 80, 5,5)
out_w, out_h = 3, 3
out_layer1 = nn.Sequential(
    nn.Conv2d(80, 32,kernel_size=3,groups=2),
    nn.LayerNorm([32,out_w,out_h]),
    nn.Dropout(0.1)
)
out = out_layer1(data)
print(out.shape)