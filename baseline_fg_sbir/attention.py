import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Attention_global(nn.Module):
    def __init__(self):
        super(Attention_global, self).__init__()
        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        self.net = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=1),
                                 nn.BatchNorm2d(1024),
                                 nn.ReLU(),
                                 nn.Conv2d(1024, 1, kernel_size=1))
       
    def forward(self, backbone_tensor):
        identify = backbone_tensor
        backbone_tensor_1 = self.net(backbone_tensor)
        backbone_tensor_1 = backbone_tensor_1.view(backbone_tensor_1.size(0), -1)
        backbone_tensor_1 = nn.Softmax(dim=1)(backbone_tensor_1)
        backbone_tensor_1 = backbone_tensor_1.view(backbone_tensor_1.size(0), 1, backbone_tensor.size(2), backbone_tensor.size(3))
        fatt = identify*backbone_tensor_1
        fatt1 = identify +fatt
        fatt1 = self.pool_method(fatt1).view(-1, 2048)
        return  F.normalize(fatt1)

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        self.norm = nn.LayerNorm(2048)
        # self.mha = nn.MultiheadAttention(2048, num_heads=args.num_heads, batch_first=True)
        self.mha = nn.MultiheadAttention(2048, num_heads=8, batch_first=True)
    
    def forward(self, x):
        bs, c, h, w = x.shape
        x_att = x.reshape(bs, c, h*w).transpose(1, 2)
        x_att = self.norm(x_att)
        att_out, _  = self.mha(x_att, x_att, x_att)
        att_out = att_out.transpose(1, 2).reshape(bs, c, h, w)
        att_out = self.pool_method(att_out).view(-1, 2048)
        return F.normalize(att_out)
    
    
class Linear_global(nn.Module):
    def __init__(self, feature_num):
        super(Linear_global, self).__init__()
        self.head_layer = nn.Linear(2048, feature_num)
    
    def forward(self, x):
        return F.normalize(self.head_layer(x))
    
# input_tensor = torch.randn(68, 2048, 8, 8)
# model = SelfAttention(None)
# output = model(input_tensor)

# print("Output shape:", output.shape)