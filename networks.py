import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn import init
import math


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output
def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)


def deconv3x3(in_channels, out_channels, stride=2, padding=1, output_padding=1, bias=False):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        bias=bias)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=math.sqrt(5), mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inconv, self).__init__()
        # self.conv = conv3x3(in_channels, out_channels)
        self.conv = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Downconv(nn.Module):
    """
    A helper Module that performs 3 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels):
        super(Downconv, self).__init__()

        self.downconv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 196),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),

            conv3x3(196, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.downconv(x)
        return x


class DOWN(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(DOWN, self).__init__()
        self.mpconv = nn.Sequential(
            Downconv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=384, out_channels=1):
        super(Decoder, self).__init__()

        self.conv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            conv3x3(64, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()

        self.inc = inconv(in_channels, 64)

        self.down1 = DOWN(64, 128)
        self.down2 = DOWN(128, 128)
        self.down3 = DOWN(128, 128)

    def forward(self, x):
        dx1 = self.inc(x)
        dx2 = self.down1(dx1)
        dx3 = self.down2(dx2)
        dx4 = self.down3(dx3)

        re_dx2 = F.adaptive_avg_pool2d(dx2, 32)
        re_dx3 = F.adaptive_avg_pool2d(dx3, 32)

        catfeat = torch.cat([re_dx2, re_dx3, dx4], 1)

        return catfeat, dx4


class FeatEmbedder(nn.Module):
    def __init__(self, in_channels=128):
        super(FeatEmbedder, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        #
        # self.classifier = nn.Sequential(nn.Linear(512, 128),
        #                                 nn.BatchNorm1d(128),
        #                                 nn.Dropout(p=0.3),
        #                                 nn.ReLU(),
        #                                 nn.Linear(128, 2))

    def forward(self, x):
        x = self.conv(x)
        # x = self.avgpooling(x)
        # x = x.view(x.size(0), -1)
        # feat = x
        # pred = self.classifier(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, nc=128, ndf=128):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),

        )

    def forward(self, x):
        output = self.model(x)
        return output

class Encoder_mv(nn.Module):

    def __init__(self):
        super(Encoder_mv, self).__init__()
        # model_resnet50 = torchvision.models.resnet50(pretrained=True)
        model_resnet18 = torchvision.models.resnet18(pretrained=True)
        # model_resnet101 = torchvision.models.resnet101(pretrained=True)
        layers_to_keep = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1','layer2','layer3','layer4']
        self.model_resnet18_layer1 = torch.nn.Sequential(*[getattr(model_resnet18, layer) for layer in layers_to_keep])

    def forward(self, x):  # x [3, 256, 256]

        x = self.model_resnet18_layer1(x)
        # print(x.shape)
        return x

# class Project_feature_con(nn.Module):
#     def __init__(self):
#         super(Project_feature_con, self).__init__()
#         device = torch.device("cuda:0")
#         self.weight_x = nn.Parameter(torch.tensor(1.0).to(device))
#         self.weight_y = nn.Parameter(torch.tensor(1.0).to(device))
#         self.weight_z = nn.Parameter(torch.tensor(0.5).to(device))
#         self.conv = nn.Sequential(
#             conv3x3(512, 512),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#         )
#         self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
#         self.project_feat_fc = nn.Linear(512, 512)
#         self.project_feat_fc.weight.data.normal_(0, 0.005)
#         self.project_feat_fc.bias.data.fill_(0.1)
#         self.project_layer = nn.Sequential(
#             self.project_feat_fc,
#             nn.ReLU(),
#         )
#     def forward(self, lf,hf,mv):
#         fre = self.weight_x * lf + self.weight_y * hf + self.weight_z * mv
#         fre = self.conv(fre)
#         x = self.avgpooling(fre)
#         x = x.view(x.size(0), -1)
#         project_feat = self.project_layer(x)
#         feature_norm = torch.norm(project_feat, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
#         feature = torch.div(project_feat, feature_norm)
#         return  feature
class Project_feature_con(nn.Module):
    def __init__(self):
        super(Project_feature_con, self).__init__()
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.project_feat_fc = nn.Linear(512, 512)
        self.project_feat_fc.weight.data.normal_(0, 0.005)
        self.project_feat_fc.bias.data.fill_(0.1)
        self.project_layer = nn.Sequential(
            self.project_feat_fc,
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        project_feat = self.project_layer(x)
        feature_norm = torch.norm(project_feat, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
        feature = torch.div(project_feat, feature_norm)
        return  feature
class Project_feature_con_depth(nn.Module):
    def __init__(self):
        super(Project_feature_con_depth, self).__init__()
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.project_feat_fc = nn.Linear(512, 512)
        self.project_feat_fc.weight.data.normal_(0, 0.005)
        self.project_feat_fc.bias.data.fill_(0.1)
        self.project_layer = nn.Sequential(
            self.project_feat_fc,
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        project_feat = self.project_layer(x)
        feature_norm = torch.norm(project_feat, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
        feature = torch.div(project_feat, feature_norm)
        return  feature


# class Project_feature_cls(nn.Module):
#     def __init__(self):
#         super(Project_feature_cls, self).__init__()
#         self.classifier_layer = nn.Linear(1280, 256)
#         self.classifier_layer.weight.data.normal_(0, 0.01)
#         self.classifier_layer.bias.data.fill_(0.0)
#         # self.project_feat = nn.Sequential(
#         #                                   nn.Dropout(p=0.3),
#         #                                   nn.ReLU(),
#         #                                   nn.Linear(128, 2),
#         #                                   )
#
#     def forward(self, x):
#         self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
#         x = self.classifier_layer(x)
#         # x = self.project_feat(x)
#         return x

class Project_feature_cls(nn.Module):
    def __init__(self):
        super(Project_feature_cls, self).__init__()
        # self.classifier_layer = nn.Linear(512, 2)
        # self.classifier_layer.weight.data.normal_(0, 0.01)
        # self.classifier_layer.bias.data.fill_(0.0)
        # self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_layer = nn.Sequential(
                                          nn.Linear(512, 128),
                                          nn.BatchNorm1d(128),
                                          nn.Dropout(p=0.5),
                                          nn.ReLU(),
                                          nn.Linear(128, 2),
                                          )

    def forward(self, x):
        # self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
        # x = self.avgpooling(x)
        # x = x.view(x.size(0),-1)
        # print(x.shape)
        x = self.classifier_layer(x)
        return x


import torch
import torch.nn as nn
normalizer = lambda x: x / (torch.norm(x, dim=-1, keepdim=True) + 1e-10)
# class Project_feature_cls(nn.Module):
#     def __init__(self):
#         super(Project_feature_cls, self).__init__()
#         self.fc0 = NormedLogisticRegression(512, 1,  use_bias=False)
#         self.fc1 = NormedLogisticRegression(512, 1,  use_bias=False)
#         self.fc2 = NormedLogisticRegression(512, 1,  use_bias=False)
#
#         self.bn_scale = nn.BatchNorm1d(1)
#         self.fc_scale = nn.Linear(512, 1)
#
#
#     def update_weight_v4(self, alpha=0.99):
#         beta0 = normalizer(self.fc0.weight.data.squeeze())
#         beta1 = normalizer(self.fc1.weight.data.squeeze())
#         beta2 = normalizer(self.fc2.weight.data.squeeze())
#
#         beta0_fr =  min([((beta0 * beta1).sum().item(), 0, beta1), ((beta0 * beta2).sum().item(), 1, beta2)])[2]
#         beta1_fr =  min([((beta1 * beta0).sum().item(), 0, beta0), ((beta1 * beta2).sum().item(), 1, beta2)])[2]
#         beta2_fr =  min([((beta2 * beta0).sum().item(), 0, beta0), ((beta2 * beta1).sum().item(), 1, beta1)])[2]
#
#         # 我明白了，这里的一维度结果起到了权重的作用。
#         self.fc0.weight.data = (normalizer(alpha * beta0 + (1 - alpha) * beta0_fr) * torch.norm(self.fc0.weight.data)).unsqueeze(0)
#         self.fc1.weight.data = (normalizer(alpha * beta1 + (1 - alpha) * beta1_fr) * torch.norm(self.fc1.weight.data)).unsqueeze(0)
#         self.fc2.weight.data = (normalizer(alpha * beta2 + (1 - alpha) * beta2_fr) * torch.norm(self.fc2.weight.data)).unsqueeze(0)
#
#         return ((beta0 * beta1).sum() + (beta2 * beta1).sum() + (beta0 * beta2).sum()) / 3
#     def forward(self, x, scale=None):
#         # x = x.view(x.size(0), -1)
#         if scale is None:
#             scale = torch.exp(self.bn_scale(self.fc_scale(x)))
#         else:
#             scale = torch.ones_like(torch.exp(self.bn_scale(self.fc_scale(x)))) * scale
#
#
#         return (self.fc0(x, scale) + self.fc1(x, scale)+ self.fc2(x, scale)) / 3
#         # return self.fc0(x,scale)




class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        # self.frefusion = AttentionFeatureFusion()
        device = torch.device("cuda:0")
        self.weight_x = nn.Parameter(torch.tensor(1.0).to(device))
        self.weight_y = nn.Parameter(torch.tensor(1.0).to(device))
        self.weight_z = nn.Parameter(torch.tensor(0.1).to(device))
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv_down = nn.Sequential(
        #     conv3x3(512, 512),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU()
        # )
        self.weight_fre = nn.Parameter(torch.tensor(0.5).to(device))
        self.weight_depth = nn.Parameter(torch.tensor(1.0).to(device))
        self.weight_fusion = nn.Parameter(torch.tensor(1.0).to(device))

        self.MutanFusion1 = MutanFusion()
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, lf,hf,mv,depth):
        # tmp = self.frefusion(hf,lf,mv)
        tmp = self.weight_x * hf + self.weight_y * lf + self.weight_z * mv
        # self.fre = tmp
        # tmp = self.conv_down(tmp)
        # tmp = self.avgpooling(tmp)
        # tmp = tmp.view(tmp.size(0), -1)
        # tmp = self.project_feat(tmp)
        fusion = self.MutanFusion1(tmp, depth)
        fusion = self.weight_fre * tmp + self.weight_fusion * fusion + self.weight_depth * depth
        # fusion = torch.cat([self.weight_branch_fre * tmp,depth],dim=1)
        # fusion = tmp +  depth
        # fusion = torch.mul(self.weight_branch,tmp) + torch.mul(1-self.weight_branch, depth)
        return tmp



class MutanFusion(nn.Module):

    def __init__(self):
        super(MutanFusion, self).__init__()
        # self.linear_v = nn.Linear(512, 512)
        # self.linear_q = nn.Linear(512, 512)
        self.list_linear_hv = nn.ModuleList([
            nn.Linear(512, 512)
            for i in range(2)])

        self.list_linear_hq = nn.ModuleList([
            nn.Linear(512, 512)
            for i in range(2)])
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 2:
            raise ValueError

        # 这里很有可能导致性能不佳。
        if not input_v.is_contiguous():
            input_v = input_v.contiguous()
        if not input_q.is_contiguous():
            input_q = input_q.contiguous()


        # height = input_v.size(2)
        # weight = input_v.size(3)
        # batch  = input_v.size(0)
        # feat = input_v.size(1)
        #
        # input_v = input_v.view(input_v.size(0) * input_v.size(2) * input_v.size(3), input_v.size(1))
        # input_q = input_q.view(input_q.size(0) * input_q.size(2) * input_q.size(3), input_q.size(1))


        x_mm = []
        for i in range(2):

            x_hv = F.dropout(input_v, p=0.5, training=self.training)
            x_hv = self.list_linear_hv[i](x_hv)
            x_hv = F.tanh(x_hv)

            x_hq = F.dropout(input_q, p=0.5, training=self.training)
            x_hq = self.list_linear_hq[i](x_hq)
            x_hq = F.tanh(x_hq)

            x_mm.append(torch.mul(x_hq, x_hv))

        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1)
        x_mm = F.tanh(x_mm)
        # x_mm = x_mm.view(batch, feat, height, weight)
        # x_mm = self.avgpooling(x_mm)
        # x_mm = x_mm.view(x_mm.size(0), -1)
        return x_mm







