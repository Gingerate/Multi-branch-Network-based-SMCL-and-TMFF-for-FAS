import networks
import losses
import torch
from torch import nn
import os
import itertools
from collections import OrderedDict
import torch.nn.functional as F
from hard_triplet_loss import HardTripletLoss
class FaceModel(nn.Module):
    def __init__(self, opt, isTrain=True, input_nc=3):
        super(FaceModel, self).__init__()
        self.opt = opt
        self.model = opt.model
        self.w_cls = opt.w_cls
        self.w_L1 = opt.w_L1
        self.w_gan = opt.w_gan
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        # torch.backends.cudnn.benchmark = True
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.isTrain = isTrain


        self.netEncoder = networks.init_net(networks.Encoder(in_channels=input_nc), gpu_ids=self.gpu_ids)
        # bestEncoder_path = "//home//jinli//DFA-m//DFA-HF//checkpoints//Open-set Pro.2//train on iom test on casia//best_net_Encoder.pth"
        # bestEncoder_dict = torch.load(bestEncoder_path)
        # self.netEncoder.load_state_dict(bestEncoder_dict)


        self.netClassifier = networks.init_net(networks.FeatEmbedder(), gpu_ids=self.gpu_ids)
        # bestClassifier_path = "//home//jinli//DFA-m//DFA-HF//checkpoints//Open-set Pro.2//train on iom test on casia//best_net_Classifier.pth"
        # bestClassifier_dict = torch.load(bestClassifier_path)
        # self.netClassifier.load_state_dict(bestClassifier_dict)


        self.netDepthDecoder = networks.init_net(networks.Decoder(), gpu_ids=self.gpu_ids)
        self.netDepthDiscriminator = networks.init_net(networks.Discriminator(nc=4), gpu_ids=self.gpu_ids)

        self.netEncoderlf = networks.init_net(networks.Encoder_mv(), gpu_ids=self.gpu_ids)

        self.netEncoderhf = networks.init_net(networks.Encoder_mv(), gpu_ids=self.gpu_ids)

        self.netEncodermv = networks.init_net(networks.Encoder_mv(), gpu_ids=self.gpu_ids)


        self.netProConlf = networks.init_net(networks.Project_feature_con(), gpu_ids=self.gpu_ids)
        self.netProConhf = networks.init_net(networks.Project_feature_con(), gpu_ids=self.gpu_ids)
        self.netProConmv = networks.init_net(networks.Project_feature_con(), gpu_ids=self.gpu_ids)
        self.netProCondepth = networks.init_net(networks.Project_feature_con_depth(), gpu_ids=self.gpu_ids)
        # self.netProConRre = networks.init_net(networks.Project_representation_con(), gpu_ids=self.gpu_ids)

        self.netFeatFusion = networks.init_net(networks.FeatureFusion(), gpu_ids=self.gpu_ids)

        self.netProCls = networks.init_net(networks.Project_feature_cls(), gpu_ids=self.gpu_ids)


        self.model_names = ["Encoder", "DepthDecoder", "DepthDiscriminator", "Classifier", "Encoderlf", "Encoderhf", "Encodermv", "ProConlf", "ProConhf", "ProConmv","ProCondepth","FeatFusion", "ProCls"]
        self.visual_names = ["real_A", "real_B", "fake_B"]
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'C']
        if self.isTrain:
            # Discriminator loss
            self.criterionGan = losses.GANLoss().to(self.device)
            # Decoder loss
            self.criterionL1 = torch.nn.L1Loss()
            # cls loss
            self.criterionCls = [torch.nn.CrossEntropyLoss(), losses.FocalLoss()]
            # tri loss
            self.criterionTri = HardTripletLoss(margin=0.1, hardest=False).to(self.device)
            # net G/
            self.optimizer_depth = torch.optim.Adam(itertools.chain(self.netEncoder.parameters(),
                                                                    self.netDepthDecoder.parameters()), lr=opt.lr,
                                                    betas=(opt.beta1, 0.999))

            # net D/
            self.optimizer_discriminate = torch.optim.Adam(self.netDepthDiscriminator.parameters(), lr=opt.lr,
                                                           betas=(opt.beta1, 0.999))

            # net cls
            self.optimizer_cls = torch.optim.Adam(itertools.chain(self.netEncoder.parameters(),
                                                                  self.netClassifier.parameters(),
                                                                  self.netEncoderlf.parameters(),
                                                                  self.netEncoderhf.parameters(),
                                                                  self.netEncodermv.parameters(),
                                                                  self.netProConlf.parameters(),
                                                                  self.netProConhf.parameters(),
                                                                  self.netProConmv.parameters(),
                                                                  self.netProCondepth.parameters(),
                                                                  self.netFeatFusion.parameters(),
                                                                  self.netProCls.parameters()), lr=opt.lr,
                                                  betas=(opt.beta1, 0.999), weight_decay=0.01)
            self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))

            # self.scheduler_cls = torch.optim.lr_scheduler.StepLR(self.optimizer_cls, step_size=5, gamma=0.1)


    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_A_32 = input['A_32'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.label = torch.tensor(input['label']).to(self.device)
        self.LF = input['LF'].to(self.device)
        self.HF = input['HF'].to(self.device)
        self.MV = input['MV'].to(self.device)
        # self.label = torch.tensor(input['label']).to(self.device)
        self.UUID = input['UUID'].detach().clone().to(self.device)


    def forward(self):
        self.lantent_0, self.lantent_1 = self.netEncoder(self.real_A)
        self.fake_B = self.netDepthDecoder(self.lantent_0)
        self.depth = self.netClassifier(self.lantent_1)
        self.lf = self.netEncoderlf(self.LF)
        self.hf = self.netEncoderhf(self.HF)
        self.mv = self.netEncodermv(self.MV)
        # self.frecon = self.netProConlf(self.lf,self.hf,self.mv)

        self.lfcon = self.netProConlf(self.lf)
        self.hfcon = self.netProConhf(self.hf)
        self.mvcon = self.netProConmv(self.mv)
        self.depthcon = self.netProCondepth(self.depth)
        self.representation = self.netFeatFusion(self.lfcon,self.hfcon,self.mvcon,self.depthcon)
        # self.representation = self.mvcon
        # self.representation = self.netFeatFusion(self.frecon,self.depthcon)
        # self.representation = self.netProConRre(self.representation)
        # self.representation = self.avgpooling(self.representation)
        # self.representation = self.representation.view(self.representation.size(0), -1)
        self.output = self.netProCls(self.representation)

    def backward_D(self):

        fake_AB = torch.cat((self.real_A_32, self.fake_B), 1)
        pred_fake = self.netDepthDiscriminator(fake_AB.detach())
        self.loss_D_fake = self.criterionGan(pred_fake, False)

        real_AB = torch.cat((self.real_A_32, self.real_B), 1)
        pred_real = self.netDepthDiscriminator(real_AB)
        self.loss_D_real = self.criterionGan(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.w_gan
        self.loss_D.backward()
        # depth

    def backward_G(self):
        fake_AB = torch.cat((self.real_A_32, self.fake_B), 1)
        pred_fake = self.netDepthDiscriminator(fake_AB)
        self.loss_G_GAN = self.criterionGan(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_G = self.loss_G_L1 * self.w_L1 + self.loss_G_GAN * self.w_gan
        self.loss_G.backward()
    def backward_C(self):
        # 分类损失
        output = self.output
        self.loss_Cls = (2 * self.criterionCls[0](output, self.label) + self.criterionCls[1](output, self.label)) * self.w_cls

        # 对比损失

        # losses.SupConLoss(torch.cat([self.lfcon.unsqueeze(1), self.mvcon.unsqueeze(1)], dim=1), self.label) + \

        #                   self.label) + \
        # losses.SupConLoss(torch.cat([self.mvcon.unsqueeze(1), self.lfcon.unsqueeze(1)], dim=1),
        #                   self.label) + \
        # losses.SupConLoss(torch.cat([self.depthcon.unsqueeze(1), self.mvcon.unsqueeze(1)], dim=1),
        #                   self.label) + \
        # losses.SupConLoss(torch.cat([self.depthcon.unsqueeze(1), self.lfcon.unsqueeze(1)], dim=1),
        #                   self.label) + \
        # losses.SupConLoss(torch.cat([self.depthcon.unsqueeze(1), self.hfcon.unsqueeze(1)], dim=1),
        #                   self.label)
        # losses.SupConLoss(torch.cat([self.lfcon.unsqueeze(1), self.hfcon.unsqueeze(1)], dim=1),
        #                   self.label) + \
        # losses.SupConLoss(torch.cat([self.mvcon.unsqueeze(1), self.hfcon.unsqueeze(1)], dim=1),
        #                   self.label) + \

        self.loss_Con = (
                        losses.SupConLoss(torch.cat([self.hfcon.unsqueeze(1), self.mvcon.unsqueeze(1)], dim=1),
                                          self.label) + \
                        losses.SupConLoss(torch.cat([self.hfcon.unsqueeze(1), self.lfcon.unsqueeze(1)], dim=1),
                                          self.label) + \
                        losses.SupConLoss(torch.cat([self.lfcon.unsqueeze(1), self.depthcon.unsqueeze(1)], dim=1),
                                          self.label) + \
                        losses.SupConLoss(torch.cat([self.mvcon.unsqueeze(1), self.depthcon.unsqueeze(1)], dim=1),
                                          self.label) + \
                        losses.SupConLoss(torch.cat([self.hfcon.unsqueeze(1), self.depthcon.unsqueeze(1)], dim=1),
                                          self.label) / 5)

        # 三元组损失
        self.loss_Tri = self.criterionTri(self.representation, self.label)
        # self.loss_Tri = self.criterionTri(self.lfcon, self.label) + self.criterionTri(self.hfcon, self.label) + self.criterionTri(self.mvcon, self.label) + self.criterionTri(self.depthcon, self.label)

        self.loss_C = self.loss_Cls + 0.1 * self.loss_Con + 2 * self.loss_Tri

        self.loss_C.backward()

    def optimize_parameters(self):

        self.forward()
        self.set_requires_grad(self.netDepthDiscriminator, True)
        self.optimizer_discriminate.zero_grad()
        self.backward_D()
        self.optimizer_discriminate.step()
        self.set_requires_grad(self.netDepthDiscriminator, False)
        self.optimizer_depth.zero_grad()
        self.backward_G()
        self.optimizer_depth.step()
        self.forward()
        self.optimizer_cls.zero_grad()
        self.backward_C()
        self.optimizer_cls.step()


    def eval(self):
        self.isTrain = False
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """Make models eval mode during test time"""
        self.isTrain = True
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                net.load_state_dict(state_dict)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                try:
                    errors_ret[name] = float(
                        getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
                except Exception as e:
                    errors_ret[name] = -1
        return errors_ret

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)