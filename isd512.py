import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc300, voc512, coco
import os
import warnings
import math
import numpy as np
import cv2
import copy


class SSD_CON(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD_CON, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        if(size==300):
            self.cfg = (coco, voc300)[num_classes == 21]
        else:
            self.cfg = (coco, voc512)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)

        if phase == 'test':
            # self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            self.vgg_t = copy.deepcopy(self.vgg)
            self.extras_t = copy.deepcopy(self.extras)
            self.loc_t = copy.deepcopy(self.loc)
            self.conf_t = copy.deepcopy(self.conf)

        ### mt
        self.ema_factor = 0.999
        self.global_step = 0
        # self.vgg_t = copy.deepcopy(self.vgg)
        # self.extras_t = copy.deepcopy(self.extras)
        # self.loc_t = copy.deepcopy(self.loc)
        # self.conf_t = copy.deepcopy(self.conf)

        # for p in self.vgg_t.parameters(): p.requires_grad = False
        # for p in self.extras_t.parameters(): p.requires_grad = False
        # for p in self.loc_t.parameters(): p.requires_grad = False
        # for p in self.conf_t.parameters(): p.requires_grad = False

#    def clone_model(self):
#        self.vgg_t = copy.deepcopy(self.vgg)
#        self.extras_t = copy.deepcopy(self.extras)
#        self.loc_t = copy.deepcopy(self.loc)
#        self.conf_t = copy.deepcopy(self.conf)


    # def forward(self, x, x_flip, x_shuffle):
    def forward(self, images, conf_consistency_criterion):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """


        gpu_batch_size = images.size()[0]

        images_flip = images.clone()
        images_flip = flip(images_flip, 3)

        images_shuffle = images_flip.clone()
        images_shuffle[:int(gpu_batch_size / 2), :, :, :] = images_flip[int(gpu_batch_size / 2):, :, :, :]
        images_shuffle[int(gpu_batch_size / 2):, :, :, :] = images_flip[:int(gpu_batch_size / 2), :, :, :]

        lam = np.random.beta(5.0, 5.0)

        images_mix = lam * images.clone() + (1 - lam) * images_shuffle.clone()

        x = images
        x_flip = images_flip
        x_shuffle = images_shuffle


        # x_flip = x.clone()
        # x_flip = flip(x_flip,3)

        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # zero_mask = torch.cat([o.view(o.size(0), -1) for o in zero_mask], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        loc = loc.view(loc.size(0), -1, 4)
        conf = self.softmax(conf.view(conf.size(0), -1, self.num_classes))
        # basic


        sources_flip = list()
        loc_flip = list()
        conf_flip = list()
        loc_shuffle = list()
        conf_shuffle = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x_flip = self.vgg[k](x_flip)

        s_flip = self.L2Norm(x_flip)
        sources_flip.append(s_flip)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x_flip = self.vgg[k](x_flip)
        sources_flip.append(x_flip)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x_flip = F.relu(v(x_flip), inplace=True)
            if k % 2 == 1:
                sources_flip.append(x_flip)

        # apply multibox head to source layers
        for (x_flip, l, c) in zip(sources_flip, self.loc, self.conf):
            loc_shuffle.append(l(x_flip).permute(0, 2, 3, 1).contiguous())
            conf_shuffle.append(c(x_flip).permute(0, 2, 3, 1).contiguous())
            append_loc = l(x_flip).permute(0, 2, 3, 1).contiguous()
            append_conf = c(x_flip).permute(0, 2, 3, 1).contiguous()
            append_loc = flip(append_loc,2)
            append_conf = flip(append_conf,2)
            loc_flip.append(append_loc)
            conf_flip.append(append_conf)

        loc_shuffle = torch.cat([o.view(o.size(0), -1) for o in loc_shuffle], 1)
        conf_shuffle = torch.cat([o.view(o.size(0), -1) for o in conf_shuffle], 1)

        loc_flip = torch.cat([o.view(o.size(0), -1) for o in loc_flip], 1)
        conf_flip = torch.cat([o.view(o.size(0), -1) for o in conf_flip], 1)

        loc_shuffle = loc_flip.view(loc_shuffle.size(0), -1, 4)
        conf_shuffle = self.softmax(conf_shuffle.view(conf_shuffle.size(0), -1, self.num_classes))

        loc_flip = loc_flip.view(loc_flip.size(0), -1, 4)
        conf_flip = self.softmax(conf_flip.view(conf_flip.size(0), -1, self.num_classes))



        sources_interpolation = list()
        loc_interpolation = list()
        conf_interpolation = list()

#        # apply vgg up to conv4_3 relu
        for k in range(23):
            x_shuffle = self.vgg[k](x_shuffle)

        s_shuffle = self.L2Norm(x_shuffle)
        sources_interpolation.append(s_shuffle)



#        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x_shuffle = self.vgg[k](x_shuffle)
        sources_interpolation.append(x_shuffle)



#        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x_shuffle = F.relu(v(x_shuffle), inplace=True)
            if k % 2 == 1:
                sources_interpolation.append(x_shuffle)



#        # apply multibox head to source layers
        for (x_shuffle, l, c) in zip(sources_interpolation, self.loc, self.conf):
            loc_interpolation.append(l(x_shuffle).permute(0, 2, 3, 1).contiguous())
            conf_interpolation.append(c(x_shuffle).permute(0, 2, 3, 1).contiguous())


        loc_interpolation = torch.cat([o.view(o.size(0), -1) for o in loc_interpolation], 1)
        conf_interpolation = torch.cat([o.view(o.size(0), -1) for o in conf_interpolation], 1)

        loc_interpolation = loc_interpolation.view(loc_interpolation.size(0), -1, 4)

        conf_interpolation = self.softmax(conf_interpolation.view(conf_interpolation.size(0), -1, self.num_classes))


        if self.phase == "test":
            return output
        else:
            conf_class = conf[:, :, 1:].clone()
            background_score = conf[:, :, 0].clone()
            each_val, each_index = torch.max(conf_class, dim=2)
            mask_val = each_val > background_score
            mask_val = mask_val.data

            mask_conf_index = mask_val.unsqueeze(2).expand_as(conf)
            mask_loc_index = mask_val.unsqueeze(2).expand_as(loc)

            # conf_mask_sample = conf.clone()
            # loc_mask_sample = loc.clone()
            conf_sampled = conf[mask_conf_index].view(-1, 21).clone()
            loc_sampled = loc[mask_loc_index].view(-1, 4).clone()

            # conf_mask_sample_flip = conf_flip.clone()
            # loc_mask_sample_flip = loc_flip.clone()
            conf_sampled_flip = conf_flip[mask_conf_index].view(-1, 21).clone()
            loc_sampled_flip = loc_flip[mask_loc_index].view(-1, 4).clone()

            if (mask_val.sum() > 0):
                ## JSD !!!!!1
                conf_sampled_flip = conf_sampled_flip + 1e-7
                conf_sampled = conf_sampled + 1e-7
                consistency_conf_loss_a = conf_consistency_criterion(conf_sampled.log(),
                                                                     conf_sampled_flip.detach()).sum(-1).mean()
                consistency_conf_loss_b = conf_consistency_criterion(conf_sampled_flip.log(),
                                                                     conf_sampled.detach()).sum(-1).mean()
                consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b

                ## LOC LOSS
                consistency_loc_loss_x = torch.mean(torch.pow(loc_sampled[:, 0] + loc_sampled_flip[:, 0], exponent=2))
                consistency_loc_loss_y = torch.mean(torch.pow(loc_sampled[:, 1] - loc_sampled_flip[:, 1], exponent=2))
                consistency_loc_loss_w = torch.mean(torch.pow(loc_sampled[:, 2] - loc_sampled_flip[:, 2], exponent=2))
                consistency_loc_loss_h = torch.mean(torch.pow(loc_sampled[:, 3] - loc_sampled_flip[:, 3], exponent=2))

                consistency_loc_loss = torch.div(
                    consistency_loc_loss_x + consistency_loc_loss_y + consistency_loc_loss_w + consistency_loc_loss_h,
                    4)

            else:
                consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
                consistency_loc_loss = Variable(torch.cuda.FloatTensor([0]))
                consistency_conf_loss = consistency_conf_loss.data[0]
                consistency_loc_loss = consistency_loc_loss.data[0]

            consistency_loss = torch.div(consistency_conf_loss, 2) + consistency_loc_loss


            ### interpolation regularization
            # out, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation
            conf_temp = conf_shuffle.clone()
            loc_temp = loc_shuffle.clone()
            conf_temp[:int(gpu_batch_size/2),:,:] = conf_shuffle[int(gpu_batch_size/2):,:,:]
            conf_temp[int(gpu_batch_size / 2):, :, :] = conf_shuffle[:int(gpu_batch_size / 2), :, :]
            loc_temp[:int(gpu_batch_size / 2), :, :] = loc_shuffle[int(gpu_batch_size / 2):, :, :]
            loc_temp[int(gpu_batch_size / 2):, :, :] = loc_shuffle[:int(gpu_batch_size / 2), :, :]


            ## original background elimination
            left_conf_class = conf[:, :, 1:].clone()
            left_background_score = conf[:, :, 0].clone()
            left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
            left_mask_val = left_each_val > left_background_score
            left_mask_val = left_mask_val.data

            ## flip background elimination
            right_conf_class = conf_shuffle[:, :, 1:].clone()
            right_background_score = conf_shuffle[:, :, 0].clone()
            right_each_val, right_each_index = torch.max(right_conf_class, dim=2)
            right_mask_val = right_each_val > right_background_score
            right_mask_val = right_mask_val.data

            ## both background elimination
            only_left_mask_val = left_mask_val * (1-right_mask_val)
            only_right_mask_val = right_mask_val * (1-left_mask_val)

            intersection_mask_val = left_mask_val * right_mask_val

## intersection_ict
            intersection_mask_conf_index = intersection_mask_val.unsqueeze(2).expand_as(conf)

            intersection_left_conf_mask_sample = conf.clone()
            intersection_left_conf_sampled = intersection_left_conf_mask_sample[intersection_mask_conf_index].view(-1,
                                                                                                                   21)

            intersection_right_conf_mask_sample = conf_temp.clone()
            intersection_right_conf_sampled = intersection_right_conf_mask_sample[intersection_mask_conf_index].view(-1,
                                                                                                                     21)

            intersection_intersection_conf_mask_sample = conf_interpolation.clone()
            intersection_intersection_sampled = intersection_intersection_conf_mask_sample[
                intersection_mask_conf_index].view(-1, 21)

            if (intersection_mask_val.sum() > 0):

                mixed_val = lam * intersection_left_conf_sampled + (1 - lam) * intersection_right_conf_sampled

                mixed_val = mixed_val + 1e-7
                intersection_intersection_sampled = intersection_intersection_sampled + 1e-7

                interpolation_consistency_conf_loss_a = conf_consistency_criterion(mixed_val.log(),
                                                                                   intersection_intersection_sampled.detach()).sum(
                    -1).mean()
                interpolation_consistency_conf_loss_b = conf_consistency_criterion(
                    intersection_intersection_sampled.log(),
                    mixed_val.detach()).sum(-1).mean()
                interpolation_consistency_conf_loss = interpolation_consistency_conf_loss_a + interpolation_consistency_conf_loss_b
                interpolation_consistency_conf_loss = torch.div(interpolation_consistency_conf_loss, 2)
            else:
                interpolation_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
                interpolation_consistency_conf_loss = interpolation_consistency_conf_loss.data[0]

## original fixmatch
            only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)
            only_left_mask_loc_index = only_left_mask_val.unsqueeze(2).expand_as(loc)

            ori_fixmatch_conf_mask_sample = conf.clone()
            ori_fixmatch_loc_mask_sample = loc.clone()
            ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)
            ori_fixmatch_loc_sampled = ori_fixmatch_loc_mask_sample[only_left_mask_loc_index].view(-1, 4)

            ori_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
            ori_fixmatch_loc_mask_sample_interpolation = loc_interpolation.clone()
            ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
                only_left_mask_conf_index].view(-1, 21)
            ori_fixmatch_loc_sampled_interpolation = ori_fixmatch_loc_mask_sample_interpolation[
                only_left_mask_loc_index].view(-1, 4)

            if (only_left_mask_val.sum() > 0):
                ## KLD !!!!!1
                ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_sampled_interpolation + 1e-7
                ori_fixmatch_conf_sampled = ori_fixmatch_conf_sampled + 1e-7
                only_left_consistency_conf_loss_a = conf_consistency_criterion(
                    ori_fixmatch_conf_sampled_interpolation.log(),
                    ori_fixmatch_conf_sampled.detach()).sum(-1).mean()
                # consistency_conf_loss_b = conf_consistency_criterion(conf_sampled_flip.log(),
                #                                                      conf_sampled.detach()).sum(-1).mean()
                # consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b
                only_left_consistency_conf_loss = only_left_consistency_conf_loss_a

                ## LOC LOSS
                only_left_consistency_loc_loss_x = torch.mean(torch.pow(
                    ori_fixmatch_loc_sampled_interpolation[:, 0] - ori_fixmatch_loc_sampled[:, 0].detach(),
                    exponent=2))
                only_left_consistency_loc_loss_y = torch.mean(torch.pow(
                    ori_fixmatch_loc_sampled_interpolation[:, 1] - ori_fixmatch_loc_sampled[:, 1].detach(),
                    exponent=2))
                only_left_consistency_loc_loss_w = torch.mean(torch.pow(
                    ori_fixmatch_loc_sampled_interpolation[:, 2] - ori_fixmatch_loc_sampled[:, 2].detach(),
                    exponent=2))
                only_left_consistency_loc_loss_h = torch.mean(torch.pow(
                    ori_fixmatch_loc_sampled_interpolation[:, 3] - ori_fixmatch_loc_sampled[:, 3].detach(),
                    exponent=2))

                only_left_consistency_loc_loss = torch.div(
                    only_left_consistency_loc_loss_x + only_left_consistency_loc_loss_y + only_left_consistency_loc_loss_w + only_left_consistency_loc_loss_h,
                    4)

            else:
                only_left_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
                only_left_consistency_loc_loss = Variable(torch.cuda.FloatTensor([0]))
                only_left_consistency_conf_loss = only_left_consistency_conf_loss.data[0]
                only_left_consistency_loc_loss = only_left_consistency_loc_loss.data[0]


            # consistency_loss = consistency_conf_loss  # consistency_loc_loss
            only_left_consistency_loss = only_left_consistency_conf_loss + only_left_consistency_loc_loss
        #            only_left_consistency_loss = only_left_consistency_conf_loss

## flip fixmatch

            only_right_mask_conf_index = only_right_mask_val.unsqueeze(2).expand_as(conf)
            only_right_mask_loc_index = only_right_mask_val.unsqueeze(2).expand_as(loc)

            flip_fixmatch_conf_mask_sample = conf_temp.clone()
            flip_fixmatch_loc_mask_sample = loc_temp.clone()
            flip_fixmatch_conf_sampled = flip_fixmatch_conf_mask_sample[only_right_mask_conf_index].view(-1, 21)
            flip_fixmatch_loc_sampled = flip_fixmatch_loc_mask_sample[only_right_mask_loc_index].view(-1, 4)

            flip_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
            flip_fixmatch_loc_mask_sample_interpolation = loc_interpolation.clone()
            flip_fixmatch_conf_sampled_interpolation = flip_fixmatch_conf_mask_sample_interpolation[
                only_right_mask_conf_index].view(-1, 21)
            flip_fixmatch_loc_sampled_interpolation = flip_fixmatch_loc_mask_sample_interpolation[
                only_right_mask_loc_index].view(-1, 4)

            if (only_right_mask_val.sum() > 0):
                ## KLD !!!!!1
                flip_fixmatch_conf_sampled_interpolation = flip_fixmatch_conf_sampled_interpolation + 1e-7
                flip_fixmatch_conf_sampled = flip_fixmatch_conf_sampled + 1e-7
                only_right_consistency_conf_loss_a = conf_consistency_criterion(
                    flip_fixmatch_conf_sampled_interpolation.log(),
                    flip_fixmatch_conf_sampled.detach()).sum(-1).mean()
                # consistency_conf_loss_b = conf_consistency_criterion(conf_sampled_flip.log(),
                #                                                      conf_sampled.detach()).sum(-1).mean()
                # consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b
                only_right_consistency_conf_loss = only_right_consistency_conf_loss_a

                ## LOC LOSS
                only_right_consistency_loc_loss_x = torch.mean(
                    torch.pow(
                        flip_fixmatch_loc_sampled_interpolation[:, 0] - flip_fixmatch_loc_sampled[:, 0].detach(),
                        exponent=2))
                only_right_consistency_loc_loss_y = torch.mean(
                    torch.pow(
                        flip_fixmatch_loc_sampled_interpolation[:, 1] - flip_fixmatch_loc_sampled[:, 1].detach(),
                        exponent=2))
                only_right_consistency_loc_loss_w = torch.mean(
                    torch.pow(
                        flip_fixmatch_loc_sampled_interpolation[:, 2] - flip_fixmatch_loc_sampled[:, 2].detach(),
                        exponent=2))
                only_right_consistency_loc_loss_h = torch.mean(
                    torch.pow(
                        flip_fixmatch_loc_sampled_interpolation[:, 3] - flip_fixmatch_loc_sampled[:, 3].detach(),
                        exponent=2))

                only_right_consistency_loc_loss = torch.div(
                    only_right_consistency_loc_loss_x + only_right_consistency_loc_loss_y + only_right_consistency_loc_loss_w + only_right_consistency_loc_loss_h,
                    4)

            else:
                only_right_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
                only_right_consistency_loc_loss = Variable(torch.cuda.FloatTensor([0]))
                only_right_consistency_conf_loss = only_right_consistency_conf_loss.data[0]
                only_right_consistency_loc_loss = only_right_consistency_loc_loss.data[0]


            # consistency_loss = consistency_conf_loss  # consistency_loc_loss
            only_right_consistency_loss = only_right_consistency_conf_loss + only_right_consistency_loc_loss
        #            only_right_consistency_loss = only_right_consistency_conf_loss

            fixmatch_loss = only_left_consistency_loss + only_right_consistency_loss


            return output, consistency_loss, interpolation_consistency_conf_loss, fixmatch_loss
            #return output, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



#    def moving_average(self):
#        ema_factor = min(1 - 1 / (self.global_step+1), self.ema_factor)
#        for emp_p, p in zip(self.vgg_t.parameters(), self.vgg.parameters()):
#            emp_p.data = ema_factor * emp_p.data + (1 - ema_factor) * p.data
#        for emp_p, p in zip(self.extras_t.parameters(), self.extras.parameters()):
#            emp_p.data = ema_factor * emp_p.data + (1 - ema_factor) * p.data
#        for emp_p, p in zip(self.loc_t.parameters(), self.loc.parameters()):
#            emp_p.data = ema_factor * emp_p.data + (1 - ema_factor) * p.data
#        for emp_p, p in zip(self.conf_t.parameters(), self.conf.parameters()):
#            emp_p.data = ema_factor * emp_p.data + (1 - ema_factor) * p.data



# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            elif v=='K':
                layers += [nn.Conv2d(in_channels, 256,
                           kernel_size=4, stride=1, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers



def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'K'],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 6, 4, 4],
}

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]

class GaussianNoise(nn.Module):
    def __init__(self, batch_size, input_size=(3, 300, 300), mean=0, std=0.15):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size, ) + input_size
        self.noise = Variable(torch.zeros(self.shape).cuda())
        self.mean = mean
        self.std = std

    def forward(self, x):
        self.noise.data.normal_(self.mean, std=self.std)
        if x.size(0) == self.noise.size(0):
            return x + self.noise
        else:
            #print('---- Noise Size ')
            return x + self.noise[:x.size(0)]


def build_ssd_con(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    # if size != 300:
    #     print("ERROR: You specified size " + repr(size) + ". However, " +
    #           "currently only SSD300 (size=300) is supported!")
    #     return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD_CON(phase, size, base_, extras_, head_, num_classes)


