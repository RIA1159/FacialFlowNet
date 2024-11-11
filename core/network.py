import torch
import torch.nn as nn
import torch.nn.functional as F

from update import GMAUpdateBlock
from extractor import BasicEncoder
from corr import CorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from gma import Attention, Aggregate
from dad_3dhead_utils.dad_3dheads_encoder import get_encoder, dad_3dhead_encoder

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        if stride != 1 or in_channels != out_channels:
            self.adjustment = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.adjustment = nn.Identity()
        
        self.requires_grad_(False)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.adjustment(residual)

        out += residual
        out = self.relu(out)

        return out


class RAFTGMA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if args.frozen:
            grad_flag = True
        else:
            grad_flag = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout, no_grad = grad_flag)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout, no_grad = grad_flag)
        self.update_block_head = GMAUpdateBlock(self.args, hidden_dim=hdim)
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim, no_grad = grad_flag)
        self.att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim, no_grad = grad_flag)


        self.dad_encoder = get_encoder()
        self.change_dim = nn.Conv2d(512, 256, kernel_size=1)
        self.res1 = ResidualBlock(512, 256)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            # cnet = self.cnet(image1)


            c = self.cnet(image1)

            d = dad_3dhead_encoder(self.dad_encoder,image1)
            d = self.change_dim(d)

            cnet = torch.cat((c,d),dim=1)
            cnet = self.res1(cnet)


            net, inp = torch.split(cnet, [hdim, cdim], dim=1)

            net = torch.tanh(net)
            inp = torch.relu(inp)
            # attention, att_c, att_p = self.att(inp)
            attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init



        facial_predictions = []
        head_predictions = []


        coords1_f = coords1
        coords1_h = coords1

        net_f = net
        net_h = net

        inp_f = inp
        inp_h = inp

        for itr in range(iters):
            coords1_f = coords1_f.detach()
            corr_f = corr_fn(coords1_f)  # index correlation volume

            coords1_h = coords1_h.detach()
            corr_h = corr_fn(coords1_h) 


            flow_f = coords1_f - coords0
            flow_h = coords1_h - coords0

            with autocast(enabled=self.args.mixed_precision):
                
                net_f, up_mask_f, delta_flow_f = self.update_block(net_f, inp_f, corr_f, flow_f, attention)

                net_h, up_mask_h, delta_flow_h = self.update_block_head(net_h, inp_h, corr_h, flow_h, attention)

            
            # F(t+1) = F(t) + \Delta(t)
            coords1_f = coords1_f + delta_flow_f
            
            coords1_h = coords1_h + delta_flow_h

            # upsample predictions
            if up_mask_f is None:
                flow_up_f = upflow8(coords1_f - coords0)
    
            else:
                flow_up_f = self.upsample_flow(coords1_f - coords0, up_mask_f)


            if up_mask_h is None:
                flow_up_h = upflow8(coords1_h - coords0)
    
            else:
                flow_up_h = self.upsample_flow(coords1_h - coords0, up_mask_h)

            
            facial_predictions.append(flow_up_f)
            head_predictions.append(flow_up_h)


        if test_mode:
            return coords1 - coords0, flow_up_f, flow_up_h, flow_up_f - flow_up_h 

        return facial_predictions, head_predictions
