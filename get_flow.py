import sys
sys.path.append('core')

from PIL import Image
import argparse
import cv2
import numpy as np
import torch

from network import RAFTGMA

from utils.utils import InputPadder, forward_interpolate
from core.utils.flow_viz import flow_to_image
from core.utils.frame_utils import writeFlow





def read_img(img_path):
    img = np.array(Image.open(img_path)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()

def caculate_flow(model, image1, image2):
    model.eval()
    with torch.no_grad():

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up, head_flow, exp_flow = model(image1, image2, iters=24, test_mode=True)
        

        facialflow = flow_up[0].permute(1,2,0).cpu().numpy()
        headflow = head_flow[0].permute(1,2,0).cpu().numpy()
        expflow = exp_flow[0].permute(1,2,0).cpu().numpy()


    return facialflow, headflow, expflow


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default="./checkpoints/decflow-facialflownet.pth")
    parser.add_argument('--model_name', help="define model name", default="DecFlow")
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--frozen', default=False, action='store_true',
                        help='frozen facialflow head')
    
    args = parser.parse_args()


    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model))
    print(f"Loaded checkpoint at {args.model}")

    model = model.module
    model.cuda()
    model.eval()

    
    img1 = read_img("./demo/demo4/0001.jpg")
    img2 = read_img("./demo/demo4/0002.jpg")


    facialflow, headflow, expflow = caculate_flow(model, img1, img2)


    facialflow_v = flow_to_image(facialflow)
    headflow_v = flow_to_image(headflow)
    expflow_v = flow_to_image(expflow)

    res = np.concatenate([facialflow_v, headflow_v, expflow_v], axis=1)
    cv2.imwrite("./demo/demo4/flow_viz.jpg", res)

