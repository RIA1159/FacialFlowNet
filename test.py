'''
Date: 2024-11-10 17:02:25
LastEditors: ljz 
LastEditTime: 2024-11-10 19:36:04
FilePath: \FacialFlowNet\test.py
Description: 

Copyright (c) 2024 by Fudan University/Shanghai Key Laboratory of Intelligent Information Processing, All Rights Reserved. 
'''
import numpy as np
import os
import cv2

import os.path as osp
from glob import glob



split_list = [5,10,15,20]

split = 5
split_list = [10,15,20]
for split in split_list:
    img_list = []

    sub_list = []
    exp_flow_list = []
    head_flow_list = []
    exp_root = "datasets/ExpFlow"
    head_root = "datasets/HeadFlow1"


    exp_flow_root = osp.join(exp_root, "test", "flow")
    head_flow_root = osp.join(head_root, "test",  "flow")

    exp_image_root = osp.join(exp_root, "test")
    head_image_root = osp.join("datasets/HeadFlow", "test")


    mask_root = osp.join(exp_root, "test", "mask")

    for emotion in os.listdir(exp_flow_root):
        sub_list = os.listdir(osp.join(exp_flow_root, emotion))
        for sub in sub_list:

            image_list = sorted(glob(osp.join(exp_image_root, emotion, sub, "*.jpg")))
            if len(image_list) != split :
                continue


            for i in range(len(image_list) - 1):
                img_list += [ [image_list[i], image_list[i+1]] ]
            sub_list.append("{}_{}".format(emotion, sub))

            #     self.image_list += [ [image_list[i], image_list[i+1]] ]
            #     self.extra_info += [ ("{}_{}".format(emotion, sub), i) ]

            # self.mask_list += sorted(glob(osp.join(mask_root, emotion, sub, "*.npy")))

            exp_flow_list += sorted(glob(osp.join(exp_flow_root, emotion, sub, "*.flo")))
            head_flow_list += sorted(glob(osp.join(head_flow_root, emotion, sub, "*.flo")))


    # for img in img_list:
    #     print(img)

    for flow in exp_flow_list:
        print(flow)
    
    break