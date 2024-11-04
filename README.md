<!--
 * @Date: 2024-07-24 14:27:51
 * @LastEditors: ljz 
 * @LastEditTime: 2024-11-04 11:02:08
 * @FilePath: \X-Gaussiand:\desktop\study\FacialFlowNet\README.md
 * @Description: 
 * 
 * Copyright (c) 2024 by Fudan University/Shanghai Key Laboratory of Intelligent Information Processing, All Rights Reserved. 
-->
# FacialFlowNet
Official release of FacialFlowNet: Advancing Facial Optical Flow Estimation with a Diverse Dataset and a Decomposed Model (ACMMM2024)

## DecFlow
![DecFlow](./assets/decflow.png)
```Shell
```
Coming soon ......

## Demos



## FacialFlowNet Dataset
![FlowPipeline](./assets/flowpipeline.png)
You can download the FacialFlowNet dataset from [here](https://pan.baidu.com/s/1qNSZ_5hjIr3_5srxaavR7w) with the extraction code `jjm3`, and extract the compressed file to the following path:
```Shell
├── FacialFlowNet
    ├── image
        ├── facial
            ├── train 
            ├── test 
            ├── val
        ├── head
            ├── ...
    ├── flow
        ├── facial
            ├── train 
            ├── test 
            ├── val
        ├── head
            ├── ...
    ├── mask
        ├── ...
```

## Acknowledgement
Parts of code are adapted from the following repositories. We thank the authors for their great contribution to the community:
* [RAFT](https://github.com/princeton-vl/RAFT)
* [GMA](https://github.com/zacjiang/GMA)
* [DAD-3DHeads](https://github.com/PinataFarms/DAD-3DHeads)

## Citation
```Shell
@inproceedings{lu2024facialflownet,
  title={FacialFlowNet: Advancing Facial Optical Flow Estimation with a Diverse Dataset and a Decomposed Model},
  author={Lu, Jianzhi and He, Ruian and Zhou, Shili and Tan, Weimin and Yan, Bo},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={2194--2203},
  year={2024}
}
```