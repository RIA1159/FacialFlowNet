<!--
 * @Date: 2024-07-24 14:27:51
 * @LastEditors: ljz 
 * @LastEditTime: 2024-11-11 14:34:47
 * @FilePath: \lab2d:\desktop\study\FacialFlowNet\README.md
 * @Description: 
 * 
 * Copyright (c) 2024 by Fudan University/Shanghai Key Laboratory of Intelligent Information Processing, All Rights Reserved. 
-->

# FacialFlowNet
Official release of FacialFlowNet: 
[Advancing Facial Optical Flow Estimation with a Diverse Dataset and a Decomposed Model](https://dl.acm.org/doi/10.1145/3664647.3680921)
ACMMM2024.
Jianzhi Lu, Ruian He, Shili Zhou, Weimin Tan, Bo Yan.
## FacialFlowNet Dataset
![FlowPipeline](./assets/flowpipeline.png)
You can download the FacialFlowNet dataset from [here](https://pan.baidu.com/s/1u9fQsGdqhjqDN6jVhXxNrA) with the extraction code `c2z2`, and extract the compressed file to the following path:
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

Fudan, Shanghai, China

## DecFlow
![DecFlow](./assets/decflow.png)
### Environments
You will have to choose cudatoolkit version to match your compute environment. The code is tested on PyTorch 1.10.0 and cuda 11.8, but other versions might also work.
```Shell
conda create --name decflow python==3.8
conda activate decflow
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt

```

### Demos
```Shell
python get_flow.py
```

### Train
```Shell
# stage1 train both flow decoder heads
python train.py --name decflow-facialflownet-part1 --stage facialflow --validation facialflow --output checkpoints --restore_ckpt checkpoints/gma-sintel.pth  --num_steps 10000 --lr 0.000125 --image_size 480 480 --wdecay 0.00001 --gamma 0.85 --gpus 0 --batch_size 6 --val_freq 1000 --print_freq 100 --mixed_precision
# stage2 frozen facialflow decoder and train headflow decoder
python train.py --name decflow --stage facialflow --validation facialflow  --output checkpoints --restore_ckpt checkpoints/decflow-facialflownet-part1.pth  --num_steps 10000 --lr 0.000125 --image_size 480 480 --wdecay 0.00001 --gamma 0.85 --gpus 0 --batch_size 6 --val_freq 1000 --print_freq 100 --mixed_precision --frozen
```

### Evaluate
```Shell
python evaluate.py --model ./checkpoints/decflow-facialflownet.pth --dataset facialflow
```


## FacialFlowNet Dataset
![FlowPipeline](./assets/flowpipeline.png)
You can download the FacialFlowNet dataset from [here](https://pan.baidu.com/s/1u9fQsGdqhjqDN6jVhXxNrA) with the extraction code `c2z2`, and extract the compressed file to the following path:
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
If you use the FacialFlowNet Dataset and/or DecFlow - implicitly or explicitly - for your research projects, please cite the following paper:
```Shell
@inproceedings{lu2024facialflownet,
  title={FacialFlowNet: Advancing Facial Optical Flow Estimation with a Diverse Dataset and a Decomposed Model},
  author={Lu, Jianzhi and He, Ruian and Zhou, Shili and Tan, Weimin and Yan, Bo},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={2194--2203},
  year={2024}
}
```