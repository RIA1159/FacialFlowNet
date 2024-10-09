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
You can download the FacialFlowNet dataset from [here](https://pan.baidu.com/s/1u9fQsGdqhjqDN6jVhXxNrA) with the extraction code `xpxw`, and extract the compressed file to the following path:
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
```