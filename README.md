## A Discriminatively Learned CNN Embedding for Person Re-identification (in Pytorch)

![](https://github.com/layumi/2016_person_re-ID/blob/master/paper.jpg)

In this package, we provide our training and testing code written in [pytorch](https://pytorch.org/) for the paper [A Discriminatively Learned CNN Embedding for Person Re-identification](https://arxiv.org/abs/1611.05666).

Compared with the original version, I do some modification:
- I use `x*y` instead of `(x-y)^2` as `Square Layer`. (We do not need to worry about the scale of `x` and `y`.)
- I add the bottle-neck fully-connected layer for classification. I use the `512-dim` fully-connected feature as pedestrian descriptor.
- I tune some hyperparameters. 

We arrived **Rank@1=88.66%, mAP=72.58%** with ResNet-50. The code is largely borrowed from my another repo [strong Pytorch baseline](https://github.com/layumi/Person_reID_baseline_pytorch) .
Here we provide hyperparameters and architectures, that were used to generate the result. Some of them (i.e. learning rate) are far from optimal. Do not hesitate to change them and see the effect. 
	
Any suggestion is welcomed.

**This code is ONLY released for academic use.**

## Resources
* [Zhedong Zheng](https://github.com/layumi) The original [Matconvnet](https://github.com/layumi/2016_person_re-ID) version in the paper. ([![GitHub stars](https://img.shields.io/github/stars/layumi/2016_person_re-ID.svg?style=flat&label=Star)](https://github.com/layumi/2016_person_re-ID))
* [Weihang Chen](https://github.com/ahangchen) also realizes our paper in [Keras](https://github.com/ahangchen/rank-reid/tree/release). ([![GitHub stars](https://img.shields.io/github/stars/ahangchen/rank-reid.svg?style=flat&label=Star)](https://github.com/ahangchen/rank-reid/tree/release))
* [Xuanyi Dong](https://github.com/D-X-Y) also realizes our paper in [Caffe](https://github.com/D-X-Y/caffe-reid). ([![GitHub stars](https://img.shields.io/github/stars/D-X-Y/caffe-reid.svg?style=flat&label=Star)](https://github.com/D-X-Y/caffe-reid))
* [Zhun Zhong](https://github.com/zhunzhong07/IDE-baseline-Market-1501) provides a extensive [Caffe baseline code](https://github.com/zhunzhong07/IDE-baseline-Market-1501). You may check it. ([![GitHub stars](https://img.shields.io/github/stars/zhunzhong07/IDE-baseline-Market-1501.svg?style=flat&label=Star)](https://github.com/zhunzhong07/IDE-baseline-Market-1501))
* [Zhedong Zheng](https://github.com/layumi) provides a [strong Pytorch baseline](https://github.com/layumi/Person_reID_baseline_pytorch) ([![GitHub stars](https://img.shields.io/github/stars/layumi/Person_reID_baseline_pytorch.svg?style=flat&label=Star)](https://github.com/layumi/Person_reID_baseline_pytorch))

## Model Structure
You may learn more from `model.py`.  

## Prerequisites

- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+

**(Some reports found that updating numpy can arrive the right accuracy. If you only get 50~80 Top1 Accuracy, just try it.)**
We have successfully run the code based on numpy 1.12.1 and 1.13.1 .

## Getting started
### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
Because pytorch and torchvision are ongoing projects.

Here we noted that our code is tested based on Pytorch 0.3.0/0.4.0 and Torchvision 0.2.0.

## Dataset & Preparation
Download [Market1501 Dataset](http://www.liangzheng.org/Project/project_reid.html)

Preparation: Put the images with the same id in one folder. You may use 
```bash
python prepare.py
```
Remember to change the dataset path to your own path.

Futhermore, you also can test our code on [DukeMTMC-reID Dataset](https://github.com/layumi/DukeMTMC-reID_evaluation).
Our baseline code is not such high on DukeMTMC-reID **Rank@1=64.23%, mAP=43.92%**. Hyperparameters are need to be tuned.

To save trained model, we make a dir.
```bash
mkdir model 
```

## Train
Train a model by
```bash
python train_new.py --gpu_ids 0 --name ft_ResNet50 --alpha 1.0 --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of model.

`--data_dir` the path of the training data.

`--batchsize` batch size.

`--erasing_p` random erasing probability.

`--alpha` the weight of the verification loss.

Train a model with random erasing by
```bash
python train_new.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path --erasing_p 0.5
```

## Test
Use trained model to extract feature by
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --which_epoch 59
```
`--gpu_ids` which gpu to run.

`--name` the dir name of trained model.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.


## Evaluation
```bash
python evaluate.py
```
It will output Rank@1, Rank@5, Rank@10 and mAP results.
You may also try `evaluate_gpu.py` to conduct a faster evaluation with GPU.

For mAP calculation, you also can refer to the [C++ code for Oxford Building](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp). We use the triangle mAP calculation (consistent with the Market1501 original code).

## Citation
Please cite this paper in your publications if it helps your research:
```
@article{zheng2016discriminatively,
  title={A Discriminatively Learned CNN Embedding for Person Re-identification},
  author={Zheng, Zhedong and Zheng, Liang and Yang, Yi},
  doi={10.1145/3159171},
  journal={ACM Transactions on Multimedia Computing Communications and Applications},
  year={2017}
}
```

## Related Repos
1. [Pedestrian Alignment Network](https://github.com/layumi/Pedestrian_Alignment)
2. [2stream Person re-ID](https://github.com/layumi/2016_person_re-ID)
3. [Pedestrian GAN](https://github.com/layumi/Person-reID_GAN)
4. [Language Person Search](https://github.com/layumi/Image-Text-Embedding)
