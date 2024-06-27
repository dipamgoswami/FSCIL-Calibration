# FSCIL-Calibration [![Paper](https://img.shields.io/badge/arXiv-2210.07207-brightgreen)](https://arxiv.org/pdf/2404.06622.pdf)
## Code for CLVision workshop (CVPR 2024) paper - Calibrating Higher-Order Statistics for Few-Shot Class-Incremental Learning with Pre-trained Vision Transformers

## Abstract
Few-shot class-incremental learning (FSCIL) aims to adapt the model to new classes from very few data (5 samples) without forgetting the previously learned classes. Recent works in many-shot CIL (MSCIL) (using all available training data) exploited pre-trained models to reduce forgetting and achieve better plasticity. In a similar fashion, we use ViT models pre-trained on large-scale datasets for few-shot settings, which face the critical issue of low plasticity. FSCIL methods start with a many-shot first task to learn a very good feature extractor and then move to the few-shot setting from the second task onwards. While the focus of most recent studies is on how to learn the many-shot first task so that the model generalizes to all future few-shot tasks, we explore in this work how to better model the few-shot data using pre-trained models, irrespective of how the first task is trained. Inspired by recent works in MSCIL, we explore how using higher-order feature statistics can influence the classification of few-shot classes. We identify the main challenge of obtaining a good covariance matrix from few-shot data and propose to calibrate the covariance matrix for new classes based on semantic similarity to the many-shot base classes. Using the calibrated feature statistics in combination with existing methods significantly improves few-shot continual classification on several FSCIL benchmarks.

<img src="https://github.com/dipamgoswami/FSCIL-Calibration/blob/main/figs/FSCIL_calibration.png" width="100%" height="100%">

```
@inproceedings{goswami2024calibrating,
  title={Calibrating Higher-Order Statistics for Few-Shot Class-Incremental Learning with Pre-trained Vision Transformers},
  author={Goswami, Dipam and Twardowski, Bartłomiej and van de Weijer, Joost},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2024}
}
```

## Run experiments

1. Run:
```
python main.py --config=./exps/[MODEL NAME].json
```
- Tune the model at first session with ViT adapter, and then classify with NCM/TEEN/FeCAM/RanPAC. 
- Update the `exps/adam_adapter.json` to perform classification with NCM or TEEN or FeCAM.
- Use `exps/ranpac.json` to use RanPAC.
- To run TEEN, set fecam to false and calibartion to true.
- To run NCM, set calibration and fecam to false.

2. Hyper-parameters: You can edit the algorithm-speciifc hyperparameters in their respective json files.

## Acknowledgements

The code is based on the framework from [PILOT](https://github.com/sun-hailong/LAMDA-PILOT). We also refer to code from [TEEN](https://github.com/wangkiw/TEEN) and [FeCAM](https://github.com/dipamgoswami/FeCAM).
