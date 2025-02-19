# LightMFF 🚀

[![Paper](https://img.shields.io/badge/Paper-The%20Visual%20Computer-blue)](https://link.springer.com/article/10.1007/s00371-024-03327-0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![stars](https://img.shields.io/github/stars/Xinzhe99/LightMFF?style=social)](https://github.com/Xinzhe99/LightMFF)

> 🔔 **Note**: Our paper is currently under review. The complete paper and citation information will be available upon acceptance.

<div align="center">
    <img src="assets/framework.png" width="800"/>
</div>

## 📝 Introduction

This is the official implementation of the paper "LightMFF: A Real-time Ultra-lightweight Multi-focus Image Fusion Network Based on Focus Property and Edge Detection". LightMFF is an ultra-lightweight multi-focus image fusion network with the following features:

- 🚄 **Real-time Performance**: Only 0.02 seconds processing time per image pair
- 🎯 **Ultra-lightweight**: Only 0.02M parameters and 0.06G FLOPs
- 🏆 **High Performance**: Surpasses existing methods across standard fusion quality metrics
- 💡 **Innovative Approach**: Reformulates the fusion problem from a classification perspective into a refinement approach

## 🛠️ Installation

```bash
git clone https://github.com/Xinzhe99/LightMFF.git
cd LightMFF
pip install -r requirements.txt
```

## 📥 Pre-trained Model

Download the pre-trained model from:
```bash
https://pan.baidu.com/s/1mTouAcH-cGMr6VgDCqcWaw?pwd=cite
```

## 🚀 Quick Start

### Inference
```bash
python predict.py --model_path [model path] --test_dataset_path [dataset path] --GPU_parallelism [True/False]
```
Note: There should be two folders under the dataset path named A and B, which store the corresponding image pairs (1.jpg, 2.jpg...)

### Creating Training Data
```bash
# Create training set
python tools/make_datasets_DUTS.py --mode TR --data_root [data_root] --out_dir_name [DUTS_MFF_NEW_256]
# Create validation set
python tools/make_datasets_DUTS.py --mode TE --data_root [data_root] --out_dir_name [DUTS_MFF_NEW_256]
```
Note: You should download the DUTS dataset first. There should be three folders under [data_root]: DUTS-OURS, DUTS-TR, DUTS-TE.
- Download: http://saliencydetection.net/duts/

## 📊 Results

<div align="center">
    <img src="assets/result.png" width="800"/>
</div>

### Quantitative Results on Lytro Dataset

| Method | Q<sup>AB/F</sup>↑ | Q<sub>MI</sub>↑ | Q<sup>P</sup>↑ | Q<sub>W</sub>↑ | Q<sub>E</sub>↑ | Q<sub>CB</sub>↑ |
|--------|-------------------|------------------|-----------------|-----------------|-----------------|------------------|
| **Methods based on image transform domain** |||||||
| DWT | 0.6850 | 0.8677 | 0.2878 | 0.8977 | 0.8356 | 0.6117 |
| DTCWT | 0.6929 | 0.8992 | 0.2925 | 0.8987 | 0.8408 | 0.6234 |
| NSCT | 0.6901 | 0.9039 | 0.2928 | 0.9030 | 0.8413 | 0.6174 |
| CVT | 0.7243 | 0.8968 | 0.7966 | 0.9388 | 0.9023 | 0.7277 |
| DCT | 0.7031 | 0.9383 | 0.7825 | 0.9093 | 0.8073 | 0.6624 |
| GFF | 0.6998 | 1.0020 | 0.2952 | 0.8982 | 0.8351 | 0.6518 |
| SR | 0.6944 | 1.0003 | 0.2921 | 0.8984 | 0.8309 | 0.6406 |
| ASR | 0.6951 | 1.0024 | 0.2926 | 0.8986 | 0.8308 | 0.6413 |
| MWGF | 0.7037 | 1.0545 | 0.3176 | 0.8913 | 0.8107 | 0.6758 |
| ICA | 0.6766 | 0.8687 | 0.2964 | 0.9084 | 0.8219 | 0.5956 |
| NSCT-SR | 0.6995 | 1.0189 | 0.2949 | 0.9000 | 0.8385 | 0.6501 |
| **Methods based on image spatial domain** |||||||
| SSSDI | 0.6966 | 1.0351 | 0.2915 | 0.8961 | 0.8279 | 0.6558 |
| QUADTREE | 0.7027 | 1.0630 | 0.2940 | 0.8962 | 0.8265 | 0.6681 |
| DSIFT | 0.7046 | 1.0642 | 0.2954 | 0.8977 | 0.8354 | 0.6675 |
| SRCF | 0.7036 | 1.0590 | 0.2954 | 0.8978 | 0.8369 | 0.6669 |
| GFDF | 0.7049 | 1.0524 | 0.2974 | 0.8989 | 0.8399 | 0.6657 |
| BRW | 0.7040 | 1.0516 | 0.2964 | 0.8984 | 0.8371 | 0.6650 |
| MISF | 0.6984 | 1.0391 | 0.2945 | 0.8929 | 0.8063 | 0.6607 |
| MDLSR_RFM | 0.7518 | 1.1233 | 0.8294 | 0.9394 | 0.9021 | *0.8064* |
| **End-to-end methods based on deep learning** |||||||
| IFCNN-MAX | 0.6784 | 0.8863 | 0.2962 | 0.9013 | 0.8324 | 0.5986 |
| U2Fusion | 0.6190 | 0.7803 | 0.2994 | 0.8909 | 0.7108 | 0.5159 |
| SDNet | 0.6441 | 0.8464 | 0.3072 | 0.8934 | 0.7464 | 0.5739 |
| MFF-GAN | 0.6222 | 0.7930 | 0.2840 | 0.8887 | 0.7660 | 0.5399 |
| SwinFusion | 0.6597 | 0.8404 | 0.3117 | 0.9011 | 0.7460 | 0.5745 |
| MUFusion | 0.6614 | 0.8030 | 0.7160 | 0.9089 | 0.8036 | 0.6758 |
| FusionDiff | 0.6744 | 0.8692 | 0.2900 | 0.8980 | 0.8261 | 0.5747 |
| SwinMFF | 0.7321 | 0.9605 | 0.8222 | 0.9390 | 0.8986 | 0.7543 |
| DDBFusion | 0.5026 | 0.8152 | 0.5610 | 0.8391 | 0.4947 | 0.6057 |
| **Decision map-based methods using deep learning** |||||||
| CNN | 0.7019 | 1.0424 | 0.2968 | 0.8976 | 0.8311 | 0.6628 |
| ECNN | 0.7030 | 1.0723 | 0.2945 | 0.8946 | 0.8169 | 0.6698 |
| DRPL | 0.7574 | 1.1405 | 0.8435 | 0.9397 | *0.9060* | 0.8035 |
| SESF | 0.7031 | 1.0524 | 0.2950 | 0.8977 | 0.8353 | 0.6657 |
| MFIF-GAN | 0.7029 | 1.0618 | 0.2960 | 0.8982 | 0.8395 | 0.6660 |
| MSFIN | 0.7045 | 1.0601 | 0.2973 | 0.8990 | 0.8436 | 0.6664 |
| GACN | *0.7581* | *1.1334* | *0.8443* | **0.9405** | 0.9013 | 0.8024 |
| ZMFF | 0.6635 | 0.8694 | 0.2890 | 0.8951 | 0.8253 | 0.6136 |
| **LightMFF** | **0.7588** | **1.1462** | **0.8450** | *0.9400* | **0.9061** | **0.8067** |

### Computational Efficiency Comparison

| Method | Model Size (M) | FLOPs (G) | Time (s) | Device |
|--------|---------------|------------|-----------|---------|
| End-to-end methods based on deep learning | | | | |
| IFCNN-MAX | 0.08 | 8.54 | 0.09 | GPU |
| U2Fusion | 0.66 | 86.40 | 0.16 | CPU |
| SDNet | 0.07 | 8.81 | 0.10 | CPU |
| MFF-GAN | 0.05 | 3.08 | 0.06 | CPU |
| SwinFusion | 0.93 | 63.73 | 1.79 | GPU |
| MUFusion | 2.16 | 24.07 | 0.72 | GPU |
| FusionDiff | 26.90 | 58.13 | 81.47 | GPU |
| SwinMFF | 41.25 | 22.38 | 0.46 | GPU |
| DDBFusion | 10.92 | 184.93 | 1.69 | GPU |
| Decision map-based methods using deep learning | | | | |
| CNN | 8.76 | 142.23 | 0.06 | GPU |
| ECNN | 1.59 | 14.93 | 125.53 | GPU |
| DRPL | 1.07 | 140.49 | 0.22 | GPU |
| SESF | 0.07 | 4.90 | 0.26 | GPU |
| MFIF-GAN | 3.82 | 693.03 | 0.32 | GPU |
| MSFIN | 4.59 | 26.76 | 1.10 | GPU |
| GACN | 0.07 | 10.89 | 0.16 | GPU |
| ZMFF | 6.33 | 464.53 | 165.38 | GPU |
| **LightMFF** | **0.02** | **0.06** | **0.02** | GPU |
| Reduction (%) | 60.00 | 98.05 | 66.67 | - |

## ⚠️ Training Notes

If you want to train LightMFF yourself, please note:
- You need to comment out all codes in the original training script that visualize fusion results
- The original code will save fusion results of Lytro, MFI-WHU, and MFFW during the training process
- You need to prepare these datasets or comment out all related codes

## 📚 Citation

If our work is helpful to you, please cite the following paper:

```bibtex
@article{anonymous2024lightmff,
  title={LightMFF: A Real-time Ultra-lightweight Multi-focus Image Fusion Network Based on Focus Property and Edge Detection},
  author={Anonymous},
  journal={Under Review},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Issues and contributions are welcome! Feel free to submit issues or pull requests.

## 📧 Contact

If you have any questions, please feel free to contact us through:
- Email: [anonymous@example.com](mailto:anonymous@example.com)
- GitHub Issues
