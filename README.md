# LightMFF 🚀

[![Paper](https://img.shields.io/badge/Paper-The%20Visual%20Computer-blue)](https://link.springer.com/article/10.1007/s00371-024-03327-0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![stars](https://img.shields.io/github/stars/Xinzhe99/LightMFF?style=social)](https://github.com/Xinzhe99/LightMFF)

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
    <img src="assets/results.png" width="800"/>
</div>

## ⚠️ Training Notes

If you want to train LightMFF yourself, please note:
- You need to comment out all codes in the original training script that visualize fusion results
- The original code will save fusion results of Lytro, MFI-WHU, and MFFW during the training process
- You need to prepare these datasets or comment out all related codes

## 📚 Citation

If our work is helpful to you, please cite the following paper:

```bibtex
@article{xie2024swinmff,
  title={SwinMFF: toward high-fidelity end-to-end multi-focus image fusion via swin transformer-based network},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and He, Shuangyan and Zhou, Sangjun},
  journal={The Visual Computer},
  pages={1--24},
  year={2024},
  publisher={Springer}
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Issues and contributions are welcome! Feel free to submit issues or pull requests.

## 📧 Contact

If you have any questions, please feel free to contact us through:
- Email: [guobuyuwork@163.com](mailto:guobuyuwork@163.com)
- GitHub Issues
