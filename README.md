# LightMFF
Code for "LightMFF: Ultra-lightweight real-time multi-focus image fusion network via focus property and edge detection"
## Download trained model
```
https://pan.baidu.com/s/1mTouAcH-cGMr6VgDCqcWaw?pwd=cite
```
## Inference
```
python predict.py --model_path [model path] --test_dataset_path [dataset path] --GPU_parallelism [True/False]
```
Note: There should be two folders under the dataset path named A and B, which store the corresponding image pairs 1.jpg, 2.jpg...

## Creating training data
```
python tools/make_datasets_DUTS.py --mode TR --data_root [data_root] --out_dir_name [DUTS_MFF_NEW_256] #Create training set
python tools/make_datasets_DUTS.py --mode TE --data_root [data_root] --out_dir_name [DUTS_MFF_NEW_256] #Create validation set
```
Note: You should download DUTS dataset fisrt. There should be three folders under [data_root], namely DUTS-OURS, DUTS-TR, DUTS-TE. (Download：http://saliencydetection.net/duts/)

## Train
Note: If you want to train LightMFF yourself, please comment out all the codes in the original training script that visualize the fusion results during the training process. Because the original code will save the fusion results of Lytro, MFI-WHU, and MFFW during the training process, you need to prepare these data or comment out all related codes.

## Cite
If our work is helpful to you, please cite the following article:
```
@article{xie2024swinmff,
  title={SwinMFF: toward high-fidelity end-to-end multi-focus image fusion via swin transformer-based network},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and He, Shuangyan and Zhou, Sangjun},
  journal={The Visual Computer},
  pages={1--24},
  year={2024},
  publisher={Springer}
}
```
