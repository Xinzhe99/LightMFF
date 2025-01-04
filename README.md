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
