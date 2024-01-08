# Detecting and Recognizing Saudi Car License Plates Using Surveillance Cameras and Deep Learning for Smart Parking Systems 

## Abstract
Smart parking systems play a vital role in enhancing the efficiency and sustainability of smart cities. However, most existing systems depend on sensors to monitor the occupancy of parking spaces, which entail high installation and maintenance costs and limited functionality in tracking vehicle movement within the car park. To address these challenges, we propose a novel learning-based approach for smart parking systems that utilizes existing surveillance cameras within the car park and leverages a self-collected dataset of Saudi car license plates. We train and fine-tune state-of-the-art YOLO series (i.e., YOLOv5, YOLOv7, and YOLOv8) for accurate Saudi car license plate detection and character segmentation. Moreover, we propose a new CNN architecture for improved license plate character recognition which outperforms the state-of-the-art frameworks. Additionally, we develop a web-based dashboard for real-time visualization and statistical analysis. We evaluate our approach on various metrics and demonstrate its effectiveness and efficiency in facilitating car park management.


<summary>Image Preprocessing </summary>

```
python image_preprocessing.py --images_dir  --annotations_dir  --output_dir path/to/output --names class1,class2,class3 --number_of_classes 3 --split_ratio 0.7 0.2 0.1
```


<summary>Traing Yolov7 models </summary>

```
python train.py  --data data/data.yaml --img 640  --cfg cfg/training/yolov7x.yaml --weights 'yolov7x.pt' --name yolov7 
```


<summary>Traing Yolov5 models </summary>

```
python train.py --data data/data.yaml --epochs 50 --weights 'yolov5x.pt' --cfg yolov5x.yaml  
```


<summary>Traing Yolov8 models </summary>

```
python train.py --data_path data.yaml --model_path yolov8x.pt --number_of_epochs 
```

<summary>OCR Testing </summary>

```
python OCR_train.py --train_dir  --val_dir  --test_dir  --img_width 50 --img_height 50 --batch_size 16 --epochs 100
```


<summary>System Testing </summary>

```
python system_testing.py --image_test_path  --image_save_path  --y_true_path  --lp_model_path  --text_model_path  --ocr_model_path 
```



### Models Weight
https://drive.google.com/drive/folders/1MchFtNTKPsoVuzCtncD8-ooWammW5swp?usp=sharing


