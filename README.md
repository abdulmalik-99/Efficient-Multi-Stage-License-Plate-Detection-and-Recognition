# Efficient Multi-Stage License Plate Detection and Recognition Using YOLOv8 and CNN for Smart Parking Systems

## Abstract
Smart parking systems play a vital role in enhancing the efficiency and sustainability of smart cities. However, most existing systems depend on sensors to monitor the occupancy of parking spaces, which entail high installation and maintenance costs and limited functionality in tracking vehicle movement within the car park. To address these challenges, we propose a multi-stage learning-based approach that leverages existing surveillance cameras within the car park and a self-collected dataset of Saudi license plates. The approach combines YOLOv5 for license plate detection, YOLOv8 for character detection, and a new CNN architecture for improved character recognition. We show that our approach outperforms the single-stage approach, achieving an overall accuracy of 96.1% versus 83.9% of the single stage approach. The approach is also integrated into a web-based dashboard for real-time visualization and statistical analysis of car park occupancy and vehicle movement, with an acceptable time efficiency. Our work demonstrates how existing technology can be leveraged to improve the efficiency and sustainability of smart cities.


<summary>Image Preprocessing </summary>

```
python image_preprocessing.py --images_dir  --annotations_dir  --output_dir  --names class1,class2,class3 --number_of_classes 3 
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

<summary>OCR Training </summary>

```
python OCR_train.py --train_dir  --val_dir  --test_dir  --img_width 50 --img_height 50 --batch_size 16 --epochs 100
```


<summary>System Testing </summary>

```
python system_testing.py --image_test_path  --image_save_path  --y_true_path  --lp_model_path  --text_model_path  --ocr_model_path 
```



### Models Weight
https://drive.google.com/drive/folders/1MchFtNTKPsoVuzCtncD8-ooWammW5swp?usp=sharing

### How to Cite This Work:
Mejdl Safran, Abdulmalik Alajmi, Sultan Alfarhood, "Efficient Multistage License Plate Detection and Recognition Using YOLOv8 and CNN for Smart Parking Systems", Journal of Sensors, vol. 2024, Article ID 4917097, 18 pages, 2024. https://doi.org/10.1155/2024/4917097


