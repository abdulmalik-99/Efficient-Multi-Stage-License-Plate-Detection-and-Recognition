import argparse
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from paper_utils import *
import tensorflow as tf 



dict_from_arr={0: '0',
 1: '1',
 2: '2',
 3: '3',
 4: '4',
 5: '5',
 6: '6',
 7: '7',
 8: '8',
 9: '9',
 10: 'A',
 11: 'B',
 12: 'D',
 13: 'E',
 14: 'G',
 15: 'H',
 16: 'J',
 17: 'K',
 18: 'L',
 19: 'N',
 20: 'R',
 21: 'S',
 22: 'T',
 23: 'U',
 24: 'V',
 25: 'X',
 26: 'Z'}


def load_license_plate_models(paths=[]):
    lp = YOLO(paths[0])
    nch = YOLO(paths[1])
    ocr=tf.keras.models.load_model(paths[2])

    return ocr,lp, nch



def process_images(image_test_path, image_save_path,ocr_model, y_true_path, model_lp, model_nch):
    
    texts=[]
    for i,item in enumerate(os.listdir(image_test_path)):
        image_path = os.path.join(image_test_path,item)

        # Load the image using OpenCV
        image = cv2.imread(image_path)



        # Perform the object detection
        results = lp_model(image, size=640)

        # Get the detected objects and their bounding boxes
        objects = results.pandas().xyxy[0]

        print(results)
        # Iterate over the detected objects
        for index, obj in objects.iterrows():
            label = obj['name']
            confidence = obj['confidence']
            bbox = obj[['xmin', 'ymin', 'xmax', 'ymax']].values

            # Crop the object from the original image
            cropped_image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            result=nch_model(cropped_image)
            res=list(result)[0]
            space_dict,list_xyxy,img_latters , image_num=num_latter(res.boxes,cropped_image.copy())
            

            text=''
            for n,lists in enumerate([image_num,img_latters]):
                for j in range(len(lists)):
                    x1,y1,x2,y2=lists[j]
                    new_size = (50, 50)
                    resized_image = cv2.resize(cropped_image.copy()[y1:y2,x1:x2], new_size)
                    reshaped_image = np.reshape(resized_image, (1,) + resized_image.shape)
                    p=ocr_model.predict(reshaped_image)
                    
                    if n == 1:
                        p=np.argmax(p.tolist()[0][10:])+10
                    elif n==0:
                        p=np.argmax(p.tolist()[0][:10])
                    text=text+str(dict_from_arr[p])
                    texts.append(text)

            img = cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            plt.imsave(os.path.join(image_save_path, item + '.png'), img)

    with open(y_true_path, 'r') as file:
        file_text = file.readlines()

    y_true = pd.Series(file_text).str.replace('\n', '')
    y_pred = pd.Series(texts)

    accuracy = (y_true == y_pred).sum() / len(file_text)
    print("Test accuracy:", accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="License Plate and Number Character Recognition Script")
    parser.add_argument("--image_test_path", type=str, help="Path to the test images")
    parser.add_argument("--image_save_path", type=str, help="Path to save the output images")
    parser.add_argument("--y_true_path", type=str, help="Path to the ground truth file")
    parser.add_argument("--lp_model_path", type=str, help="Path to the License Plate model")
    parser.add_argument("--nch_model_path", type=str, help="Path to the Number Character Detection model")
    parser.add_argument("--ocr_model_path", type=str, help="Path to the Number Character Recognition model")
    args = parser.parse_args()

    image_test_path = args.image_test_path
    image_save_path = args.image_save_path
    y_true_path = args.y_true_path
    lp_model_path = args.lp_model_path
    nch_model_path = args.nch_model_path
    ocr_model_path= args.ocr_model_path

    ocr_model , lp_model, nch_model = load_license_plate_models(paths=[lp_model_path, nch_model_path,ocr_model_path])

    process_images(image_test_path, image_save_path, y_true_path, lp_model, nch_model,ocr_model)





