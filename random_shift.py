#! /usr/bin/python3

"""
This code shifts images and corresponding XML annotations randomly.

Author: Burak Ulas -  github.com/burakulas
2024, Konkoly Observatory, COMU
"""

import cv2
import xml.etree.ElementTree as ET
import os
from random import randrange
import numpy as np

# Define the directory containing images and XML annotations
images_dir = 'data/' 
annotations_dir = 'data/' 

# Define the translation amounts
if not os.path.exists('translated_images'):
    os.makedirs('translated_images')
if not os.path.exists('modified_annotations'):
    os.makedirs('modified_annotations')

# Loop through each image in the directory
for image_file in os.listdir(images_dir):
    if image_file.endswith('.png'):
        # Load the image
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)

        # Load the corresponding annotation file

        annotation_file = os.path.splitext(image_file)[0] + '.xml'
        annotation_path = os.path.join(annotations_dir, annotation_file)
        print(annotation_path)
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Update the image by translation
        tx = randrange(-15, 55)  
        ty = randrange(-20, 20)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

        # Update the annotation coordinates accordingly
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            bbox.find('xmin').text = str(int(bbox.find('xmin').text) + tx)
            bbox.find('xmax').text = str(int(bbox.find('xmax').text) + tx)
            bbox.find('ymin').text = str(int(bbox.find('ymin').text) + ty)
            bbox.find('ymax').text = str(int(bbox.find('ymax').text) + ty)

        # Save the modified image
        translated_image_path = os.path.join('translated_images/', image_file)
        cv2.imwrite(translated_image_path, translated_image)

        # Save the modified annotation file
        modified_annotation_path = os.path.join('modified_annotations/', annotation_file)
        tree.write(modified_annotation_path)

print("Translation complete for all images and annotations.")
