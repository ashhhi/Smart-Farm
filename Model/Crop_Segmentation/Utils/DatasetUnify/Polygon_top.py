import os
import shutil

import numpy as np
import yaml
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET


image_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Broccoli/train'

img_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Broccoli/img'
label_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/Broccoli/mask'
with open('/Users/shijunshen/Documents/Code/PycharmProjects/Smart-Farm/Model/Crop_Segmentation/config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    class_map = yaml_data['Train']['Class_Map']

def polygon_to_mask(label_path, save_name):
    # Parse the XML label
    tree = ET.parse(label_path)
    root = tree.getroot()

    leaf_polygons = []
    stem_polygons = []
    potplant_polygons = []

    # Process each object element
    for object_element in root.findall('object'):
        # Extract information from the XML elements
        name = object_element.find('name').text
        xmin = int(object_element.find('bndbox/xmin').text)
        xmax = int(object_element.find('bndbox/xmax').text)
        ymin = int(object_element.find('bndbox/ymin').text)
        ymax = int(object_element.find('bndbox/ymax').text)

        polygon_element = object_element.find('polygon')
        polygon_points = []
        for i in range(1, 999999):
            try:
                x = float(polygon_element.find(f'x{i}').text)
                y = float(polygon_element.find(f'y{i}').text)
                polygon_points.append((x, y))
            except Exception as e:
                break

        # Print the extracted information for each object
        print('Object Name:', name)
        print('Bounding Box:', xmin, ymin, xmax, ymax)
        print('Polygon Points:', polygon_points)
        print('------------------------')
        if name == 'leaf' or name == 'lead':
            leaf_polygons.append(polygon_points)
        elif name == 'plant' or name == 'pot':
            potplant_polygons.append(polygon_points)
        else:
            stem_polygons.append(polygon_points)


    # Get the image size from the XML or provide it manually
    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    # Create a blank RGB mask image
    mask = Image.new('RGB', (image_width, image_height), (0, 0, 0))

    # Create a draw object
    draw = ImageDraw.Draw(mask)

    # Set colors for stem and leaf
    leaf_color = tuple(class_map['Leaf'])  # Green color
    if len(leaf_polygons) == 0:
        return False
    # Draw leaf polygons in green
    for item in leaf_polygons:
        # Convert polygon points to integer tuples
        polygon_points = [(int(x), int(y)) for x, y in item]

        # Draw the polygon on the mask
        draw.polygon(polygon_points, outline=leaf_color, fill=leaf_color)
    # Convert the mask to a NumPy array
    mask.save(save_name)
    return True

g = os.walk(image_path)
image = []
label = []
for path, dir_list, file_list in g:
    for file_name in file_list:
        if file_name.split('.')[-1] == 'xml':
            if polygon_to_mask(str(os.path.join(path, file_name)), str(os.path.join(label_path, str(file_name.replace('.xml', '.png'))))):
                shutil.copy(os.path.join(path, file_name.replace('.xml', '.jpg')), img_path)