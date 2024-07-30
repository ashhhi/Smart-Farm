import tensorflow as tf
import yaml
import numpy as np
from tqdm import tqdm
import cv2 as cv

with open('../config.yml', 'r') as file:
    yaml_data = yaml.safe_load(file)
    Width = yaml_data['Train']['Image']['Width']
    Height = yaml_data['Train']['Image']['Height']

image_path = 'photo/iphone.jpg'
