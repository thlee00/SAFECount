import json
import numpy as np
import scipy.spatial
import scipy.ndimage
import cv2
import os
import random

dataset_name = 'mnm'
neurocle_file_name = f'{dataset_name}_bbox_annotation'
anno_file_name = f'{dataset_name}_annotation'

# neurocle json to custom json
def convert_json(json_data):
    converted_data = {}

    for item in json_data['data']:
        file_name = item['fileName']
        if file_name not in converted_data:
                converted_data[file_name] = {
                    'box_examples_coordinates': [],
                    'points': []
                }

        for region in item['regionLabel']:
            class_name = region['className']
            x = region['x']
            y = region['y']
            width = region['width']
            height = region['height']

            x_top_left = x
            y_top_left = y
            x_bottom_right = x + width
            y_bottom_right = y + height

            converted_data[file_name]['box_examples_coordinates'].append([
                [x_top_left, y_top_left],
                [x_top_left, y_bottom_right],
                [x_bottom_right, y_bottom_right],
                [x_bottom_right, y_top_left]
            ])

            center_x = (x_top_left + x_bottom_right) / 2
            center_y = (y_top_left + y_bottom_right) / 2

            converted_data[file_name]['points'].append([center_x, center_y])

        with open(f'{dataset_name}/{anno_file_name}.json', 'w') as f:
            json.dump(converted_data, f, indent=4)
    return converted_data


with open(f'{dataset_name}/{neurocle_file_name}.json', 'r') as f:
    json_data = json.load(f)

# neurocle json to custom json
converted_data = convert_json(json_data)