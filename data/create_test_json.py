import json
import os
import random
import copy

dataset_name = 'FSC147_384_V2'
neurocle_file_name = f'{dataset_name}_bbox_annotation'
anno_file_name = f'{dataset_name}_annotation'

split_name = ''
set_name = '_all'



with open(f"{dataset_name}/{split_name}/{anno_file_name}.json", "r") as anno_json:
    data = json.load(anno_json)

    data_annotations = []
    for item in data:
        data_annotations.append({
                "fileName": item,
                "density": item.split('.jpg')[0] + '.npy',
                "boxes": data[item]["box_examples_coordinates"],
                "points": data[item]["points"]
        })
        
    # "train.json" 파일에 데이터 쓰기
    with open(f"{dataset_name}/{split_name}/test{set_name}.json", "w") as output:
        json.dump(data_annotations, output, indent=4)
