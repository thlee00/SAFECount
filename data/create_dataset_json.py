import json
import os
import random

dataset_name = 'mvtec'
neurocle_file_name = f'{dataset_name}_bbox_annotation'
anno_file_name = f'{dataset_name}_annotation'
split_name = ''
set_name = ''
train_test_val_set = ['train', 'val', 'test']

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

# create train_test_val json from annotation json
def train_test_val(data):
    new_data = {
        "train": [],
        "val": [],
        "test": []
    }

    with open(f'{dataset_name}/{anno_file_name}.json', 'r') as f:
        annotation = json.load(f)
    
    train_items = []
    for data in annotation:
        if len(annotation[data]["box_examples_coordinates"]) > 0:
            train_items.append(data)
        else:
            new_data["test"].append(data)
    random.shuffle(train_items)
    num_train = 50
    num_train_val = int(num_train * 0.8)
    new_data["train"] = train_items[:num_train_val]
    new_data["val"] = train_items[num_train_val:num_train]
    new_data["test"] = train_items[num_train:]

    createDirectory(f'{split_name}')

    with open(f'{dataset_name}/{split_name}/{dataset_name}_train_test_val.json', 'w') as f:
        json.dump(new_data, f, indent=4)


if not os.path.isfile(f'{dataset_name}/{split_name}/{dataset_name}_train_test_val.json'):
    with open(f'{dataset_name}/{neurocle_file_name}.json', 'r') as f:
        json_data = json.load(f)

    # create train_test_val json
    train_test_val(json_data)

# train_test_val 파일에서 "train"에 해당하는 데이터 이름 추출하여 리스트로 저장
for data_subset in train_test_val_set:
    data_names = []
    with open(f"{dataset_name}/{split_name}/{dataset_name}_train_test_val.json", "r") as train_test_val:
        data = json.load(train_test_val)
        for item in data[data_subset]:
            data_names.append(item)

    # annotation 파일에서 "box_examples_coordinates" 값 가져오기
    boxes_data = {}
    with open(f"{dataset_name}/{anno_file_name}.json", "r") as annotation:
        data = json.load(annotation)
        for item in data:
            if "box_examples_coordinates" in data[item]:
                boxes_data[item] = data[item]["box_examples_coordinates"]

    # annotation 파일에서 "points" 값 가져오기
    points_data = {}
    with open(f"{dataset_name}/{anno_file_name}.json", "r") as annotation:
        data = json.load(annotation)
        for item in data:
            if "points" in data[item]:
                points_data[item] = data[item]["points"]

    # "train.json" 파일에 데이터 저장
    data_annotations = []
    for data_name in data_names:
        if data_name in boxes_data:
            data_annotations.append({
                "fileName": data_name,
                "density": data_name.split('.jpg')[0] + '.npy',
                "boxes": boxes_data[data_name],
                "points": points_data[data_name]
            })
        else:
            data_annotations.append({
                "fileName": data_name,
                "density": data_name.split('.jpg')[0] + '.npy',
                "boxes": [],
                "points": []
            })
    # "train.json" 파일에 데이터 쓰기
    with open(f"{dataset_name}/{split_name}/{data_subset}{set_name}.json", "w") as output:
        json.dump(data_annotations, output, indent=4)
        print(f"{data_subset} done...")
