import os
import yaml

def Dataloader():
    with open('config.yml', 'r') as file:
        yaml_data = yaml.safe_load(file)
        image_path = yaml_data['Train']['image_path']
        label_path = yaml_data['Train']['label_path']
    g = os.walk(image_path)
    image = []
    label = []
    image_name = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name.split('.')[-1] == 'jpg':
                image.append(str(os.path.join(path, file_name)))
                label.append(str(os.path.join(label_path, file_name.replace('.jpg','.png'))))
                image_name.append(file_name)

    # 处理xml文件
    return image, label, image_name
