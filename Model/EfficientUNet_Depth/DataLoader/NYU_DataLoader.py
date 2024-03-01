import os
import yaml
def Dataloader():
    with open('./config.yml', 'r') as file:
        yaml_data = yaml.safe_load(file)
        image_path = yaml_data['Path']['image_path']
    g = os.walk(image_path)
    image = []
    label = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            image.append(str(os.path.join(path, file_name)))
            label.append(str(os.path.join(path[:-6]+"depths", file_name.replace("jpg", "png"))))
    return image, label