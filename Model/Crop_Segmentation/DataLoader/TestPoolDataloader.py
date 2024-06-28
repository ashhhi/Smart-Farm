import os

def Dataloader(image_path, label_path):
    g = os.walk(image_path)
    image = []
    label = []
    image_name = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if file_name.split('.')[-1] == 'jpg':
                image.append(str(os.path.join(path, file_name)))
                label.append(str(os.path.join(label_path, file_name.replace('.jpg', '.png'))))
                image_name.append(file_name)

    # 处理xml文件
    return image, label, image_name
