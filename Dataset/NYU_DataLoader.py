import os

image_path = "/Users/shijunshen/Documents/Code/dataset/NYUv2pics/nyu_images"
label_path = "/Users/shijunshen/Documents/Code/dataset/NYUv2pics/nyu_depths"


with open('../Model/EfficientNet/image_path.txt', 'w') as f:    #设置文件对象
    g = os.walk(image_path)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            info = str(os.path.join(path, file_name)) + " " + str(os.path.join(path[:-6]+"depths", file_name.replace("jpg", "png")) + '\n')
            f.write(info)

