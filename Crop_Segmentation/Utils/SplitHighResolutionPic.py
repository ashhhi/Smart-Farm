import numpy as np
from PIL import Image

Split_Size = [1024, 1024]

img_path = '/Users/shijunshen/Documents/Code/dataset/Smart-Farm-All/XIAOMICamera/6.18/PIC_20240617_092620595.jpg'

img = Image.open(img_path)

img = np.array(img)

img_height, img_width, img_channel = img.shape

h_div_num = img_height // Split_Size[0]

w_div_num = img_width // Split_Size[1]


split_pic_list = []
for i in range(h_div_num):
    for j in range(w_div_num):
        split_pic = img[i*Split_Size[0]:(i+1)*Split_Size[0], j*Split_Size[1]:(j+1)*Split_Size[1]]
        split_pic_list.append(split_pic)
    if img_height % Split_Size[1] != 0:
        split_pic = img[i*Split_Size[0]:(i+1)*Split_Size[0], -1 - Split_Size[1]:-1]
        split_pic_list.append(split_pic)
if img_width % Split_Size[0] != 0:
    for j in range(w_div_num):
        split_pic = img[-1 - Split_Size[0]:-1, j * Split_Size[1]:(j + 1) * Split_Size[1]]
        split_pic_list.append(split_pic)


split_pic_list = np.array(split_pic_list)
print(split_pic_list[0].shape, len(split_pic_list))
print(h_div_num, w_div_num)

for i in split_pic_list:
    i = Image.fromarray(i.astype('uint8')).convert('RGB')
    i.show()
    i.split()