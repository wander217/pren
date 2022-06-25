import json
import os.path
import random

dict_path = r'D:\workspace\project\pren\tool\checked_dict.txt'
with open(dict_path, 'r', encoding='utf-8') as f:
    line_dict = [line.strip() for line in f.readlines()]
random.shuffle(line_dict)
save_path = r'D:\workspace\project\pren\tool\data'
valid_len = 100000
valid = line_dict[:valid_len]
with open(os.path.join(save_path, 'valid.txt'), 'w', encoding='utf-8') as f:
    for item in valid:
        f.write(item)
        f.write("\n")
test = line_dict[valid_len:valid_len + 100000]
with open(os.path.join(save_path, 'test.txt'), 'w', encoding='utf-8') as f:
    for item in test:
        f.write(item)
        f.write("\n")
train = line_dict[valid_len + 100000:]
with open(os.path.join(save_path, 'train.txt'), 'w', encoding='utf-8') as f:
    for item in train:
        f.write(item)
        f.write("\n")
