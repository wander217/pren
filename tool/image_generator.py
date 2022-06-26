import copy
import math
import os.path
import random
from PIL import Image, ImageDraw, ImageFont
import argparse
import lmdb


def resize(text_image, org_size, new_h):
    o_w, o_h = org_size
    scale = new_h / o_h
    new_w, new_h = math.ceil(scale * o_w), math.ceil(scale * o_h)
    text_image = text_image.resize((new_w, new_h))
    return text_image


def generate_image(text, background, font, padding):
    pw, ph = random.randint(*padding), random.randint(*padding)
    w, h = font.getsize(text)
    new_w, new_h = w + 2 * pw, h + 2 * ph
    image = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 0))
    image_draw = ImageDraw.Draw(image)
    image_draw.text((pw, ph), text, fill=(0, 0, 0), font=font)
    text_image = copy.deepcopy(background).resize((new_w, new_h))
    text_image.paste(image, (0, 0), image)
    text_image = resize(text_image, (new_w, new_h), 32)
    padded_image = Image.new("RGB", (900, 32))
    padded_image.paste(text_image, (0, 0))
    return padded_image


def generator(dict_path: str, background_path: str, save_path: str, font_path: str):
    with open(dict_path, 'r', encoding='utf-8') as f:
        line_dict = [line.strip() for line in f.readlines()]
    font = ImageFont.truetype(font_path, size=30)
    bg_item = os.listdir(background_path)
    background = Image.open(os.path.join(background_path, random.choice(bg_item))).convert("RGB")
    padding = (10, 20)
    env = lmdb.open(save_path, map_size=8589934592)
    sample = 0
    with env.begin(write=True) as txn:
        for i1, item1 in enumerate(line_dict):
            if i1 % 10000 == 0:
                print(i1, item1)
            image = generate_image(item1, background, font, padding)

            image_code = 'image-%09d' % (i1 + 1)
            label_code = 'label-%09d' % (i1 + 1)
            txn.put(label_code.encode(encoding='utf-8', errors='ignore'),
                    item1.encode(encoding='utf-8', errors='ignore'))
            image.save("tmp.png")
            with open("tmp.png", 'rb') as f:
                image_byte = f.read()
            txn.put(image_code.encode(encoding='utf-8', errors='ignore'), image_byte)
            sample += 1
        txn.put("num-samples".encode(encoding='utf-8', errors='ignore'),
                str(sample).encode(encoding='utf-8', errors='ignore'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("abc")
    parser.add_argument("-d", "--dict_path", default="", type=str, help="Path of dictionary")
    parser.add_argument("-s", "--save_path", default="", type=str, help="Path of saving directory")
    parser.add_argument("-f", "--font_path", default="", type=str, help="Path of font directory")
    parser.add_argument("-b", "--bg_path", default="", type=str, help="Path of background")
    args = parser.parse_args()
    os.mkdir(args.save_path)
    i = 2
    os.mkdir(os.path.join(args.save_path, "font_{}".format(i + 1)))
    generator(args.dict_path,
              args.bg_path,
              os.path.join(args.save_path, "font_{}".format(i + 1)),
              args.font_path)
