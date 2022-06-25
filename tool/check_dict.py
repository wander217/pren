from dataset.alphabet import Alphabet

alphabet = Alphabet(r'D:\workspace\project\pren\asset\viet_alphabet.txt', 128)
dict_path = r'D:\workspace\project\pren\tool\line_dict.txt'
with open(dict_path, 'r', encoding='utf-8') as f:
    line_dict = {alphabet.decode(alphabet.encode(line.strip())): 1 for line in f.readlines()}

with open(r"D:\workspace\project\pren\tool\checked_dict.txt", 'w', encoding='utf-8') as f:
    for item, _ in line_dict.items():
        f.write(item)
        f.write("\n")
