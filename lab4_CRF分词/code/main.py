import re
from collections import Counter
import json
from tqdm import tqdm

file_dir = "./data/train.txt"
data_dir = "./data/train_data3.txt"
line_number = 0
sign_number_dict = {
    'Ag': 0, 'a': 0, 'ad': 0, 'an': 0, 'Bg': 0,
    'b': 0, 'c': 0, 'Dg': 0, 'd': 0, 'e': 0,
    'f': 0, 'g': 0, 'h': 0, 'i': 0, 'j': 0,
    'k': 0, 'l': 0, 'Mg': 0, 'm': 0, 'Ng': 0,
    'n': 0, 'nr': 0, 'ns': 0, 'nt': 0, 'nx': 0,
    'nz': 0, 'o': 0, 'p': 0, 'Qg': 0, 'q': 0,
    'Rg': 0, 'r': 0, 's': 0, 'Tg': 0, 't': 0,
    'Ug': 0, 'u': 0, 'Vg': 0, 'v': 0, 'vd': 0,
    'vn': 0, 'w': 0, 'x': 0, 'Yg': 0, 'y': 0, 'z': 0
}
signs = ['Ag', 'a', 'ad', 'an', 'Bg', 'b', 'c', 'Dg', 'd', 'e',
        'f', 'g', 'h', 'i', 'j', 'k', 'l', 'Mg', 'm', 'Ng',
        'n', 'nr', 'ns', 'nt', 'nx', 'nz', 'o', 'p', 'Qg', 'q',
        'Rg', 'r', 's', 'Tg', 't', 'Ug', 'u', 'Vg', 'v', 'vd',
        'vn', 'w', 'x', 'Yg', 'y', 'z']
number = ['０', '１', '２', '３', '４', '５', '６', '６', '８', '９', '零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
punctuation = ['，', '。', '！', '？', '；', '：', '“', '”', '‘', '’', '（', '）', '【', '】', '《', '》', '、', '·', '—', '…', ',', '.', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '<', '>', '/', '|', '\\', '-']
all_sign_dict = {key:{} for key in signs}
data_list = []

# 添加数字标点特征，如果是数字则T
def number_feature(word):
    if word in number:
        word = word + ' ' + 'T'
    elif word in punctuation:
        word = word + ' ' + 'P'
    else:
        word = word + ' ' + 'F'
    return word

# 分词训练集
def write_data(list):
    for data in list:
        sign_list = re.findall("/([\S]{1,2})", data) 
        sign = sign_list[0]
        words_list = re.findall("([\S]*)/", data)
        words = words_list[0]
        if sign == 'nr':
            count = 1
            for word in words:
                word = number_feature(word)
                if count:
                    word = word + ' ' + 'B_NR'
                    data_list.append(word)
                    count -= 1
                else:
                    word = word + ' ' + 'I'
                    data_list.append(word)
        elif sign == 'ns':
            count = 1
            for word in words:
                word = number_feature(word)
                if count:
                    word = word + ' ' + 'B_NS'
                    data_list.append(word)
                    count -= 1
                else:
                    word = word + ' ' + 'I'
                    data_list.append(word)
        elif sign == 'nt':
            count = 1
            for word in words:
                word = number_feature(word)
                if count:
                    word = word + ' ' + 'B_NT'
                    data_list.append(word)
                    count -= 1
                else:
                    word = word + ' ' + 'I'
                    data_list.append(word)
        else:
            for word in words:
                word = number_feature(word)
                word = word + ' ' + 'O'
                data_list.append(word)
      
# 处理行：处理特殊情况，并返回处理后的行，对[]单独统计
def line_processing(line):
    process_name = re.sub("(\s.)(/nr[\s]*)([\S]*/nr)", r'\1\3', line) # 保留1、3组（\1表示第一组）
    all_ns = re.findall("\[(.*?\][\S]{1,2})", process_name) # 找到所有的[]
    
    for special in all_ns: # 单独统计[]
        sign_list = re.findall("]([\S]{1,2})", special)
        sign = sign_list[0]
        special_string = re.findall("([\S]*)/", special)
        combined_string = ''.join(special_string)
        combined_string = combined_string + '/'
        if combined_string in all_sign_dict[sign]:
            all_sign_dict[sign][combined_string] += 1
        else:
            all_sign_dict[sign][combined_string] = 1
        process_name = re.sub("\[.*?\]", combined_string, process_name, count=1) # 从line中删去[]
        # print(combined_string)
        # print(process_name)
    return process_name
    
# 测试用：测试各种sign出现的次数
def find_all(line):
    # print(line)
    all = re.findall("\s([\S]*)/", line) # step3：匹配，match从开头匹配，findall找到所有子串（找到' '和'/'之间的所有子串）
    all_sign = re.findall("\]([\S]{1,2})", line)
    sign_counter = Counter(all_sign) # 统计标记数目
    for string, count in sign_counter.items():
        sign_number_dict[string] += count

# 对单类sign进行排序
def sort(dict, key):
    sorted_items = sorted(dict[key].items(), key=lambda x: x[1], reverse=True) # key=lambda x: x[1]代表比较键值。x指代dict[key].items()的元素（即元祖列表的元素，即一个元祖），x[1]指代元祖的第二个元素
    if len(sorted_items) >= 10:
        top_keys = [item[0] for item in sorted_items[:10]]
        top_values = [item[1] for item in sorted_items[:10]]  
        print("{}: ".format(key))
        for i in range(10):
            print("{}.  {}:{}".format(i+1, top_keys[i], top_values[i]))
        print('--------------') 
    else:
        length = len(sorted_items)
        top_keys = [item[0] for item in sorted_items[:length]]
        top_values = [item[1] for item in sorted_items[:length]]  
        print("{}: ".format(key))
        for i in range(length):
            print("{}.  {}:{}".format(i+1, top_keys[i], top_values[i]))
        print('--------------')

# 对所有sign进行排序
def sort_all(dict):
    sorted_items = []
    for sign, sub_dict in dict.items():
        for sub_key, sub_value in sub_dict.items():
            sorted_items.append((sign, sub_key, sub_value))

    # 对排序元祖列表进行降序排序
    sorted_items.sort(key=lambda x: x[2], reverse=True) # x指代元祖，x[2]指代元祖的第三元素
    print("total:")
    for i in range(10):
        print("{}.  {} {}:{}".format(i+1, sorted_items[i][0], sorted_items[i][1], sorted_items[i][2]))
    print('--------------')

def main():
    global line_number
    with open(file_dir, 'r') as file: # step1：文件太大无法直接读取，按行读取文件 train.txt不用encoding='utf-8'
        total_lines = 22723
        with tqdm(total=total_lines, desc="Processing") as pbar:
            while True:
                line = file.readline()
                if not line:
                    break
                line = line[22:] # step2：删除标识符
                
                # 对行进行处理
                
                find_all(line) # 处理人名
                processed_line = line_processing(line) # 处理地名
                # print(processed_line)

                list = re.findall("[\S]*/[\S]{1,2}", processed_line)
                # print(list)

                write_data(list)
                data_list.append('')
                
                line_number += 1
                pbar.update(1)
                # print(data_list)

    file.close()

    with open(data_dir, 'w') as file:
        # 将列表中的每个元素写入文件的一行
        for item in data_list:
            file.write(item + '\n')

if __name__ == "__main__":
    main() 
