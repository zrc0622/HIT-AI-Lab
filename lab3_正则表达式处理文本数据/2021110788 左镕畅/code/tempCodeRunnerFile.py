import re
from collections import Counter
import json

file_dir = "./data/199801.txt"
json_dir = "./data/data.json"
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
all_sign_dict = {key:{} for key in signs}

# 处理行：处理特殊情况，并返回处理后的行，对[]单独统计
def line_processing(line):
    # 处理人名：出现单字的时候查看后续是不是/nr或/n，如果是则合并
    # 1. 沈/nr  安/nr √
    # 2. 果/nr  永毅/nr √
    # 3. 克林顿/nr √
    # 4. 江/nr  主席/n √
    process_name = re.sub("(\s.)(/nr[\s]*)([\S]*/nr)", r'\1\3', line) # 保留1、3组（\1表示第一组）
    # print("1 "+process_name)
    # 处理地名
    # 先把地名[]抽出来，组合成正确地名后，再将原列中的[]删去，只有i、l、ns、nt、nz
    all_ns = re.findall("\[(.*?\][\S]{1,2})", process_name) # 找到所有的[]
    process_special = re.sub("\[.*?\]", '', process_name) # 从line中删去[]
    for special in all_ns: # 单独统计[]
        sign_list = re.findall("]([\S]{1,2})", special)
        sign = sign_list[0]
        special_string = re.findall("([\S]*)/", special)
        combined_string = ''.join(special_string)
        if combined_string in all_sign_dict[sign]:
            all_sign_dict[sign][combined_string] += 1
        else:
            all_sign_dict[sign][combined_string] = 1
        # if special[-1] == 'i':
        #     special_string = re.findall("([\S]*)/", special)
        #     combined_string = ''.join(special_string)
        #     if combined_string in idict:
        #         idict[combined_string] += 1
        #     else:
        #         idict[combined_string] = 1
        # elif special[-1] == 'l':
        #     special_string = re.findall("([\S]*)/", special)
        #     combined_string = ''.join(special_string)
        #     if combined_string in ldict:
        #         ldict[combined_string] += 1
        #     else:
        #         ldict[combined_string] = 1
        # elif special[-1] == 's':
        #     special_string = re.findall("([\S]*)/", special)
        #     combined_string = ''.join(special_string)
        #     if combined_string in nsdict:
        #         nsdict[combined_string] += 1
        #     else:
        #         nsdict[combined_string] = 1
        # elif special[-1] == 't':
        #     special_string = re.findall("([\S]*)/", special)
        #     combined_string = ''.join(special_string)
        #     if combined_string in ntdict:
        #         ntdict[combined_string] += 1
        #     else:
        #         ntdict[combined_string] = 1
        # elif special[-1] == 'z':
        #     special_string = re.findall("([\S]*)/", special)
        #     combined_string = ''.join(special_string)
        #     if combined_string in nzdict:
        #         nzdict[combined_string] += 1
        #     else:
        #         nzdict[combined_string] = 1
    return process_special
    # print("2 "+process_special)
    # print(all_ns)
    
# 测试用：测试各种sign出现的次数
def find_all(line):
    # print(line)
    all = re.findall("\s([\S]*)/", line) # step3：匹配，match从开头匹配，findall找到所有子串（找到' '和'/'之间的所有子串）
    # for find in all:
    #     print(find)
    # all_sign = re.findall("/([\S]{0,2})\s", line) # step4：记录所有的标记，注意n]nt问题
    all_sign = re.findall("\]([\S]{1,2})", line)
    sign_counter = Counter(all_sign) # 统计标记数目
    for string, count in sign_counter.items():
        sign_number_dict[string] += count
    # for find in all_type:
    #     print(find)

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
    with open(file_dir, 'r') as file: # step1：文件太大无法直接读取，按行读取文件
        while True:
            line = file.readline()
            if not line:
                break
            line = line[22:] # step2：删除标识符
            
            # 对行进行处理
            
            find_all(line)
            processed_line = line_processing(line)

            list = re.findall("[\S]*/[\S]{1,2}", processed_line)

            for key in list:
                sign_list = re.findall("/([\S]{1,2})", key) # 修改全局dict不需要声明
                sign = sign_list[0]
                string_list = re.findall("([\S]*)/", key)
                string = string_list[0]
                if string in all_sign_dict[sign]:
                    all_sign_dict[sign][string] += 1
                else:
                    all_sign_dict[sign][string] = 1

            global line_number # 修改全局int需要声明
            line_number += 1
            # print(line)

    file.close()

    # print(sign_number_dict)
    # print(idict)
    # print(ldict)
    # print(nzdict)
    # print(ntdict)
    # print(nsdict)
    # print(all_sign_dict)
    for sign in signs:
        sort(all_sign_dict, sign)
    sort_all(all_sign_dict)
    with open(json_dir, 'w', encoding='utf-8') as json_file:
        json.dump(all_sign_dict, json_file, ensure_ascii=False)

if __name__ == "__main__":
    main() 
