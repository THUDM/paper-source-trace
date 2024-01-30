import json
with open("test.json", "r") as read_file:
    data_dic = json.load(read_file)
write_data = open("test2.json", "w")
write_ID_num = open("ID_num_dic.json", "w")
ID_num_dic = {}
n = 0
for item in data_dic:
    this_data = data_dic[item]
    ID_num_dic[item] = []
    for jtem in this_data:
        ID_num_dic[item].append(n)
        n += 1
        write_data.write(str(json.dumps(jtem)) + '\n')
json.dumps(ID_num_dic, write_ID_num, indent=2)
