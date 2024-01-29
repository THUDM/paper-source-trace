import requests
import json

url = ""  # Your url

headers = {
    "Authorization": "",  # Your Authorization
    "content-type": "application/json"
}
with open("result/claude.json", "r") as read_file:
    finished_dic = json.load(read_file).keys()
result_dic = {}
write_file = open("result/claude.json", "a")
try:
    with open("test.json", "r") as read_file:
        data_dic = json.load(read_file)
    for item in data_dic.keys():
        if item in finished_dic:
            continue
        result_list = []
        for jtem in data_dic[item]:
            this_data = json.loads(jtem.strip())
            content, summery = this_data["content"], this_data["summary"]
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "model": "claude-instant-1-100k",
                "max_tokens_to_sample": 300,
            }
            response = requests.post(url, headers=headers, json=data)
            answer = json.loads(response.text)["choices"][0]["message"]["content"]
            result_list.append({"labels": summery, "predict":answer})
        result_dic[item] = result_list
except:
    pass
finally:
    json.dump(result_dic, write_file, indent=2)