import requests
import json

url = ""  # Your url

headers = {
    "Authorization": "",  # Your Authorization
    "content-type": "application/json"
}
with open("result/claude.json", "r") as read_file:
    line_num = len(read_file.readlines())
with open("test.json", "r") as read_file:
    all_lines = read_file.readlines()
write_file = open("result/claude.json", "a")
all_lines = all_lines[line_num:]
for item in all_lines:
    data_dic = json.loads(item.strip())
    content, summery = data_dic["content"], data_dic["summary"]
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
    write_file.write(json.dumps({"labels": summery, "predict":answer})+"\n")
    write_file.flush()
