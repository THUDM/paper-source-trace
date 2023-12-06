import openai
import json
openai.api_key = ""  # your api key
openai.api_base = ""  #  your api base
model_list = ["gpt-3.5-turbo","gpt-4"]
for model in model_list:
    read_dic = open("result/"+model+"2.json", "r")
    all_lines = read_dic.readlines()
    if len(all_lines) >= 1 and len(all_lines[-1].strip()) == 0:
        all_lines = all_lines[:-1]
    length = len(all_lines)
    read_dic.close()
    write_dic = open("result/" + model + ".json", "a")
    with open("test.json", "r") as read_file:
        all_lines = read_file.readlines()
        all_lines = all_lines[length:]
        for data_line in all_lines:
            this_dic = json.loads(data_line.strip())
            question = this_dic["content"]
            result = this_dic["summary"]
            flag = True
            while flag:
                try:
                    chat_completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content":question}], stream=True)
                    completion_text = ""
                    for event in chat_completion:
                            if len(event["choices"]) > 0:
                                    completion_text += event["choices"][0]["delta"].get("content", "")
                    write_dic.write(json.dumps({"labels":result, "predict":completion_text})+"\n")
                    write_dic.flush()
                    flag = False
                except openai.error.APIError:
                    continue
    write_dic.close()
    