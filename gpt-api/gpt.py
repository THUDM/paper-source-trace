import openai
import json
openai.api_key = ""  # your api key
openai.api_base = ""  #  your api base
model_list = ["gpt-3.5-turbo","gpt-4"]
for model in model_list:
    result_dic = {}
    with open("result/" + model + "2.json", "r") as read_dic:
        finished_dic = json.load(read_dic).keys()
    write_dic = open("result/" + model + "2.json", "a")
    try:
        with open("test.json", "r") as read_file:
            data_dic = json.load(read_file)
        for this_data in data_dic.keys():
            if this_data in finished_dic:
                continue
            result_list = []
            for jtem in data_dic[this_data]:
                question = jtem["content"]
                result = jtem["summary"]
                flag = True
                while flag:
                    try:
                        chat_completion = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content":question}], stream=True)
                        completion_text = ""
                        for event in chat_completion:
                                if len(event["choices"]) > 0:
                                        completion_text += event["choices"][0]["delta"].get("content", "")
                        result_list.append(json.dumps({"labels":result, "predict":completion_text})+"\n")
                        flag = False
                    except openai.error.APIError:
                        continue
            result_dic[this_data] = result_list
    except:
        pass
    finally:
        json.dump(result_dic, write_dic, indent=2)