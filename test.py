# Top two lines are not necessary when do outer import
import sys
sys.path.append("../")
import time

# From here
from praise_detect.predictor import Predictor


model_path = "./model_zoo/package_model_praise"
model = Predictor(model_path)

# predict one piece of sentence
"""
Input : string
Output : (list<int>,list<float>, list<list<string>>) ; 
         (labels, probs, keywords) ;
         ([1], [0.9628134965896606] )
"""
labels, probs, keywords = model.predict("你真的好棒啊!")
print(labels, probs)
print(keywords)


# predict list of sentences
"""
Input : list<string>
Output:(list<int>,list<float>, list<list<string>>) ; 
       (labels, probs, keywords) ;
       assert len(list<int>) == len(list<string>)
"""
input_list = ["你你真的很棒，我觉得你不错，现在这样真的很好，嗯嗯，你应该继续努力的你真的好棒你真的好棒你真的好棒你应该这样去做不应该这样你真的很棒，我觉得你不错，现在这样真的很好，嗯嗯，你应该继续努力的你真的好棒你真的好棒你真的好棒你应该这样去做不应该这样你真的很棒，我觉得你不错，现在这样真的很好，嗯嗯，你应该继续努力的你真的好棒你真的好棒你真的好棒你应该这样去做不应该这样你真的很棒，我觉得你不错，现在这样真的很好，嗯嗯，你应该继续努力的你真的好棒你真的好棒你真的好棒你应该这样去做不应该这样真的很棒，我觉得你不错，现在这样真的很好，嗯嗯，你应该继续努力的你真的好棒你真的好棒你真的好棒你应该这样去做不应该这样取做！"]*100
average_cost_list = []
eval_num = 0
average_char_num = []
while True:

    start_time = time.time()
    labels, probs,_ = model.predict(input_list)
    end_time = time.time()
    cost_time = end_time - start_time
    char_num = len(input_list[0])*len(input_list)
    char_num_per_s = char_num / cost_time
    average_char_num.append(char_num_per_s)
    average_cost_list.append(cost_time)
    print("Step {}".format(eval_num))
    print("#1 current step cost {}".format(cost_time))
    print("#2 current step char nums per s: {}".format(char_num_per_s))
    print("#3 average cost time:{}".format(sum(average_cost_list)/len(average_cost_list)))
    print("#4 average char nums per s: {}".format(sum(average_char_num)/len(average_char_num)))

    assert len(input_list) == len(labels) == len(probs)
    eval_num += 1
    if eval_num %100 ==0:
        print("Sleep ...")
        time.sleep(10)

print(cost_time)
