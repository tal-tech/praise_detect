# Top two lines are not necessary when do outer import
import sys
sys.path.append("../")

# From here
from praise_detect import model

model_path = "./model_zoo/package_model_praise"
model = model(model_path)
input_text = [
    {
        "text":"你也不是特",
        "begin_time":1326750,
        "end_time":1332165
    },
    {
        "text": "真的",
        "begin_time": 1326751,
        "end_time": 1332165
    },
    {
        "text": "haha",
        "begin_time": 1326752,
        "end_time": 1332165
    }
]
print(model(input_text))
# """[
#     {
#     'keyword_list':
#                     [
#                         {
#                         'keyword': '很棒',
#                         'word_count': 2
#                         }
#                     ],
#     'sentence': '这句话呢，其我操我靠实都是很棒告诉你很棒规则，
#                     他就看你能不能看到他这个给你的规定了。',
#     'begin_time': 1326750,
#     'end_time': 1332165}
#     ]
# """
