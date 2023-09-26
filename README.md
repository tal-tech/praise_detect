## Algorithm praise detector


用来检测教师鼓励行为的模型，模型存放在`./model_zoo`：

|行为 | 模型文件名称 | Precision | Recall | F1|Accuracy|
|-----|-------|--------|------|------|-----|
|鼓励| `package_model_praise`|0.86|0.91|0.82|0.84|

### Efficiency Performace 
* Please confirm your environment and installed necessary modules before running model.
* You can acquire some information from requirements.txt.

* test environment
  * Tesla P4, ubuntu 18.04, 4-cores CPU, 16G memory, GPU 7611MB(Not use), driver version= 418.67

  * Duration: 17 hours

* Performace

  * speed 

| 单位|value|
|-----|-----|
|字符/秒|26857|

### Quick Start

#### if you want to run this model, you can enter these on your command line.

```shell script
> python3 test.py
```


#### USAGE

* Basic version

```python
# Top two lines are not necessary when do outer import
import sys
sys.path.append("../")

from algorithm_praise_detector.predictor import Predictor


model_path = "/workspace/tmp/package_model_praise"
model = Predictor(model_path)

# predict one piece of sentence
labels, probs, keywords = model.predict("你真的好棒啊!")
print(labels, probs)
print(keywords)

# predict list of sentences
input_list = ["你真的好棒！"]*3000
labels, probs, _ = model.predict(input_list)
assert len(input_list) == len(labels) == len(probs)

```


* Interface version

```python
# Top two lines are not necessary when do outer import
import sys
sys.path.append("../")

from algorithm_praise_detector import model


model_path = "./model_zoo/package_model_praise"
model = model(model_path)
input_text = [
    {
        "text":"这句话呢，其我我靠实都是很棒告诉你很棒规则，他就看你能不能看到他这个给你的规定了。",
        "begin_time":1326750,
        "end_time":1332165
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

```

