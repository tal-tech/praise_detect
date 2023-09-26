# Top two lines are not necessary when do outer import
import sys
sys.path.append("../")

# From here
from algorithm_praise_detector.predictor import Predictor
import re


class Responser:
    def __init__(self, model_path):
        self.model = Predictor(model_path)

    @staticmethod
    def parse_input(input_data):
        text_key, bt_key, et_key = "text", "begin_time", "end_time"

        texts, begin_times, end_times = [], [], []
        for d in input_data:
            texts.append(d[text_key])
            begin_times.append(d[bt_key])
            end_times.append(d[et_key])
        return texts, begin_times, end_times

    @staticmethod
    def response_body_format(labels, keywords, texts, begin_times, end_times):
        response_body = []
        for i, ll in enumerate(labels):
            if ll == 1:
                keyword_list = []
                for k in set(keywords[i]):
                    keyword_list.append({"keyword": k, "word_count": len(re.findall(k, texts[i]))})
                info = {
                    "keyword_list":keyword_list,
                    "sentence": texts[i],
                    "begin_time": begin_times[i],
                    "end_time": end_times[i]
                }
                response_body.append(info)
        return response_body

    def __call__(self, input_data):
        texts, begin_times, end_times = self.parse_input(input_data)
        labels, _, keywords = self.model.predict(texts)
        response = self.response_body_format(labels, keywords, texts, begin_times, end_times)
        return response

