import configparser
class configs():
    def __init__(self,path):
        cf = configparser.ConfigParser()
        cf.read(path,encoding="utf-8")
        self.cf = cf
        self.add_variables(cf.items("model"))

    def add_variables(self,configs):
        for k,v in configs:
            self.__dict__[k] = v
    def add_params(self,key,value):
        self.__dict__[key] = value

if __name__ == "__main__":
    con_dict = configs("./hparams.ini")
    print(con_dict.embedding_size)

