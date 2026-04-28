from DataPreprocessing import *

if __name__ == "__main__":
    paths = list()
    paths.append('dataset/frequency/azure_functions_2019/data.json')
    paths.append('dataset/frequency/huawei_private_2023/data.json')
    paths.append('dataset/frequency/huawei_private_2025/data.json')
    for path in paths:
        dataPreprocessing = DataPreprocessing(data_path=path)
        dataPreprocessing.init()
        dataPreprocessing.process()