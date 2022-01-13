import json
import pandas as pd
import sklearn
import random
import numpy as np
import os

class Sample_utils(object):
    def __init__(self, path):
        self.path = path
    
    def get_sample_agnews(self, sample_num, seed):
        data = pd.read_csv(self.path + 'train.csv', header=None)
        
        data1 = data[data[0]==1]
        data2 = data[data[0]==2]
        data3 = data[data[0]==3]
        data4 = data[data[0]==4]

        data1 = data1.sample(n=sample_num, random_state=seed)
        data2 = data2.sample(n=sample_num, random_state=seed)
        data3 = data3.sample(n=sample_num, random_state=seed)
        data4 = data4.sample(n=sample_num, random_state=seed)

        sample_data = pd.concat([data1, data2, data3, data4], axis=0)

        np.random.seed(seed)
        shuffle_index = np.random.permutation(len(sample_data))
        sample_data = sample_data.iloc[shuffle_index]

        filename = os.path.join(self.path, 'train_sample_' + str(sample_num) + '_' + str(seed) + '.csv')

        sample_data.to_csv(filename, header=0, index=0)

        print("saving sample data at " + filename)
    
    def get_sample_dbpedia(self, sample_num, seed):
        data_path = os.path.join(self.path, 'train.txt')
        label_path = os.path.join(self.path, 'train_labels.txt')
        with open(data_path, 'r') as f:
            data = f.readlines()
        with open(label_path, 'r') as f:
            labels = f.readlines()
        assert len(data) == len(labels)

        nums = 40000 # 40000 for sampling train
        np.random.seed(seed)
        shuffle_index = np.random.permutation(nums)

        new_data = []
        new_labels = []

        for i in range(14):
            start = nums * i
            cnt = 0
            for i in range(shuffle_index.shape[0]):
                if cnt == sample_num:
                    break
                index = start + shuffle_index[i]
                new_data.append(data[index])
                new_labels.append(labels[index])
                cnt += 1
        
        data_filename = os.path.join(self.path, 'train_sample_' + str(sample_num) + '_' + str(seed) + '.txt')
        labels_filename = os.path.join(self.path, 'train_sample_' + str(sample_num) + '_' + str(seed) + '_labels.txt')

        with open(data_filename, 'w') as f:
            for d in new_data:
                f.write(d)
        with open(labels_filename, 'w') as f:
            for l in new_labels:
                f.write(l)
        print("saving sample data at " + data_filename)

    def get_sample_yahoo(self, sample_num, seed):
        data = pd.read_csv(self.path + 'train.csv', header=None)
        
        data1 = data[data[0]==1]
        data2 = data[data[0]==2]
        data3 = data[data[0]==3]
        data4 = data[data[0]==4]
        data5 = data[data[0]==5]
        data6 = data[data[0]==6]
        data7 = data[data[0]==7]
        data8 = data[data[0]==8]
        data9 = data[data[0]==9]
        data0 = data[data[0]==10]

        data1 = data1.sample(n=sample_num, random_state=seed)
        data2 = data2.sample(n=sample_num, random_state=seed)
        data3 = data3.sample(n=sample_num, random_state=seed)
        data4 = data4.sample(n=sample_num, random_state=seed)
        data5 = data5.sample(n=sample_num, random_state=seed)
        data6 = data6.sample(n=sample_num, random_state=seed)
        data7 = data7.sample(n=sample_num, random_state=seed)
        data8 = data8.sample(n=sample_num, random_state=seed)
        data9 = data9.sample(n=sample_num, random_state=seed)
        data0 = data0.sample(n=sample_num, random_state=seed)

        sample_data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data0], axis=0)

        np.random.seed(seed)
        shuffle_index = np.random.permutation(len(sample_data))
        sample_data = sample_data.iloc[shuffle_index]

        filename = os.path.join(self.path, 'train_sample_' + str(sample_num) + '_' + str(seed) + '.csv')

        sample_data.to_csv(filename, header=0, index=0)

        print("saving sample data at " + filename)

def main():
    path = '../dataset/dbpedia/'
    seed = 8
    sample_num = 20
    sample_util = Sample_utils(path)
    sample_util.get_sample_dbpedia(sample_num, seed)


if __name__ == "__main__":
    main()

        
