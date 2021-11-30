import numpy as np
import pandas as pd

'''
首先将数据处理为candidate_set的形式,
label_length为总的label种数，candidate_set_length为candidate_set的长度
'''

def data_processing(config, data, candidate_set_length):
    partial_label = []  #记录partial_label
    # print(data)
    if config.data_name in ['glass']:
        label_length = 7
        data_pro = data.iloc[:, 1:-1]
        truth_label = np.array(data.iloc[:, -1])
    elif config.data_name in ['segmentation']:
        label_length = 7
        data_pro = data.iloc[:, 1:]
        truth_label = np.array(data.iloc[:, 0])
    else:
        pass
    # print(truth_label)
    for i in range(data.shape[0]):
        probs = np.ones(label_length) #初始化p
        probs[int(truth_label[i])] = 0  
        probs = probs / (label_length - 1)
        #print((truth_label[i],probs))
        '''每一个非truth_label取到等概率'''
        candidate_set = np.array(list(np.random.choice(label_length, candidate_set_length-1, 
                                               replace = False, p = probs)))   #用numpy的广播
        partial_label.append(candidate_set)

    columns_names = ['candidate_label_{}'.format(i+1) for i in range(candidate_set_length-1)]
    partial_label = pd.DataFrame(partial_label, columns = columns_names)
    print(partial_label.shape, data.shape)
    data_partial_label = pd.concat([data, partial_label], axis = 1)
    return data_partial_label

def Generate_Data(config):
    if config.data_name in ['glass']:
        columns_names = ['Id number', 'refractive index', 'Sodium', 'Magnesium', 'Aluminum',
                          'Silicon','Potassium','Calcium','Barium','Iron','Truth label']
        data = pd.read_csv(config.glass_path,
                            header = None, names = columns_names)
        labels_ = data['Truth label'].values - 1
        data['Truth label'] = labels_
        data_partial_label = data_processing(config, data, candidate_set_length = config.candidate_set_length)
        data_partial_label.to_csv(config.glass_partial_path, index = False)  #保存数据
    elif config.data_name in ['segmentation']:
        columns_names = ['Label', 'region-centroid-col', 'region-centroid-row', 'region-pixel-count', 'short-line-density-5',
                         'short-line-density-2', 'vedge-mean', 'vegde-sd', 'hedge-mean', 'hedge-sd', 'intensity-mean',
                         'rawred-mean', 'rawblue-mean', 'rawgreen-mean', 'exred-mean', 'exblue-mean', 'exgreen-mean', 'value-mean',
                         'saturatoin-mean', 'hue-mean']
        data_1 = pd.read_csv(config.segmentation_path,
                             header = 1, names = columns_names)
        data_2 = pd.read_csv(config.segmentation_test_path,
                             header = 1, names = columns_names)
        data = pd.concat([data_1, data_2])
        label = list(set(data.iloc[:, 0]))
        label_nums = np.zeros(data.shape[0])  # label数字化
        for i in range(data.shape[0]):
            label_nums[i] = label.index(data.iloc[i, 0])
        data['Label'] = label_nums
        data.to_csv('./data/segmentation/segmentation.csv', index = False)
        data = pd.read_csv('./data/segmentation/segmentation.csv')
        data_partial_label = data_processing(config, data, candidate_set_length = config.candidate_set_length)
        data_partial_label.to_csv(config.segmentation_partial_path, index = False)
    else:
        pass

if __name__ == '__main__':
    # #glass数据集
    # columns_names = ['Id number','refractive index','Sodium','Magnesium','Aluminum',
    #                   'Silicon','Potassium','Calcium','Barium','Iron','Truth label']
    # data = pd.read_csv("./data/glass/glass.data",
    #                     header = None, names = columns_names)
    # data_partial_label = data_processing(data, candidate_set_length = 3)
    # data_partial_label.to_csv('./data/glass/glass_partial.csv', index = False)  #保存数据

    #segement数据集
    columns_names = ['Label','region-centroid-col','region-centroid-row','region-pixel-count','short-line-density-5',
                     'short-line-density-2','vedge-mean','vegde-sd','hedge-mean','hedge-sd','intensity-mean','rawred-mean',
                     'rawblue-mean','rawgreen-mean','exred-mean','exblue-mean','exgreen-mean','value-mean','saturatoin-mean',
                     'hue-mean']
    data_1 = pd.read_csv("./data/segmentation/segmentation.data",
                        header = 1, names = columns_names)
    data_2 = pd.read_csv("./data/segmentation/segmentation.test",
                        header = 1, names = columns_names)
    data = pd.concat([data_1, data_2])
    label = list(set(data.iloc[:, 0]))
    print(label)
    label_nums = np.zeros(data.shape[0])  #label数字化
    for i in range(data.shape[0]):
        label_nums[i] = label.index(data.iloc[i, 0])
    data['Label'] = label_nums
    # data.to_csv('./data/segmentation/segmentation.csv', index = False)
    # data = pd.read_csv('./data/segmentation/segmentation.csv')
    data_partial_label = data_processing(data, candidate_set_length = 3)
    data_partial_label.to_csv('./data/segmentation/segmentation_partial.csv',
                              index = False)
