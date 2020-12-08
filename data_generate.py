# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

feature_dict = {'性别':['男','女'],
                    '年龄':['0-18','19-24','25-30','31-35','36-40','41+'],
                    '城市':[0,1,2,3,4,5,6,7,8,9],
                    '车型':['迈腾', '速腾', '捷达', '宝来', '高尔夫', '奥迪'],
                    '时间':[0,1,2,3,4,5,6,7,8,9,10,11],
                    '星期':[0,1,2,3,4,5,6],
                    '油量':['0-30%', '30%-60%', '60%-100%'],
                    '目的地类型':['商圈','家','公司','机场','高铁','景点'],
                    '目的地附近有无洗车店':[0,1],
                    '目的地附近有无停车场':[0,1],
                    '天气状况':['晴天','多云','小雨','大雨'],
                    '车速':['0-20','21-40','40-60','60-100','100+'],
                    '场景':[0,1,2,3,4,5,6,7,8,9,10,11]
                    }
labels = [-1, 1]


def generate_one_data():
    data = []
    for key in feature_dict:
        value = feature_dict[key]
        n = len(value)
        interval = 1/n
        r = np.random.rand()
        for i in range(n):
            a = i*interval
            b = (i+1)*interval
            if i == n-1:
                b = 1
            if a <= r < b:
                data.append(key+'#'+str(value[i]))
                break
    return ','.join(data)


def generate_data(train_path, dev_path):
    train = open(train_path, 'w', encoding='utf-8')
    dev = open(dev_path, 'w', encoding='utf-8')
    for i in range(20000):
        data = generate_one_data()
        label = '-1' if np.random.rand() < 0.5 else '1'
        train.write(data + '\t' + label + '\n')
    train.close()
    
    for i in range(2000):
        data = generate_one_data()
        label = '-1' if np.random.rand() < 0.5 else '1'
        dev.write(data + '\t' + label + '\n')
    dev.close()
    
    
def prepare_tf_record_data(feature_id_map, path, out_path):
    """
        生成训练数据， tf.record, 单标签分类模型, 随机打乱数据
    """
    writer = tf.python_io.TFRecordWriter(out_path)
    example_count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            arr = line.strip().split('\t')
            features = arr[0]
            label = [int(arr[1])]
            feature_list = [0]*len(feature_id_map)
            for f in features.split(','):
                index = feature_id_map[f]
                feature_list[index] = 1
                
            features["input_feature_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=feature_list))
            features["labels"] = tf.train.Feature(int64_list=tf.train.Int64List(value=label))

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
            example_count += 1

    print("total example:", example_count)
    writer.close()
    
    
def get_feature_id_map(path):
    feature2id = {}
    i=0
    with open(path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if line.strip() == '':
                break
            feature = line.strip()
            if feature not in feature2id:
                feature2id[feature] = i
                i += 1
    return feature2id



if __name__ == "__main__":    
    train_file = './data/train_data.txt'
    dev_file = './data/dev_data.txt'
    generate_data(train_file, dev_file)
    feature_id_map = get_feature_id_map('./data/feature_id_map.txt')
    prepare_tf_record_data(feature_id_map, path=train_file, out_path="./data/train.tf_record")
    prepare_tf_record_data(feature_id_map, path=dev_file, out_path="./data/dev.tf_record")