# -*- coding: utf-8 -*-

        
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from data_generate_scene import get_feature_id_map
from fm_model import Fm


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_input_data(input_file, batch_size, feature_id_map):
    size = len(feature_id_map)
    def parser(record):
        name_to_features = {
            "input_feature_ids": tf.FixedLenFeature([size], tf.int64),
            "labels": tf.FixedLenFeature([], tf.int64)
        }

        example = tf.parse_single_example(record, features=name_to_features)
        input_feature_ids = example["input_feature_ids"]
        labels = example["labels"]
        return input_feature_ids, labels

    dataset = tf.data.TFRecordDataset(input_file)
    # 数据类别集中，需要较大的buffer_size，才能有效打乱，或者再 数据处理的过程中进行打乱
    dataset = dataset.map(parser).batch(batch_size).shuffle(buffer_size=3000).repeat()
    iterator = dataset.make_one_shot_iterator()
    input_feature_ids, labels = iterator.get_next()
    return input_feature_ids, labels
 
        
def auc(prob,y_dev):
    assert len(prob) == len(y_dev)
    sort_list = sorted(zip(prob,y_dev))
    sum_r = 0
    m = 0
    n = 0
    for i in range(len(sort_list)):
        if sort_list[i][1] == 1:
            m += 1
            sum_r += i+1
    n = len(sort_list) - m
    if m == 0 or n == 0:
        return -1
    auc = (sum_r - (1+m)*m*0.5)/(m*n)
    return auc
        
        
def train(feature_id_map):
    lr = 0.001
    feature_id_size = len(feature_id_map)
    fm_embedding_size = 4
    input_feature_ids = tf.placeholder(tf.float32, shape=[None, feature_id_size], name='input_feature_ids')
    #1和-1
    labels = tf.placeholder(tf.float32, shape=[None], name='labels')
    fm = Fm(input_feature_ids, feature_id_size, fm_embedding_size, labels, lr)
    
    init_global = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=2)  # 保存最后top3模型

    #动态调整gpu资源
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(init_global)
        
        input_feature_ids_train, labels_train = get_input_data("./data/train_scene.tf_record", 32, feature_id_map)
        input_feature_ids_dev, labels_dev = get_input_data("./data/dev_scene.tf_record", 64, feature_id_map)
        step = 0
        step_list = []
        auc_list = []
        for epoch in range(10):
            #格式化输出 居中对齐
            print("{:*^100s}".format(("epoch-" + str(epoch)).center(20)))
            # 读取训练数据
            for i in range(int(20000/32)):
                step += 1
                id_train, y_train = sess.run([input_feature_ids_train, labels_train])
                feed = {fm.input_feature_ids: id_train, fm.labels: y_train}
                _, loss, prob = sess.run([fm.train_op, fm.loss, fm.prob],feed_dict = feed)
                
                if step % 50 == 0:
                    total_auc = 0
                    total_loss = 0
                    total_n = 0
                    num_dev_steps = int(2000/64)
                    for j in range(num_dev_steps):
                        id_dev, y_dev = sess.run([input_feature_ids_dev, labels_dev])
                        feed = {fm.input_feature_ids: id_dev, fm.labels: y_dev}
                        _, loss, prob = sess.run([fm.train_op, fm.loss, fm.prob],feed_dict = feed)
                        o_auc = auc(prob,y_dev)
                        if o_auc >=0 :
                            total_auc += o_auc
                            total_loss += loss
                            total_n += 1
                    
                    avg_auc = total_auc/total_n
                    avg_loss = total_loss/total_n
                    
                    step_list.append(step)
                    auc_list.append(avg_auc)
                    
                    print("epoch:{:<2}, step:{:<6}, loss:{:<10.6}, auc:{:<10.3}".format(epoch, step, avg_loss, avg_auc))
                    saver.save(sess, './model_scene/fm.ckpt', global_step=step)
        plt.figure(dpi=150)
        plt.plot(step_list, auc_list, 'r')
    
    
if __name__ == "__main__":   
    feature_id_map = get_feature_id_map('./data/feature_id_map_scene.txt')
    train(feature_id_map)
