#!/usr/bin/python
# coding:utf8
"""
@author: Jin Zitian
@time: 2020-11-11 16:55
"""
import os
import tensorflow as tf

from data_generate import get_feature_id_map

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_model(model_folder):
    # We retrieve our checkpoint fullpath
    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        #input_checkpoint = 'bert.ckpt-3120' 类似这种样子
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except Exception as e:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder, repr(e))

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    tf.reset_default_graph()
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We start a session and restore the graph weights
    sess_ = tf.Session()
    saver.restore(sess_, input_checkpoint)

    # opts = sess_.graph.get_operations()
    # for v in opts:
    #     print(v.name)
    return sess_


model_path = "./model/"
sess = load_model(model_path)
input_feature_ids = sess.graph.get_tensor_by_name("input_feature_ids:0")
prob = sess.graph.get_tensor_by_name("Sigmoid_1:0")


def predict(features_list, feature_id_map):
    # 逐个分成 最大62长度的 text 进行 batch 预测
    feature_ids_list = []
    for features in features_list:    
        feature_list = [0]*len(feature_id_map)
        for f in features.split(','):
            index = feature_id_map[f]
            feature_list[index] = 1
        feature_ids_list.append(feature_list)
    feed = {input_feature_ids: feature_ids_list}

    probs = sess.run(prob, feed)
    return probs



if __name__ == "__main__":
    feature_list = ['性别#男,年龄#0-18,城市#5,车型#奥迪,时间#2,星期#6,油量#0-30%,目的地类型#高铁,目的地附近有无洗车店#1,目的地附近有无停车场#0,天气状况#大雨,车速#21-40,场景#3',
                    '性别#男,年龄#31-35,城市#1,车型#迈腾,时间#10,星期#1,油量#0-30%,目的地类型#高铁,目的地附近有无洗车店#0,目的地附近有无停车场#0,天气状况#多云,车速#60-100,场景#0']
    scene_list = ['场景#3','场景#1']
    feature_id_map = get_feature_id_map('./data/feature_id_map.txt')
    probs = predict(feature_list, feature_id_map)
    sorted(zip(scene_list, probs),key = lambda x:-x[1])
    