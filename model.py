# -*- coding: utf-8 -*-
import tensorflow as tf

class Fm(object):
    
    def __init__(self, input_feature_ids, feature_id_size, fm_embedding_size, labels, lr):
        self.input_feature_ids = input_feature_ids
        self.labels = labels
        self.create_fm(input_feature_ids, feature_id_size, fm_embedding_size, labels)
        self.optimize(lr)
        
    def create_fm(self, input_feature_ids, feature_id_size, fm_embedding_size, y):
        fm_embedding_table = tf.get_variable(name = 'fm_embedding',shape = [feature_id_size, fm_embedding_size])
        fm_w = tf.get_variable(name = 'fm_w',shape = [feature_id_size])
        fm_b = tf.get_variable(name = 'fm_b',shape = [1])
        # (a1 + a2 + ...)^2
        part_1 = tf.reduce_sum(tf.square(tf.matmul(input_feature_ids, fm_embedding_table)),axis = 1)
        # a1^2 + a2^2 + ...
        part_2 = tf.reduce_sum(tf.matmul(tf.square(input_feature_ids),tf.square(fm_embedding_table)),axis = 1)
        fm_cross = (part_1 - part_2) * 0.5
        y_p = tf.reduce_sum(input_feature_ids * fm_w, axis = 1) + fm_b + fm_cross
        batch_loss = -tf.log(tf.nn.sigmoid(y_p*y))
        loss = tf.reduce_mean(batch_loss)
        prob = tf.nn.sigmoid(y_p)
        self.loss = loss
        self.prob = prob
        
    def optimize(self, lr):
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(self.loss)
        self.train_op = train_op
