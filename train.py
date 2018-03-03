#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
  Author:  Yeliang Li
  Blog: http://blog.yeliangli.com/
  Created: 2018/2/14
"""

import tensorflow as tf
import numpy as np
from sklearn.externals import joblib
import os
import modules

tf.app.flags.DEFINE_integer("batch_size",100,"batch size for training")
tf.app.flags.DEFINE_integer("epoches",17,"epoches for training")
tf.app.flags.DEFINE_integer("buffer_size",500000,"representing the number of bytes in the read buffer")
tf.app.flags.DEFINE_integer("num_parallel_calls",100," representing the number elements to process in parallel")
tf.app.flags.DEFINE_integer("word_embedding",64,"word embedding")
tf.app.flags.DEFINE_integer("feature_embedding",2,"feature embedding")
tf.app.flags.DEFINE_integer("cell_size",64,"cell size of LSTM")
tf.app.flags.DEFINE_float("dropout_rate",0.05,"dropout rate for output of all the LSTM layers")
tf.app.flags.DEFINE_bool("training",True,"whether to train the model")
tf.app.flags.DEFINE_float("lr",0.005,"learning rate for training")
tf.app.flags.DEFINE_float("decay_steps",8000,"decay steps for learning rate")
tf.app.flags.DEFINE_float("decay_rate",0.75,"decay rate for learning rate")
FLAGS = tf.app.flags.FLAGS

def parse(serialized):
    features = {}
    features["question"] = tf.VarLenFeature(tf.int64)
    features["evidence"] = tf.VarLenFeature(tf.int64)
    features["evidence_tags"] = tf.VarLenFeature(tf.int64)
    features["q_e_comm"] = tf.VarLenFeature(tf.int64)
    features["e_e_comm"] = tf.VarLenFeature(tf.int64)
    features["question_length"] = tf.FixedLenFeature([1],tf.int64)
    features['evidence_length'] = tf.FixedLenFeature([1],tf.int64)
    features = tf.parse_single_example(serialized,features)
    question = tf.sparse_tensor_to_dense(features["question"])
    evidence = tf.sparse_tensor_to_dense(features["evidence"])
    evidence_tags = tf.sparse_tensor_to_dense(features["evidence_tags"])
    q_e_comm = tf.sparse_tensor_to_dense(features["q_e_comm"])
    e_e_comm = tf.sparse_tensor_to_dense(features["e_e_comm"])
    return question,evidence,evidence_tags,q_e_comm,e_e_comm,features["question_length"],features["evidence_length"]
    

def main(_):
    lang = joblib.load("./data/lang.pkl")
    files = ["./data/trainData.tfrecords","./data/validData.tfrecords"]
    padded_shapes = tuple([[-1]]*5+[[1]]*2)
    padding_values = tuple([np.int64(0)]*2+[np.int64(-1)]+[np.int64(0)]*4)
    questions_list = []
    evidences_list = []
    targets_tags_list = []
    q_e_comm_feas_list = []
    e_e_comm_feas_list = []
    question_length_list = []
    evidence_length_list = []
    
    for i in range(2):
        dataset = tf.data.TFRecordDataset(files[i],buffer_size=FLAGS.buffer_size).map(parse,FLAGS.num_parallel_calls).padded_batch(FLAGS.batch_size,padded_shapes,padding_values).repeat(FLAGS.epoches)
        iterator = dataset.make_one_shot_iterator()
        input_elements = iterator.get_next()
        questions_list.append(input_elements[0])
        evidences_list.append(input_elements[1])
        targets_tags_list.append(input_elements[2])
        q_e_comm_feas_list.append(input_elements[3])
        e_e_comm_feas_list.append(input_elements[4])
        question_length_list.append(tf.reshape(input_elements[5],[-1,]))
        evidence_length_list.append(tf.reshape(input_elements[6],[-1,])) 
    
    #model definition
    training = tf.placeholder(tf.bool)
    word_embedding = modules.Embedding([lang.n_words,FLAGS.word_embedding],"word_embedding")
    q_e_comm_feas_embedding = modules.Embedding([2,FLAGS.feature_embedding],
                                                "q_e_comm_feas_embedding")
    e_e_comm_feas_embedding = modules.Embedding([2,FLAGS.feature_embedding],
                                                "e_e_comm_feas_embedding")
    question_LSTM = modules.QuestionLSTM(FLAGS.cell_size,
                                         FLAGS.dropout_rate,
                                         training)
    evidence_LSTMs = modules.EvidenceLSTMs(FLAGS.cell_size,FLAGS.dropout_rate,training)
    crf = modules.CRF()
    
    with tf.device("/gpu:0"):
        reuse = False
        decoded_list = []        
        for i in range(2):
            with tf.variable_scope("model",reuse=reuse):
                questions = word_embedding(questions_list[i]) #(batch_size,max_ques_length,FLAGS.word_embedding)
                evidences = word_embedding(evidences_list[i]) #(batch_size,max_evidences_length,FLAGS.word_embedding)
                q_e_comm_feas = q_e_comm_feas_embedding(q_e_comm_feas_list[i]) #(batch_size,max_q_e_comm_feas_length,FLAGS.feature_embedding)
                e_e_comm_feas = e_e_comm_feas_embedding(e_e_comm_feas_list[i]) #(batch_size,max_e_e_comm_feas_length,FLAGS.feature_embedding)
                q_r = question_LSTM(questions,question_length_list[i]) #(batch_size,FLAGS.cell_size)
                q_r = tf.tile(tf.expand_dims(q_r,axis=1),[1,tf.shape(evidences)[1],1])
                x = tf.concat([evidences,q_r,q_e_comm_feas,e_e_comm_feas],axis=2)
                outputs = evidence_LSTMs(x,evidence_length_list[i])
                crf.build()
                decoded = crf.viterbi_decode(outputs,evidence_length_list[i]) #(batch_size,max_evidences_length)
                decoded_list.append(decoded)
                if i == 0:
                    logits = crf.neg_log_likelihood(outputs,evidence_length_list[i],targets_tags_list[i]) #(batch_size,)
                    loss = tf.reduce_sum(logits)
                    loss /= tf.to_float(tf.shape(evidences)[0])
                    with tf.name_scope("optimizer"):
                        global_step = tf.Variable(tf.constant(0,tf.int32),trainable=False)
                        lr = tf.train.exponential_decay(FLAGS.lr,
                                                        global_step,
                                                        FLAGS.decay_steps,
                                                        FLAGS.decay_rate,
                                                        staircase=True)
                        optimizer = tf.train.RMSPropOptimizer(lr)
                        tvars = tf.trainable_variables()
                        grads = tf.gradients(loss,tvars)
                        train_op = optimizer.apply_gradients(zip(grads,tvars),global_step)                    
                reuse = True
        
        record_loss = tf.placeholder(tf.float32)
        record_accuracy = tf.placeholder(tf.float32)
        train_merged = []
        train_merged.append(tf.summary.scalar("train_loss",record_loss))
        train_merged.append(tf.summary.scalar("train_accuracy",record_accuracy))
        train_merged = tf.summary.merge(train_merged)
        valid_summary = tf.summary.scalar("valid_accuracy",record_accuracy)
            
    saver = tf.train.Saver(var_list=tvars)
    log_device_placement=True 
    allow_soft_placement=True 
    config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)  
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        train_fileWriter = tf.summary.FileWriter("logs/train",sess.graph)
        valid_fileWriter = tf.summary.FileWriter("logs/valid",sess.graph)
        train_size = joblib.load("./data/trainDataSize.pkl")
        valid_size = joblib.load("./data/validDataSize.pkl")
        assert train_size % FLAGS.batch_size == 0,"train_size can't be divisible by batch."
        assert valid_size % FLAGS.batch_size == 0,"valid_size can't be divisible by batch."
        try:
            for i in range(FLAGS.epoches):
                total_loss = 0.0
                total_accuracy = 0.0            
                for j in range(train_size // FLAGS.batch_size):
                    _,cost,hypothesis,truth,step = sess.run([train_op,loss,decoded_list[0],targets_tags_list[0],global_step],{training:FLAGS.training})
                    accuracy,_,_ = crf.compute_accuracy(hypothesis,truth)
                    total_loss += cost
                    total_accuracy += accuracy 
                    print("global_step:%d epoch:%d batch:%d train_loss:%f train_accuracy:%f" 
                          %(step,i+1,j+1,cost,accuracy))
                    if step % 500 == 0:
                        saver.save(sess,"./model/QASystem",step)
                total_loss /= (train_size / FLAGS.batch_size)
                total_accuracy /= (train_size / FLAGS.batch_size)
                summary = sess.run(train_merged,{record_loss:total_loss,record_accuracy:total_accuracy})
                train_fileWriter.add_summary(summary,i+1)
                
                total_loss = 0.0
                total_accuracy = 0.0
                print("\n")
                for j in range(valid_size // FLAGS.batch_size):
                    hypothesis,truth = sess.run([decoded_list[1],targets_tags_list[1]],{training:False})
                    accuracy,_,_ = crf.compute_accuracy(hypothesis,truth)
                    total_accuracy += accuracy 
                    print("epoch:%d batch:%d valid_accuracy:%f" %(i+1,j+1,accuracy))
                total_accuracy /= (valid_size / FLAGS.batch_size)
                summary = sess.run(valid_summary,{record_accuracy:total_accuracy})
                valid_fileWriter.add_summary(summary,i+1)
                print("\n") 
        except BaseException:
            saver.save(sess,"./model/QASystem")
        saver.save(sess,"./model/QASystem")
       
if __name__ == "__main__":
    tf.app.run()
