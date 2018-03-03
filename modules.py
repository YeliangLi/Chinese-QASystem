#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
  Author:  Yeliang Li
  Blog: http://blog.yeliangli.com/
  Created: 2018/2/14
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.layers import base

class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:'PAD',1:'UNK'}
        self.n_words = 2 # Count PAD and UNK

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1   
        return self.word2index[word]

class Embedding(base.Layer):
    def __init__(self,embedding_shape,name,trainable=True):
        super(Embedding,self).__init__(trainable=trainable,name=name)
        self.embedding_shape = embedding_shape

    def build(self,_):
        assert len(self.embedding_shape) == 2, "The length of embedding_shape is not equal to 2."
        self.embedding = self.add_variable("embedding",
                                           self.embedding_shape,
                                           tf.float32
                                           )    

    def call(self, ids):
        outputs = tf.nn.embedding_lookup(self.embedding,ids)
        return outputs

class AttentionLayer(base.Layer):
    def __init__(self,
                 num_units,
                 score_value=float("-inf"),
                 trainable=True,
                 name="attentionLayer"):
        super(AttentionLayer,self).__init__(trainable=trainable,name=name)
        self.num_units = num_units
        self.score_value = score_value

    def build(self,input_shape):
        assert len(input_shape.as_list()) == 3, "The dimension of inputs is not equal to 3."
        self.W = self.add_variable("W",[input_shape[-1].value,self.num_units],tf.float32)
        self.v = self.add_variable("v",[self.num_units,1],tf.float32)

    def call(self,inputs,sequence_length=None): 
        W = tf.tile(tf.expand_dims(self.W,0),[tf.shape(inputs)[0],1,1])
        v = tf.tile(tf.expand_dims(self.v,0),[tf.shape(inputs)[0],1,1])
        scores = tf.squeeze(tf.matmul(tf.nn.tanh(tf.matmul(inputs,W)),v),
                           axis=2
                           )
        if sequence_length != None:
            mask = tf.sequence_mask(sequence_length,maxlen=tf.shape(inputs)[1])
            scores = tf.where(mask,scores,tf.ones_like(scores)*self.score_value)
        alignments = tf.nn.softmax(scores)
        r = tf.squeeze(tf.matmul(tf.expand_dims(alignments,axis=1),inputs),axis=1)
        return r
                     
class QuestionLSTM():
    def __init__(self,
                 num_units,
                 dropout_rate=0.0,
                 training = True,
                 name="questionLSTM"):
        self.dropout_rate = dropout_rate
        self.training = training
        self.name = name
        self.lstm_cell = tf.contrib.rnn.LSTMCell(num_units,use_peepholes=True)
        self.attention = AttentionLayer(num_units)

    def __call__(self,inputs,sequence_length):
        with tf.variable_scope(self.name):
            state = self.lstm_cell.zero_state(tf.shape(inputs)[0],tf.float32)       
            outputs,_ = tf.nn.dynamic_rnn(self.lstm_cell,
                                          inputs,
                                          sequence_length,
                                          state,
                                          tf.float32)
            outputs = tf.layers.dropout(outputs,self.dropout_rate,training=self.training)
            q_r = self.attention(outputs,sequence_length)
        return q_r
        
class EvidenceLSTMs():
    def __init__(self,
                 num_units,
                 dropout_rate = 0.0,
                 training = True,
                 name="evidenceLSTMs"):
        self.layer1 = tf.contrib.rnn.LSTMCell(num_units,use_peepholes=True,name="layer1")
        self.layer2 = tf.contrib.rnn.LSTMCell(num_units,use_peepholes=True,name="layer2")
        self.layer3 = tf.contrib.rnn.LSTMCell(num_units,use_peepholes=True,name="layer3")
        self.output_dense = tf.layers.Dense(len(tag_to_ix))
        self.dropout_rate = dropout_rate
        self.training = training
        self.name = name
        
    def __call__(self,inputs,sequence_length):
        with tf.variable_scope(self.name):
            state = self.layer1.zero_state(tf.shape(inputs)[0],tf.float32)
            layer1_outputs,_ = tf.nn.dynamic_rnn(self.layer1,
                                                 inputs,
                                                 sequence_length,
                                                 state,
                                                 tf.float32)
            layer1_outputs = tf.layers.dropout(layer1_outputs,self.dropout_rate,training=self.training)              
            layer1_outputs_reversed = tf.reverse_sequence(layer1_outputs,sequence_length,seq_dim=1)
            layer2_outputs,_ = tf.nn.dynamic_rnn(self.layer2,
                                                 layer1_outputs_reversed,
                                                 sequence_length,
                                                 state,
                                                 tf.float32)
            layer2_outputs = tf.reverse_sequence(layer2_outputs,sequence_length,seq_dim=1)
            layer2_outputs = tf.layers.dropout(layer2_outputs,self.dropout_rate,training=self.training)     
            layer3_inputs = tf.concat([layer1_outputs,layer2_outputs],axis=2)
            outputs,_ = tf.nn.dynamic_rnn(self.layer3,
                                          layer3_inputs,
                                          sequence_length,
                                          state,
                                          tf.float32)
            outputs = tf.layers.dropout(outputs,self.dropout_rate,training=self.training)
            outputs = self.output_dense(outputs)
        return outputs

tag_to_ix = {"B":0,"I":1,"O1":2,"O2":3,"START":4,"STOP":5}

def log_sum_exp(inputs):
    max_scores = tf.reduce_max(inputs,axis=1,keepdims=True) 
    return tf.squeeze(max_scores,1) + tf.log(tf.reduce_sum(tf.exp(inputs - max_scores),axis=1))

class CRF():
    def __init__(self,name="crf"):
        self.name = name
    def build(self):
        with tf.variable_scope(self.name):
            self.transitions = tf.get_variable("transitions",
                                               [len(tag_to_ix),len(tag_to_ix)],
                                               tf.float32)
            indices = []
            updates = []
            for i in range(len(tag_to_ix)):
                indices.append([i,tag_to_ix["START"]])
                indices.append([tag_to_ix["STOP"],i])
                updates += [-1e+8,-1e+8]
                
            self.transitions = tf.scatter_nd_update(self.transitions,indices,updates)
            self.init_alphas = tf.get_variable("init_alphas",
                                               [len(tag_to_ix)],
                                               tf.float32,
                                               tf.constant_initializer(-1e+8),
                                               trainable=False)
            self.init_alphas = tf.scatter_update(self.init_alphas,
                                                 [tag_to_ix["START"]],
                                                 [tf.constant(0.0)])
            
    def neg_log_likelihood(self,inputs,sequence_length,targets):
        sequence_length = tf.to_int32(sequence_length)
        targets = tf.to_int32(targets)
        i0 = tf.constant(0,tf.int32)
        alphas_0 = tf.tile(tf.expand_dims(self.init_alphas,0),[tf.shape(inputs)[0],1])
        max_seq_len = tf.reduce_max(sequence_length)
        alphas_array = tf.TensorArray(tf.float32,size=max_seq_len)
        
        scores_array = tf.TensorArray(tf.float32,size=max_seq_len)
        initial_scores = alphas_0[:,tag_to_ix["START"]]
        targets = tf.concat([tf.ones([tf.shape(inputs)[0],1],tf.int32)*tag_to_ix["START"],
                             targets],
                            axis=1)
        def body(i,alphas_t,ta1,scores,ta2):
            inp = inputs[:,i,:] #(batch_size,len(tag_to_ix))
            emit_scores = tf.tile(tf.expand_dims(inp,1),[1,len(tag_to_ix),1])
            forward_vars = tf.tile(tf.expand_dims(alphas_t,2),[1,1,len(tag_to_ix)])
            next_tag_vars = forward_vars + self.transitions + emit_scores
            alphas_t_plus_1 = log_sum_exp(next_tag_vars) #(batch_size,len(tag_to_ix))
            
            indices = tf.stack([tf.range(0,tf.shape(inputs)[0],dtype=tf.int32),
                                targets[:,i+1]])
            indices = tf.transpose(indices,[1,0])
            scores += tf.gather_nd(self.transitions,targets[:,i:i+2]) +\
                       tf.gather_nd(inp,indices)
            scores = tf.reshape(scores,[tf.shape(inputs)[0]])
            return i+1,alphas_t_plus_1,ta1.write(i,alphas_t_plus_1),scores,ta2.write(i,scores)
        _,_,ta1_final,_,ta2_final = tf.while_loop(lambda i,alphas_t,ta1,scores,ta2: i < max_seq_len,
                                                  body,
                                                  loop_vars=[i0,alphas_0,alphas_array,initial_scores,scores_array])
        
        ta1_final_result = ta1_final.stack()
        indices1 = tf.stack([sequence_length-1,
                            tf.range(0,tf.shape(inputs)[0],dtype=tf.int32)])
        indices1 = tf.transpose(indices1,[1,0])
        terminal_vars = tf.gather_nd(ta1_final_result,indices1) + self.transitions[:,tag_to_ix["STOP"]]
        forward_scores = log_sum_exp(terminal_vars) #(batch_size)
        
        ta2_final_result = ta2_final.stack()
        indices2 = tf.stack([tf.gather_nd(tf.transpose(targets,[1,0]),indices1),
                             tf.ones([tf.shape(inputs)[0]],tf.int32)*tag_to_ix["STOP"]])
        indices2 = tf.transpose(indices2,[1,0])
        gold_scores = tf.gather_nd(ta2_final_result,indices1) +\
                      tf.gather_nd(self.transitions,indices2)
        
        return forward_scores - gold_scores # (batch_size,)
      
    def viterbi_decode(self,inputs,sequence_length):
        sequence_length = tf.to_int32(sequence_length)
        alphas_0 = tf.tile(tf.expand_dims(self.init_alphas,0),
                           [tf.shape(inputs)[0],1]) #(batch_size,len(tag_to_ix))
        i0 = tf.constant(0,tf.int32)
        max_seq_len = tf.reduce_max(sequence_length)
        best_tag_ids_array = tf.TensorArray(tf.int32,size=max_seq_len)
        alphas_array = tf.TensorArray(tf.float32,size=max_seq_len)
        
        def body1(i,alphas_t,ta1,ta2):
            forward_vars = tf.tile(tf.expand_dims(alphas_t,2),[1,1,len(tag_to_ix)])
            next_tag_vars = forward_vars + self.transitions
            best_tag_ids = tf.argmax(next_tag_vars,axis=1,output_type=tf.int32) #(batch_size,len(tag_to_ix))
            alphas_t_plus_1 = tf.reduce_max(next_tag_vars,axis=1) + inputs[:,i,:] #(batch_size,len(tag_to_ix))
            return i+1,alphas_t_plus_1,ta1.write(i,best_tag_ids),ta2.write(i,alphas_t_plus_1)
        _,_,ta1_final,ta2_final = tf.while_loop(lambda i,alpha_t,ta1,ta2: i < max_seq_len,
                                                body1,
                                                loop_vars=[i0,alphas_0,best_tag_ids_array,alphas_array])
        
        bptrs = ta1_final.stack()
    
        ta2_final_result = ta2_final.stack()
        indices = tf.stack([sequence_length-1,
                           tf.range(0,tf.shape(inputs)[0],dtype=tf.int32)])
        indices = tf.transpose(indices,[1,0])
        terminal_vars = tf.gather_nd(ta2_final_result,indices) + self.transitions[:,tag_to_ix["STOP"]]  
        terminal_id = tf.argmax(terminal_vars,axis=1,output_type=tf.int32) #(batch_size)
        
        path_array = tf.TensorArray(tf.int32,size=tf.shape(inputs)[0])
        def body2(i,ta):
            def sub_body(j,past_id,best_path):
                best_tag_id = tf.gather_nd(bptrs,[[j,i]]) #(1,len(tag_to_ix))
                next_id = tf.gather_nd(best_tag_id,[[0,past_id]])
                best_path = tf.concat([next_id,best_path],axis=0)
                return j-1,next_id[0],best_path
            _,_,best_path = tf.while_loop(lambda j,past_id,best_path: j >= 0,
                                          sub_body,
                                          loop_vars=[sequence_length[i] - 1,terminal_id[i],terminal_id[i:i+1]],
                                          shape_invariants=[tf.TensorShape([]),tf.TensorShape([]),tf.TensorShape([None])]
                                          )
            best_path = tf.pad(best_path,[[0,max_seq_len-sequence_length[i]]],constant_values=-1)
            return i+1,ta.write(i,best_path)
        _,ta_final = tf.while_loop(lambda i,ta: i < tf.shape(inputs)[0],
                                   body2,
                                   loop_vars=[i0,path_array])
        
        best_paths = ta_final.stack() #shape=(batch_size,1+max_seq_len),padding_value=-1
        return best_paths[:,1:]
    
    @classmethod
    def compute_accuracy(cls,predicts,targets):
        '''
        predicts and targets are instances of numpy.
        
        predicts shape:(batch_size,time_step)
        targets shape: (batch_size,time_step)
        '''
        assert predicts.shape == targets.shape, "The dimensions of predicts and targets don't match."
        assert len(predicts.shape) == 2, "The dimension of predicts don't equal to 2."
        count = 0
        truth_ans_pos = []
        hypothesis_ans_pos = []
        for i in range(targets.shape[0]):
            target_answer_pos = []
            predict_answer_pos = []            
            for j in range(targets.shape[1]):
                if targets[i,j] == tag_to_ix["B"]:
                    target_answer_pos.append(j)
                    for k in range(j+1,targets.shape[1]):
                        if targets[i,k] != tag_to_ix["I"]:
                            target_answer_pos.append(k-1)
                            break
                    break
            for j in range(predicts.shape[1]):
                if predicts[i,j] == tag_to_ix["B"]:
                    predict_answer_pos.append(j)
                    for k in range(j+1,predicts.shape[1]):
                        if predicts[i,k] != tag_to_ix["I"]:
                            predict_answer_pos.append(k-1)
                            break
                    break
            if target_answer_pos == predict_answer_pos:
                    count += 1
            hypothesis_ans_pos.append(predict_answer_pos)
            truth_ans_pos.append(target_answer_pos)
        acc = count / targets.shape[0]
        return acc,hypothesis_ans_pos,truth_ans_pos