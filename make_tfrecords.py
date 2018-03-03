#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
  Author:  Yeliang Li
  Blog: http://blog.yeliangli.com/
  Created: 2018/2/14
"""

import tensorflow as tf
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import os
import json
import thulac
import re
import modules


path = "./WebQA.v1.0"
#{"B":0,"I":1,"O1":2,"O2":3,"START":4,"STOP":5}
tag_to_ix = modules.tag_to_ix 
evidences_sampling_rate = 0.8

def write(lang,file,writer,data_size):
    thul = thulac.thulac(seg_only=True)
    if not os.path.isdir("data"):
        os.mkdir("data")
    with open(file,'rt',encoding='utf-8') as file:
        dict = file.read()
    dict = json.loads(dict,encoding='utf-8')
    count = 0
    number = 0
    for key in dict:
        if count == data_size:
            break
        ques = []
        for word in thul.cut(dict[key]['question'],text=True).split():
            index = lang.addWord(word)
            ques.append(index)
        evidences = []
        evidences_tags = []
        for record in dict[key]['evidences']:
            number += 1
            print("number:%d question number:%s evidence number:%s" %(number,key,record))
            answer = dict[key]['evidences'][record]['answer'][0]
            evidence = dict[key]['evidences'][record]['evidence']
            if answer != 'no_answer':
                evidence = re.sub(answer,"XXX",evidence)
                answer = thul.cut(answer,text=True).split()
                answer_tags = []
                answer_indices = []
                for i in range(len(answer)):
                    if i != 0:
                        answer_tags.append(tag_to_ix['I'])
                    else:
                        answer_tags.append(tag_to_ix['B'])
                    answer_indices.append(lang.addWord(answer[i]))
                evidence = thul.cut(evidence,text=True).split()
                evidence_indices = []
                evidence_tags = []
                before_answer = True
                for word in evidence:
                    if word != "XXX":
                        evidence_indices.append(lang.addWord(word))
                    else:
                        evidence_indices += answer_indices
                        evidence_tags += answer_tags
                        before_answer = False
                        continue                        
                    if before_answer:
                        evidence_tags.append(tag_to_ix["O1"])
                    else:
                        evidence_tags.append(tag_to_ix["O2"])    
                evidences.append(evidence_indices)
                evidences_tags.append(evidence_tags)
            else:
                evidence_indices = []
                evidence_tags = []                
                for word in thul.cut(evidence,text=True).split():
                    evidence_indices.append(lang.addWord(word))
                    evidence_tags.append(tag_to_ix["O1"])
                evidences.append(evidence_indices)
                evidences_tags.append(evidence_tags)                
        selected_evidences,rest_evidences,selected_evidences_tags,rest_evidences_tags = train_test_split(evidences,
                                                                                                         evidences_tags,
                                                                                                         test_size=1-evidences_sampling_rate,
                                                                                                         random_state=0
                                                                                                         )      
        count += len(selected_evidences) 
        if count > data_size:
            count -= len(selected_evidences)
            selected_evidences = selected_evidences[0:(data_size - count)]
            count = data_size
        for i in range(len(selected_evidences)):
            e_e_comm_fea = []
            q_e_comm_fea = []        
            for index in selected_evidences[i]:
                if index in ques:
                    q_e_comm_fea.append(1)   
                else:
                    q_e_comm_fea.append(0)
                comm_tag = False
                for evidence in rest_evidences:
                    if index in evidence:
                        e_e_comm_fea.append(1)
                        comm_tag = True
                        break
                if not comm_tag:
                    e_e_comm_fea.append(0)
            feas = {}
            feas['question'] = tf.train.Feature(int64_list=tf.train.Int64List(value=ques))
            feas['evidence'] = tf.train.Feature(int64_list=tf.train.Int64List(value=selected_evidences[i]))
            feas['evidence_tags'] = tf.train.Feature(int64_list=tf.train.Int64List(value=selected_evidences_tags[i]))
            feas['q_e_comm'] = tf.train.Feature(int64_list=tf.train.Int64List(value=q_e_comm_fea))
            feas['e_e_comm'] = tf.train.Feature(int64_list=tf.train.Int64List(value=e_e_comm_fea))
            feas['question_length'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(ques)]))
            feas['evidence_length'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[len(selected_evidences[i])]))
            features_to_write = tf.train.Example(features=tf.train.Features(feature=feas))
            writer.write(features_to_write.SerializeToString()) 
    writer.close()
    return count
    

if __name__ == "__main__":
    lang = modules.Lang("Chi")
    if not os.path.isdir("data"):
        os.mkdir("data")

    writer = tf.python_io.TFRecordWriter("./data/trainData.tfrecords")
    train_data_size = write(lang,os.path.join(path,"me_train.json"),writer,200000)
    joblib.dump(train_data_size,"./data/trainDataSize.pkl")
       
    writer = tf.python_io.TFRecordWriter("./data/validData.tfrecords")
    valid_data_size = write(lang,os.path.join(path,"me_validation.ir.json"),writer,5000)
    joblib.dump(valid_data_size,"./data/validDataSize.pkl")
      
    joblib.dump(lang,"./data/lang.pkl")
   
