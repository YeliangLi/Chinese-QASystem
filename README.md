# Chinese-QASystem
Chinese question answering system based on BLSTM and CRF.
<br>
<br>

Requirement
-------
tensorflow 1.5<br>
numpy<br>
thulac<br>
scikit-learn<br>
matplotlib<br><br>

DataSet
-----
百度的中文问答数据集[WebQA](https://www.spaces.ac.cn/archives/4338/)，非常感谢该链接的作者对数据的整理。<br><br>

How to get start?
-----
    1.Download the raw data and extract it to the folder where the source code is located.
    2.python3 make_tfrecords.py
      Processing the raw data to generate the tfrecord files for training and testing.
    3.python3 train.py
      Training model.
    4.python3 plot.py
      Draw the loss curve and the accuracy curve about training.

<br><br>
Note
----
The code is still a development version. Because I only use about 20,000 corpus in the WebQA training set for training,so it still has not reached the high accuracy on the validation set.
In the future, I will increase the amount of training data to provide a more complete version.I will write a blog to introduce this work and you will learn how to use tensorflow's tf.while_loop 
interface to implement conditional random field training and Viterbi decoding.<br><br>
![](https://github.com/YeliangLi/Chinese-QASystem/raw/dev/picture/train_loss.png)<br> 
![](https://github.com/YeliangLi/Chinese-QASystem/raw/dev/picture/train_acc.png)<br>
![](https://github.com/YeliangLi/Chinese-QASystem/raw/dev/picture/valid_acc.png)<br><br>

References
-----
[Li P, Li W, He Z, et al. Dataset and Neural Recurrent Sequence Labeling Model for Open-Domain Factoid Question Answering[J]. 2016.](https://arxiv.org/abs/1607.06275)


