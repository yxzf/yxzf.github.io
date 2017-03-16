---
layout: post
title: DNN在CTR预估上的应用
categories:
- deeplearning
tags:
- DNN
- CTR
image:
    teaser: /development/git_branch.png
---

#### CTR预估
CTR预估是计算广告中最核心的算法之一，那么CTR预估是指什么呢？简单来说，CTR预估是对每次广告的点击情况做出预测，预测用户是点击还是不点击。具体定义可以参考 [CTR](https://en.wikipedia.org/wiki/Click-through_rate)

#### 数据
CTR的样本标签一般直接从是否点击获得，特征则会考虑很多，例如用户的人口学特征、广告自身特征、广告展示特征等。这些特征中会用到很多类别特征，例如用户所属职业、广告展示的IP地址等。一般对于类别特征会采样One-Hot编码，例如职业有三种：学生、白领、工人，那么会会用一个长度为3的向量分别表示他们：[1, 0, 0]、[0, 1, 0]、[0, 0, 1]. 可以这样会使得特征维度扩展很大，同时特征会非常稀疏。目前很多公司的广告特征库都是上亿级别的。


#### 深度学习
深度学习目前在图像、语音、自然语言等领域应用比较多，因为这些领域的数据一般是连续的，局部之间存在某些结构。比如，图像的局部与其周围存在着紧密的联系；语音和文字的前后存在强相关性。但是CTR预估的数据如前面介绍，是非常离散的，特征前后之间的关系很多是我们排列的结果，并非本身是相互联系的。

##### Embeding
Neural Network是典型的连续值模型，而CTR预估的输入更多时候是离散特征。所以一个自然的想法就是如何将将离散特征转换为连续特征。如果你对词向量模型熟悉的话，可以发现之间的共通点。在自然语言处理(NLP)中，为了将自然语言交给机器学习中的算法来处理，通常需要首先将语言数学化，词向量就是用来将语言中的词进行数学化的一种方式。

一种最简单的词向量方式是one-hot，但这么做不能很好的刻画词之间的关系(例如相似性)，另外规模变大，带来维度灾难。因此Embeding的方法被提出，基本思路是将词都映射成一个固定长度的向量(向量大小远小于one-hot编码向量大些)，向量中元素不再是只有一位是1，而是每一位都有值。将所有词向量放在一起就是一个词向量空间，这样就可以表达词之间的关系，同时达到降维的效果。
![](/images/deeplearning/dnn_ctr_fig/embeding.png =100*100)
那么接下来就是如何进行Embeding
###### FM Embeding

![](/images/deeplearning/dnn_ctr_fig/fm.png)
FM在后半部分的交叉项中为每个特征都分配一个特征向量V，这样可以看做一种Embeding的方法。那么可以先用FM训练一个模型得到特征的Embeding表达，再将其输入到DNN中。Dr.Zhang在文献[1]中设计的模型是：
![](/images/deeplearning/dnn_ctr_fig/fnn.png)
模型中做了一个假设，就是每个category field只有一个值为1。field是指特征的种类，例如将特征occupation one-hot之后是三维向量，但这个向量都属于一个field，就是occupation。这样虽然离散化后的特征有几亿，但是category field一般是几十到几百。
模型得到每个特征的Embeding向量后，将特征归纳到其属于field，得到向量z，z的大小就是1+#fields * #embeding 。z是一个固定长度的向量之后再在上面套用DNN得到FNN模型。

Dr.Zhang在上面模型的基础上又提出了下面的新模型PNN.
PNN和FNN的主要不同在于除了得到z向量，还增加了一个p向量，即Product向量。Product向量由每个category field的feature vector做inner product 或则 outer product 得到，作者认为这样做有助于特征交叉。另外PNN中Embeding层不再由FM生成，可以在整个网络中训练得到。
![](/images/deeplearning/dnn_ctr_fig/pnn.png)

###### NN Embeding
![](/images/deeplearning/dnn_ctr_fig/wide&deep.png)
Google团队最近提出Wide and Deep Model。在他们的模型中，Wide Models其实就是LR模型，输入原始的特征和一些交叉组合特征；Deep Models通过Embeding层将稀疏的特征转换为稠密的特征，再使用DNN。最后将两个模型Join得到整个大模型，他们认为模型具有memorization and generalization特性。
Wide and Deep Model中原始特征既可以是category，也可以是continue，这样更符合一般的场景。另外Embeding层是将每个category特征分别映射到embeding size的向量，如他们在TensorFlow代码中所示：
```
deep_columns = [
  tf.contrib.layers.embedding_column(workclass, dimension=8),
  tf.contrib.layers.embedding_column(education, dimension=8),
  tf.contrib.layers.embedding_column(gender, dimension=8),
  tf.contrib.layers.embedding_column(relationship, dimension=8),
  tf.contrib.layers.embedding_column(native_country, dimension=8),
  tf.contrib.layers.embedding_column(occupation, dimension=8),
  age, education_num, capital_gain, capital_loss, hours_per_week]
```
##### 结合图像
目前很多在线广告都是图片形式的，文献[4]提出将图像也做为特征的输入。这样原始特征就分为两类，图像部分使用CNN，非图像部分使用NN处理。
其实这篇文章并没有太多新颖的方法，只能说多了一种特征。对于非图像特征，作者直接使用全连接神经网络，并没有使用Embeding。
![](/images/deeplearning/dnn_ctr_fig/conv_ctr.png)

##### CNN
CNN用于提取局部特征，在图像、NLP都取得不错的效果，如果在CTR预估中使用却是个难题。我认为最大的困难时如何构建对一个样本构建如图像那样的矩阵，能够具有局部联系和结构。如果不能构造这样的矩阵，使用CNN是没有什么意思的。
文献[5]是发表在CIKM2015的一篇短文，文章提出对使用CNN来进行CTR预估进行了尝试。
一条广告展示(single ad impression)包括：element = (user; query; ad, impression time, site category, device type, etc)
用户是否点击一个广告与用户的历史ad impression有关。这样，一个样本将会是(s, label) ，s由多条l组成(数目不定)
![](/images/deeplearning/dnn_ctr_fig/s_matrix.png)

作者提出CCPM模型处理这样的数据。每个样本有n个element，对每个element使用embeding 得到定长为d的向量$e_i\in R^d$，再构造成一个矩阵$s\in R^{d* n}$，得到s矩阵之后就可以套用CNN，后面的其实没有太多创新点。
![](/images/deeplearning/dnn_ctr_fig/ccpm.png)



####参考文献
1. Deep Learning over Multi-field Categorical Data – A Case Study on User Response Prediction
2. Product-based Neural Networks for User Response Prediction
3. Wide & Deep Learning for Recommender Systems 
4. Deep CTR Prediction in Display Advertising
5. A Convolutional Click Prediction Model
6. http://www.52cs.org/?p=1046
7. http://techshow.ctrip.com/archives/1149.html
8. http://tech.meituan.com/deep-understanding-of-ffm-principles-and-practices.html 









