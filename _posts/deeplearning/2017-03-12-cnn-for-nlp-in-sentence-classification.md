---
layout: post
title: CNN在NLP的应用--文本分类
categories:
- deeplearning
tags:
- CNN
- NLP
image:
    teaser: /deeplearning/cnn_nlp/sentiment.jpg
---

刚接触深度学习时知道CNN一般用于计算机视觉，RNN等一般用于自然语言相关。CNN目前在CV领域独领风骚，自然就有想法将CNN迁移到NLP中。但是NLP与CV不太一样，NLP有语言内存的结构，所以最开始CNN在NLP领域的应用在文本分类。相比于具体的句法分析、语义分析的应用，文本分类不需要精准分析。本文主要介绍最近学习到几个算法，并用mxnet进行了实现，如有错误请大家指出。

#### 1 Convolutional Neural Networks for Sentence Classification
##### 1.1 原理
![](/images/deeplearning/cnn_nlp/model1.png)
CNN的输入是矩阵形式，因此首先是构造矩阵。句子有词构成，DL中一般使用词向量，再对句子的长度设置一个固定值（可以是最大长度），那么就可以构造一个矩阵，这样就可以应用CNN了。这篇论文就是这样的思想，如上图所示。

###### 输入层
句子长度为$n$，词向量的维度为$k$，那么矩阵就是$n*k$。具体的，词向量可以是静态的或者动态的。静态指词向量提取利用word2vec等得到，动态指词向量是在模型整体训练过程中得到。

###### 卷积层
一个卷积层的kernel大小为$h*k$，$k$为输入层词向量的维度，那么$h$为窗口内词的数目。这样可以看做为N-Gram的变种。如此一个卷积操作可以得到一个$(n-h+1)*1$的feature map。多个卷积操作就可以得到多个这样的feature map

###### 池化层
这里面的池化层比较简单，就是一个Max-over-time Pooling，从前面的1维的feature map中取最大值。[这篇文章](http://blog.csdn.net/malefactor/article/details/51078135#0-tsina-1-38411-397232819ff9a47a7b7e80a40613cfe1)中给出了NLP中CNN的常用Pooling方法。最终将会得到1维的size=m的向量(m=卷积的数目)

###### 全连接+输出层
模型的输出层就是全连接+Softmax。可以加上通用的Dropout和正则的方法来优化

##### 1.2 实现
[这篇文章](
http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)用TensorFlow实现了这个模型，[代码](https://github.com/dennybritz/cnn-text-classification-tf)。我参考这个代码用mxnet实现了，[代码](https://github.com/yxzf/cnn-text-classification-mx)

#### 2 Character-level Convolutional Networks for Text Classification
##### 2.1 原理
图像的基本都要是像素单元，那么在语言中基本的单元应该是字符。CNN在图像中有效就是从原始特征不断向上提取高阶特征，那么在NLP中可以从字符构造矩阵，再应用CNN来做。这篇论文就是这个思路。
![](/images/deeplearning/cnn_nlp/model2.png)
###### 输入层
一个句子构造一个矩阵，句子由字符构成。设定一个句子的字符数目$n$(论文中$n=1014$)。每个字符是一个字符向量，这个字符向量可以是one-hot向量，也可以是一个char embeding. 设字符向量为$k$，那么矩阵为$n*k$。
###### 卷积层
这里的卷积层就是套用在图像领域的， 论文中给出具体的设置
![](/images/deeplearning/cnn_nlp/model2_conv.png)
###### 全连接+输出层
全连接层的具体设置如下
![/images/deeplearning/cnn_nlp/model2_fc.png]

##### 2.2 实现
[代码](https://github.com/scharmchi/char-level-cnn-tf)给出了TensorFlow的实现，我用mxnet实现的[代码](https://github.com/yxzf/char-level-cnn-mx)

#### 3 Character-Aware Neural Language Models
##### 3.1 原理
这篇文章提出一种CNN+RNN结合的模型。CNN部分，每个单词由character组成，如果对character构造embeding向量，则可以对单词构造矩阵作为CNN的输入。CNN的输出为词向量，作为RNN的输入，RNN的输出则是以整个词为单位。
![](/images/deeplearning/cnn_nlp/char-cnn-rnn.png)
###### 3.1.1 CNN层
**输入层**: 一个句子(sentence)是一个输入样本，句子由词(word)构成，词由字符(character)组成。每个字符学习一个embeding字符向量，设字符向量长度为$k$，那么一个word(长度为$w$)就可以构造成一个矩阵C($k*w$)

**卷积层**:
对这个矩阵C使用多个卷积层，每个卷积层的kernel大小为($k*n$)。卷积层可以看作是character的n-gram，那么每个卷积操作后得到的矩阵为($1*(w-n+1)$)

**池化层**:
池化层仍是max-pooling，挑选出($w-n+1$)长度向量中的最大值，将所有池化层的结果拼接就可以得到定长的向量$p$，$p$的长度为所有卷积层的数目

**Highway层**:
[Highway层](https://arxiv.org/pdf/1505.00387.pdf)是最近刚提出的一种结构，借鉴LSTM的gate概念。$x$为该层的输入，那么首先计算一个非线性转换$T(x)$，$T(x)\in [0, 1]$($T$一般为sigmod)。除了$T(x)$，还有另外一个非线性转换$H(x)$，最终的输出为$y = T(x) * H(x) + (1 - T(x)) * x$。从公式来看，$T(x)$充当了gate，如果$T(x)=1$，那么输出同传统的非线性层一样，输出是$H(x)$，如果$T(x)=0$，则直接输出输入$x$。作者认为Highway层的引入避免了梯度的快速消失，这种特性可以构建更深的网络。

**输出层**:
每个单词将得到一个词向量，与一般的词向量获取不同，这里的词向量是DNN在character embeding上得到的。

###### 3.1.2 RNN层
**输入层**:
将一个句子的每个CNN输出作为词向量，整个句子就是RNN的输入层

**隐藏层**:
一般使用LSTM，也可以是GRU

**输出层**:
输出层以word为单位，而不是character. 

##### 3.2实现
[代码](https://github.com/carpedm20/lstm-char-cnn-tensorflow)给出了TensorFlow的实现，我的MXNet实现见[代码](https://github.com/yxzf/lstm-char-cnn-mx)



#### 参考资料
1. https://github.com/Lasagne/Lasagne/blob/highway_example/examples/Highway%20Networks.ipynb
2. http://www.jeyzhang.com/cnn-apply-on-modelling-sentence.html
3. http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
4. http://www.wtoutiao.com/p/H08qKy.html
5. http://karpathy.github.io/2015/05/21/rnn-effectiveness/
6. https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf
