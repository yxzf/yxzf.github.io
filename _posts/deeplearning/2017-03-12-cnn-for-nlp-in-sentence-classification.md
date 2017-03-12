---
layout: post
title: CNN在NLP的应用--文本分类
categories:
- deeplearning
tags:
- CNN
- NLP
image:
    teaser: /development/git_branch.png
---

刚接触深度学习时知道CNN一般用于计算机视觉，RNN等一般用于自然语言相关。CNN目前在CV领域独领风骚，自然就有想法将CNN迁移到NLP中。但是NLP与CV不太一样，NLP有语言内存的结构，所以最开始CNN在NLP领域的应用在文本分类。相比于具体的句法分析、语义分析的应用，文本分类不需要精准分析。本文主要介绍最近学习到几个算法，并用mxnet进行了实现，如有错误请大家指出。

#### 1 Convolutional Neural Networks for Sentence Classification
###### 1.1 原理
![](/images/deeplearning/model1.png)
CNN的输入是矩阵形式，因此首先是构造矩阵。句子有词构成，DL中一般使用词向量，再对句子的长度设置一个固定值（可以是最大长度），那么就可以构造一个矩阵，这样就可以应用CNN了。这篇论文就是这样的思想，如上图所示。

###### 输入层
句子长度为$n$，词向量的维度为$k$，那么矩阵就是n*k。具体的，词向量可以是静态的或者动态的。静态指词向量提取利用word2vec等得到，动态指词向量是在模型整体训练过程中得到。

###### 卷积层
一个卷积层的kernel大小为h*k，k为输入层词向量的维度，那么h为窗口内词的数目。这样可以看做为N-Gram的变种。如此一个卷积操作可以得到一个(n-h+1)*1的feature map。多个卷积操作就可以得到多个这样的feature map

###### 池化层
这里面的池化层比较简单，就是一个Max-over-time Pooling，从前面的1维的feature map中取最大值。[这篇文章](http://blog.csdn.net/malefactor/article/details/51078135#0-tsina-1-38411-397232819ff9a47a7b7e80a40613cfe1)中给出了NLP中CNN的常用Pooling方法。最终将会得到1维的size=m的向量(m=卷积的数目)

###### 全连接+输出层
模型的输出层就是全连接+Softmax。可以加上通用的Dropout和正则的方法来优化

###### 1.2 实现
[这篇文章](
http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)用TensorFlow实现了这个模型，[代码](https://github.com/dennybritz/cnn-text-classification-tf)。我参考这个代码用mxnet实现了，[源码](https://github.com/yxzf/cnn-text-classification-mx)

#### 2 Character-level Convolutional Networks for Text Classification
#####2.1 原理
图像的基本都要是像素单元，那么在语言中基本的单元应该是字符。CNN在图像中有效就是从原始特征不断向上提取高阶特征，那么在NLP中可以从字符构造矩阵，再应用CNN来做。这篇论文就是这个思路。
![](/images/model2.png)
###### 输入层
一个句子构造一个矩阵，句子由字符构成。设定一个句子的字符数目n(论文中n=1014)。每个字符是一个字符向量，这个字符向量可以是one-hot向量，也可以是一个char embeding. 设字符向量为k，那么矩阵为n*k。
###### 卷积层
这里的卷积层就是套用在图像领域的， 论文中给出具体的设置
![](/images/model2_conv.png)
###### 全连接+输出层
全连接层的具体设置如下
![/images/model2_fc.png]

#####2.2 实现
[代码](https://github.com/scharmchi/char-level-cnn-tf)给出了TensorFlow的实现，我用mxnet实现的[源码]

