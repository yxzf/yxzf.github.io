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
![](/images/deeplearning/model1.png)
CNN的输入是矩阵形式，因此首先是构造矩阵。句子有词构成，DL中一般使用词向量，再对句子的长度设置一个固定值（可以是最大长度），那么就可以构造一个矩阵，这样就可以应用CNN了。这篇论文就是这样的思想，如上图所示。

##### 输入层
$n_k$句子$$ n_k$$长度为$$ n$$，词向量的维度为$$k$$，那么矩阵就是n*k。具体的，词向量可以是静态的或者动态的。静态指词向量提取利用word2vec等得到，动态指词向量是在模型整体训练过程中得到。

