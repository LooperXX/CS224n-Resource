# CS224n-2019

本`Repo`主要包括课程的作业与文档(Lecture, Note, Additional Readings, Suggested Readings)

课程笔记参见我的[博客](https://looperxx.github.io/CS224n-2019-01-Introduction%20and%20Word%20Vectors/)，并在博客的`Repo`中提供笔记源文件的下载

## Assignment

本文档将简要记录作业中的要点

### Assignment 01

-   逐步完成共现矩阵的搭建，并调用 `sklearn.decomposition` 中的 `TruncatedSVD` 完成传统的基于SVD的降维算法
-   可视化展示，观察并分析其在二维空间下的聚集情况。
-   载入Word2Vec，将其与SVD得到的单词分布情况进行对比，分析两者词向量的不同之处。
-   学习使用`gensim`，使用`Cosine Similarity` 分析单词的相似度，对比单词和其同义词与反义词的`Cosine Distance` ，并尝试找到正确的与错误的类比样例
-   探寻Word2Vec向量中存在的 `Independent Bias` 问题

## Reference

-   [从SVD到PCA——奇妙的数学游戏](<https://my.oschina.net/findbill/blog/535044>)