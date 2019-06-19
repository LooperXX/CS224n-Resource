# CS224n-2019

本`Repo`主要包括课程的作业与文档(Lecture, Note, Additional Readings, Suggested Readings)

课程笔记参见我的[博客](https://looperxx.github.io/CS224n-2019-01-Introduction%20and%20Word%20Vectors/)，并在博客的[Repo](<https://github.com/LooperXX/LooperXX.github.io>)中提供笔记源文件的下载

本笔记不仅使用了Markdown的通用语法，还使用了 mkdocs-material 的一些语法以提升视觉效果，推荐通过我的[博客](<https://looperxx.github.io/CS224n-2019-Assignment/>)进行访问。

## Assignment 01

-   逐步完成共现矩阵的搭建，并调用 `sklearn.decomposition` 中的 `TruncatedSVD` 完成传统的基于SVD的降维算法
-   可视化展示，观察并分析其在二维空间下的聚集情况。
-   载入Word2Vec，将其与SVD得到的单词分布情况进行对比，分析两者词向量的不同之处。
-   学习使用`gensim`，使用`Cosine Similarity` 分析单词的相似度，对比单词和其同义词与反义词的`Cosine Distance` ，并尝试找到正确的与错误的类比样例
-   探寻Word2Vec向量中存在的 `Independent Bias` 问题

## Assignment 02

### 1  Written: Understanding word2vec

$$
P(O=o | C=c)=\frac{\exp \left(\boldsymbol{u}_{o}^{\top} \boldsymbol{v}_{c}\right)}{\sum_{w \in \mathrm{Vocab}} \exp \left(\boldsymbol{u}_{w}^{\top} \boldsymbol{v}_{c}\right)}
$$

$$
J_{\text { naive-softmax }}\left(v_{c}, o, U\right)=-\log P(O=o | C=c)
$$

真实(离散)概率分布 $p$ 与另一个分布 $q$ 的交叉熵损失为 $-\sum_i p_{i} \log \left(q_{i}\right)$

!!! question "Question a"

```
Show that the naive-softmax loss given in Equation (2) is the same as the cross-entropy loss between $y$ and $\hat y$; i.e., show that

$$
-\sum_{w \in Vocab} y_{w} \log \left(\hat{y}_{w}\right)=-\log \left(\hat{y}_{o}\right)
$$

Your answer should be one line.
```

**Answer a** : 

因为 $\textbf{y}$ 是独热向量，所以 $-\sum_{w \in Vocab} y_{w} \log (\hat{y}_{w})=-y_o\log (\hat{y}_{o}) -\sum_{w \in Vocab,w \neq o} y_{w} \log (\hat{y}_{w}) = -\log (\hat{y}_{o})$ 

!!! question "Question b"

```
Compute the partial derivative of $J_{\text{naive-softmax}}(v_c, o, \textbf{U})$ with respect to $v_c$. Please write your answer in terms of $\textbf{y}, \hat {\textbf{y}}, \textbf{U}$.
```

**Answer b** : 
$$
\begin{array}{l}
{\frac{\partial J\left(v_{c}, o, U\right)}{\partial v_{c}}} &={-\frac{\partial\left(u_{o}^{T} v_{c}\right)}{\partial v_{c}}+\frac{\partial \left(\log \left(\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)\right)\right)}{\partial v_{c}}} 
\\ &={-u_{o}+\frac{1}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)}} \frac{\partial \left(\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)\right)}{\partial v_{c}}
\\ &={-u_{o}+\sum_{w} \frac{\exp \left(u_{w}^{T} v_{c}\right)u_w}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)}} 
\\ &={-u_{o}+\sum_{w} p(O=w | C=c)u_{w}}
\\ &={-u_{o}+\sum_{w} \hat{y_w} u_{w}}
\\ &={U(\hat{y}-y)}
\end{array}
$$
!!! question "Question c"

```
Compute the partial derivatives of $J_{\text{naive-softmax}}(v_c, o, \textbf{U})$ with respect to each of the ‘outside' word vectors, $u_w$'s. There will be two cases: when $w = o$, the true ‘outside' word vector, and $w \neq o$, for all other words. Please write you answer in terms of $\textbf{y}, \hat {\textbf{y}}, \textbf{U}$.
```

**Answer c** : 
$$
\begin{array}{l}
{\frac{\partial J\left(v_{c}, o, U\right)}{\partial u_{w}}}  
&={-\frac{\partial\left(u_{o}^{T} v_{c}\right)}{\partial u_{w}}+\frac{\partial \left(\log \left(\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)\right)\right)}{\partial u_{w}}} 
\end{array}
$$
When $w \neq o$ :
$$
\begin{array}{l}
{\frac{\partial J\left(v_{c}, o, U\right)}{\partial u_{w}}}  
&= 0 + p(O=w | C=c) v_{c}
\\ &=\hat{y}_{w} v_{c}
\end{array}
$$
When $w = o$ :
$$
\begin{array}{l}
{\frac{\partial J\left(v_{c}, o, U\right)}{\partial u_{w}}}  
&= -v_c + p(O=o | C=c) v_{c}
\\ &=\hat{y}_{w} v_{c} - v_c
\\ &=(\hat{y}_{w} - 1)v_c
\end{array}
$$
Then : 
$$
{\frac{\partial J\left(v_{c}, o, U\right)}{\partial U}} = v_c(\hat y - y)^T
$$
!!! question "Question d"

```
The sigmoid function is given by the follow Equation :

$$
\sigma(x)=\frac{1}{1+e^{-x}}=\frac{e^{x}}{e^{x}+1}
$$

Please compute the derivative of $\sigma (x)$ with respect to $x$, where $x$ is a vector.
```

**Answer d** : 
$$
\begin{array}{l}
\frac{\partial \sigma(x_i)}{\partial x_i}
&=\frac{1}{(1+\exp (-x_i))^{2}} \exp (-x_i)=\sigma(x_i)(1-\sigma(x_i)) \\
\frac{\partial \sigma(x)}{\partial x}
&= \left[\frac{\partial \sigma\left(x_{j}\right)}{\partial x_{i}}\right]_{d \times d}
\\ &=\left[\begin{array}{cccc}{\sigma^{\prime}\left(x_{1}\right)} & {0} & {\cdots} & {0} \\ {0} & {\sigma^{\prime}\left(x_{2}\right)} & {\cdots} & {0} \\ {\vdots} & {\vdots} & {\vdots} & {\vdots} \\ {0} & {0} & {\cdots} & {\sigma^{\prime}\left(x_{d}\right)}\end{array}\right]
\\ &=\text{diag}(\sigma^\prime(x))
\end{array}
$$
!!! question "Question e"

```
Now we shall consider the Negative Sampling loss, which is an alternative to the Naive
Softmax loss. Assume that $K$ negative samples (words) are drawn from the vocabulary. For simplicity
of notation we shall refer to them as $w_{1}, w_{2}, \dots, w_{K}$ and their outside vectors as $u_{1}, \dots, u_{K}$. Note that
$o \notin\left\{w_{1}, \dots, w_{K}\right\}$. For a center word $c$ and an outside word $o$, the negative sampling loss function is
given by:

$$
J_{\text { neg-sample }}\left(v_{c}, o, U\right)=-\log \left(\sigma\left(u_{o}^{\top} v_{c}\right)\right)-\sum_{k=1}^{K} \log \left(\sigma\left(-u_{k}^{\top} v_{c}\right)\right)
$$

for a sample $w_{1}, w_{2}, \dots, w_{K}$, where $\sigma(\cdot)$ is the sigmoid function.

Please repeat parts b and c, computing the partial derivatives of $J_{\text { neg-sample }}$ respect to $v_c$, with
respect to $u_o$, and with respect to a negative sample $u_k$. Please write your answers in terms of the
vectors $u_o, v_c,$ and $u_k$, where $k \in[1, K]$. After you've done this, describe with one sentence why this
loss function is much more efficient to compute than the naive-softmax loss. Note, you should be able
to use your solution to part (d) to help compute the necessary gradients here.
```

**Answer e** : 

For $v_c$ :
$$
\begin{array}{l}
\frac{\partial J_{\text {neg-sample}}}{\partial v_c}
&=(\sigma(u_o^T v_c) - 1) u_o	+ \sum_{k=1}^{K}\left(1-\sigma\left(-u_{k}^{T} v_{c}\right)\right) u_{k} 
\\ &= (\sigma(u_o^T v_c) - 1) u_o+ \sum_{k=1}^{K}\sigma\left(u_{k}^{T} v_{c}\right) u_{k}
\end{array}
$$
For $u_o$, Remeber : $o \notin\left\{w_{1}, \dots, w_{K}\right\}$ 😢 :
$$
\frac{\partial J_{\text {neg-sample}}}{\partial u_o}=(\sigma(u_o^T v_c) - 1)v_c
$$
For $u_k$ :
$$
\frac{\partial J}{\partial \boldsymbol{u}_{k}}=-\left(\sigma\left(-\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right)-1\right) \boldsymbol{v}_{c} = \sigma\left(\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right)\boldsymbol{v}_{c}, \quad for\ k=1,2, \ldots, K
$$
Why this
loss function is much more efficient to compute than the naive-softmax loss?

For naive softmax loss function:
$$
\begin{array}{l}
{\frac{\partial J\left(v_{c}, o, U\right)}{\partial v_{c}}}  
&={U(\hat{y}-y)}
\\ {\frac{\partial J\left(v_{c}, o, U\right)}{\partial U}} 
&= v_c(\hat y - y)^T
\end{array}
$$
For negative sampling loss function:
$$
\begin{aligned} \frac{\partial J}{\partial \boldsymbol{v}_{c}} &=\left(\sigma\left(\boldsymbol{u}_{o}^{\top} v_{c}\right)-1\right) \boldsymbol{u}_{o} + \sum_{k=1}^{K}\sigma\left(\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right) \boldsymbol{u}_{k} 
=\sigma\left(-\boldsymbol{u}_{o}^{\top} v_{c}\right) \boldsymbol{u}_{o} + \sum_{k=1}^{K}\sigma\left(\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right) \boldsymbol{u}_{k}
\\ \frac{\partial J}{\partial \boldsymbol{u}_{o}} &=\left(\sigma\left(\boldsymbol{u}_{o}^{\top} \boldsymbol{v}_{c}\right)-1\right) \boldsymbol{v}_{c} 
= \sigma\left(-\boldsymbol{u}_{o}^{\top} \boldsymbol{v}_{c}\right)\boldsymbol{v}_{c} 
\\ \frac{\partial J}{\partial \boldsymbol{u}_{k}} &=\sigma\left(\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right) \boldsymbol{v}_{c}, \quad \text { for all } k=1,2, \ldots, K \end{aligned}
$$
从求得的偏导数中我们可以看出，原始的softmax函数每次对 $v_c$ 进行反向传播时，需要与 output vector matrix 进行大量且复杂的矩阵运算，而负采样中的计算复杂度则不再与词表大小 $V$ 有关，而是与采样数量 $K$ 有关。

!!! question "Question f"

```
Suppose the center word is $c = w_t$ and the context window is $\left[w_{t-m}, \ldots, w_{t-1}, w_{t}, w_{t+1}, \dots,w_{t+m} \right]$, where $m$ is the context window size. Recall that for the skip-gram version of **word2vec**, the
total loss for the context window is

$$
J_{\text { skip-gram }}\left(v_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right)=\sum_{-m \leq j \leq m \atop j \neq 0} \boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right)
$$

Here, $J\left(v_{c}, w_{t+j}, \boldsymbol{U}\right)$ represents an arbitrary loss term for the center word $c = w_t$ and outside word
$w_t+j$ . $J\left(v_{c}, w_{t+j}, \boldsymbol{U}\right)$ could be $J_{\text {naive-softmax}}\left(v_{c}, w_{t+j}, \boldsymbol{U}\right)$ or $J_{\text {neg-sample}}\left(v_{c}, w_{t+j}, \boldsymbol{U}\right)$, depending on your
implementation.

Write down three partial derivatives:

$$
\partial \boldsymbol{J}_{\text { skip-gram }}\left(\boldsymbol{v}_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right) / \partial \boldsymbol{U} \\ \partial \boldsymbol{J}_{\text { skip-gram }}\left(\boldsymbol{v}_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right) / \partial v_{c}
\\ \partial \boldsymbol{J}_{\text { skip-gram }}\left(\boldsymbol{v}_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right) / \partial v_{w} \text { when } w \neq c
$$

Write your answers in terms of $\partial \boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right) / \partial \boldsymbol{U}$ and $\partial \boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right) / \partial \boldsymbol{v_c}$. This is very simple -
each solution should be one line.

***Once you're done***: Given that you computed the derivatives of $\partial \boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right)$ with respect to all the
model parameters $U$ and $V$ in parts a to c, you have now computed the derivatives of the full loss
function $J_{skip-gram}$ with respect to all parameters. You're ready to implement ***word2vec*** !
```

**Answer f** : 
$$
\begin{array}{l} 
\frac{\partial J_{s g}}{\partial U} &= \sum_{-m \leq j \leq m, j \neq 0} \frac{\partial J\left(v_{c}, w_{t+j}, U\right)}{\partial U} 
\\ \frac{\partial J_{s g}}{\partial v_{c}}&= \sum_{-m \leq j \leq m, j \neq 0} \frac{\partial J\left(v_{c}, w_{t+j}, U\right)}{\partial v_{c}} 
\\ \frac{\partial J_{s g}}{\partial v_{w}}&=0(\text { when } w \neq c) \end{array}
$$

### 2 Coding: Implementing word2vec

#### word2vec.py

本部分要求实现 $sigmoid, naiveSoftmaxLossAndGradient, negSamplingLossAndGradient, skipgram$  四个函数，主要考察对第一部分中反向传播计算结果的实现。代码实现中，通过优化偏导数结合偏导数计算结果与 $\sigma(x) + \sigma(-x) = 1$ 对公式进行转化，从而实现了全矢量化。这部分需要大家自行结合代码与公式进行推导。

#### sgd.py

实现 SGD 
$$
\theta^{n e w}=\theta^{o l d} - \alpha \nabla_{\theta} J(\theta)
$$

#### run.py

首先要说明的是，这个真的要跑好久 😅

!!! question "Question"

```
Briefly explain in at most three sentences what you see in the plot.
```

上图是经过训练的词向量的可视化。我们可以注意到一些模式：

-   近义词被组合在一起，比如 amazing 和 wonderful，woman 和 female。
    -   但是 man 和 male 却距离较远
-   反义词可能因为经常属于同一上下文，它们也会与同义词一起出现，比如 enjoyable 和 annoying。
-   `man:king::woman:queen` 以及 `queen:king::female:male` 形成的两条直线基本平行

## Assignment 03

### 1. Machine Learning & Neural Networks

#### (a) Adam Optimizer

回忆一下标准随机梯度下降的更新规则
$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}-\alpha \nabla_{\boldsymbol{\theta}}{J_{\mathrm{minibatch}}(\boldsymbol{\theta})}
$$
其中，$\boldsymbol{\theta}$ 是包含模型所有参数的向量，$J$ 是损失函数，$\nabla_{\boldsymbol{\theta}} J_{\mathrm{minibatch}}(\boldsymbol{\theta})$ 是关于minibatch数据上参数的损失函数的梯度，$\alpha$ 是学习率。[Adam Optimization](https://arxiv.org/pdf/1412.6980.pdf)使用了一个更复杂的更新规则，并附加了两个步骤。

!!! question "Question 1.a.i"

```
首先，Adam使用了一个叫做 $momentum$ **动量**的技巧来跟踪梯度的移动平均值 $m$

$$
\begin{aligned}
\mathbf{m} & \leftarrow \beta_{1} \mathbf{m}+\left(1-\beta_{1}\right) \nabla_{\boldsymbol{\theta}} J_{\text { minibatch }}(\boldsymbol{\theta}) \\ \boldsymbol{\theta} & \leftarrow \boldsymbol{\theta}-\alpha \mathbf{m} \end{aligned}
$$

其中，$\beta_1$ 是一个 0 和 1 之间的超参数(通常被设为0.9)。简要说明(不需要用数学方法证明，只需要直观地说明)如何使用m来阻止更新发生大的变化，以及总体上为什么这种小变化可能有助于学习。
```

**Answer 1.a.i** : 

-   由于超参数 $\beta _1$ 一般被设为0.9，此时对于移动平均的梯度值 $m$ 而言，主要受到的是之前梯度的移动平均值的影响，而本次计算得到的梯度将会被缩放为原来的 ${1 - \beta_1}$ 倍，即时本次计算得到的梯度很大（梯度爆炸），这一影响也会被减轻，从而阻止更新发生大的变化。
-   通过减小梯度的变化程度，使得每次的梯度更新更加稳定，从而使模型学习更加稳定，收敛速度更快，并且这也减慢了对于较大梯度值的参数的更新速度，保证其更新的稳定性。

!!! question "Question 1.a.ii"

```
Adam还通过跟踪梯度平方的移动平均值 $v$ 来使用自适应学习率

$$
\begin{aligned} 
\mathbf{m} & \leftarrow \beta_{1} \mathbf{m}+\left(1-\beta_{1}\right) \nabla_{\boldsymbol{\theta}} J_{\text { minibatch }}(\boldsymbol{\theta}) \\ 
\mathbf{v} & \leftarrow \beta_{2} \mathbf{v}+\left(1-\beta_{2}\right)\left(\nabla_{\boldsymbol{\theta}} J_{\text { minibatch }}(\boldsymbol{\theta}) \odot \nabla_{\boldsymbol{\theta}} J_{\text { minibatch }}(\boldsymbol{\theta})\right) \\ 
\boldsymbol{\theta} & \leftarrow \boldsymbol{\theta}-\alpha \odot \mathbf{m} / \sqrt{\mathbf{v}}  \end{aligned}
$$

其中，$\odot, /$ 分别表示逐元素的乘法和除法（所以 $z \odot z$ 是逐元素的平方），$\beta_2$ 是一个 0 和 1 之间的超参数(通常被设为0.99)。因为Adam将更新除以 $\sqrt v$ ，那么哪个模型参数会得到更大的更新？为什么这对学习有帮助？
```

**Answer 1.a.ii** : 

-   移动平均梯度最小的模型参数将得到较大的更新。
-   一方面，将梯度较小的参数的更新变大，帮助其走出局部最优点（鞍点）；另一方面，将梯度较大的参数的更新变小，使其更新更加稳定。结合以上两个方面，使学习更加快速的同时也更加稳定。

#### (b) Dropout

[Dropout](https://www.cs.toronto.edu/˜hinton/absps/JMLRdropout.pdf) 是一种正则化技术。在训练期间，Dropout 以 $p_{drop}$ 的概率随机设置隐藏层 $h$ 中的神经元为零(每个minibatch中 dropout 不同的神经元),然后将 $h$ 乘以一个常数 $\gamma$ 。我们可以写为
$$
\mathbf{h}_{\mathrm{drop}}=\gamma \mathbf{d} \circ \mathbf{h}
$$
其中，$d \in \{0,1\}^{D_h}$ ( $D_h$ 是 $h$ 的大小)是一个掩码向量，其中每个条目都是以 $p_{drop}$ 的概率为 0 ，以 $1 - p_{drop}$ 的概率为 1。$\gamma$ 是使得 $h_{drop}$ 的期望值为 $h$ 的值
$$
\mathbb{E}_{p_{\text{drop}}}\left[\mathbf{h}_{\text{drop}}\right]_{i}=h_{i}, \text{for all } i \in \{1,\dots,D_h\}
$$
!!! question "Question 1.b.i"

```
$\gamma$ 必须等于什么(用 $p_{drop}$ 表示) ？简单证明你的答案。
```

**Answer 1.b.i** : 
$$
\gamma = \frac{1}{1 - p_{drop}} \\
$$
证明如下：
$$
\sum_i (1 -  p_{drop}) h_i = (1 -  p_{drop}) E[h] \\
\sum_i[h_{drop}]_i = \gamma\sum_i (1 -  p_{drop}) h_i = \gamma (1 -  p_{drop}) E[h] = E[h]
$$
!!! question "Question 1.b.ii"

```
为什么我们应该只在训练时使用 dropout 而在评估时不使用？
```

**Answer 1.b.ii** : 

如果我们在评估期间应用 dropout ，那么评估结果将会具有随机性，并不能体现模型的真实性能，违背了正则化的初衷。通过在评估期间禁用 dropout，从而观察模型的性能与正则化的效果，保证模型的参数得到正确的更新。

### 2. Neural Transition-Based Dependency Parsing

在本节中，您将实现一个基于神经网络的依赖解析器，其目标是在UAS(未标记依存评分)指标上最大化性能。

依存解析器分析句子的语法结构，在 head words 和 修饰 head words 的单词之间建立关系。你的实现将是一个基于转换的解析器，它逐步构建一个解析。每一步都维护一个局部解析，表示如下

-   一个存储正在被处理的单词的 栈 
-   一个存储尚未处理的单词的 缓存
-   一个解析器预测的 依赖 的列表

最初,栈只包含 ROOT ，依赖项列表是空的，而缓存则包含了这个句子的所有单词。在每一个步骤中,解析器将对部分解析使用一个转换,直到它的魂村是空的，并且栈大小为1。可以使用以下转换：

-   SHIFT：将buffer中的第一个词移出并放到stack上。
-   LEFT-ARC：将第二个(最近添加的第二)项标记为栈顶元素的依赖，并从堆栈中删除第二项
-   RIGHT-ARC：将第一个(最近添加的第一)项标记为栈中第二项的依赖，并从堆栈中删除第一项

在每个步骤中，解析器将使用一个神经网络分类器在三个转换中决定。

!!! question "Question 2.a"

```
求解解析句子 “I parsed this sentence correctly” 所需的转换顺序。这句话的依赖树如下所示。在每一步中，给出 stack 和 buffer 的结构，以及本步骤应用了什么转换，并添加新的依赖(如果有的话)。下面提供了以下三个步骤。

![1560871900131](imgs/1560871900131.png)
```

**Answer 2.a** : 

| Stack                          | Buffer                                 | New dependency         | Transition           |
| ------------------------------ | -------------------------------------- | ---------------------- | -------------------- |
| [ROOT]                         | [I, parsed, this, sentence, correctly] |                        | Initial Conﬁguration |
| [ROOT, I]                      | [parsed, this, sentence, correctly]    |                        | SHIFT                |
| [ROOT, I, parsed]              | [this, sentence, correctly]            |                        | SHIFT                |
| [ROOT, parsed]                 | [this, sentence, correctly]            | parsed $\to$ I         | LEFT-ARC             |
| [ROOT, parsed, this]           | [sentence, correctly]                  |                        | SHIFT                |
| [ROOT, parsed, this, sentence] | [correctly]                            |                        | SHIFT                |
| [ROOT, parsed, sentence]       | [correctly]                            | sentence $\to$ this    | LEFT-ARC             |
| [ROOT, parsed]                 | [correctly]                            | parsed $\to$ sentence  | RIGHT-ARC            |
| [ROOT, parsed, correctly]      | []                                     |                        | SHIFT                |
| [ROOT, parsed]                 | []                                     | parsed $\to$ correctly | RIGHT-ARC            |
| [ROOT]                         | []                                     | ROOT $\to$ parsed      | RIGHT-ARC            |

!!! question "Question 2.b"

```
一个包含 $n$ 个单词的句子需要多少步(用 $n$ 表示)才能被解析？简要解释为什么。
```

**Answer 2.b** : 

包含$n$个单词的句子需要 $2 \times n$ 步才能完成解析。因为需要进行 $n$ 步的 $SHIFT$ 操作和 共计$n 步的 LEFT-ARC 或 RIGHT-ARC 操作，才能完成解析。（每个单词都需要一次SHIFT和ARC的操作，初始化步骤不计算在内）

**Question 2.c**

实现解析器将使用的转换机制

**Question 2.d**

我们的网络将预测哪些转换应该应用于部分解析。我们可以使用它来解析一个句子，通过应用预测出的转换操作，直到解析完成。然而，在对大量数据进行预测时，神经网络的运行速度要高得多(即同时预测了对任何不同部分解析的下一个转换)。我们可以用下面的算法来解析小批次的句子

![1560906831993](imgs/1560906831993.png)

实现minibatch的解析器

我们现在将训练一个神经网络来预测，考虑到栈、缓存和依赖项集合的状态，下一步应该应用哪个转换。首先，模型提取了一个表示当前状态的特征向量。我们将使用原神经依赖解析论文中的特征集合：[A Fast and Accurate Dependency Parser using Neural Networks](http://cs.stanford.edu/people/danqi/papers/emnlp2014.pdf)。这个特征向量由标记列表(例如在栈中的最后一个词，缓存中的第一个词，栈中第二到最后一个字的依赖(如果有))组成。它们可以被表示为整数的列表$[w_1,w_2,\dots,w_m]$，m是特征的数量，每个 $0 \leq w_i \lt |V|$ 是词汇表中的一个token的索引($| V |$是词汇量)。首先，我们的网络查找每个单词的嵌入，并将它们连接成一个输入向量：
$$
\mathbf{x}=\left[\mathbf{E}_{w_{1}}, \dots, \mathbf{E}_{w_{m}}\right] \in \mathbb{R}^{d m}
$$
其中 $\mathbf{E} \in \mathbb{R}^{|V| \times d}$ 是嵌入矩阵，每一行 $\mathbf{E}_w$ 是一个特定的单词 $w$ 的向量。接着我们可以计算我们的预测：
$$
\mathbf h = \text{ReLU}(\mathbf{xW+b_1}) \\
\mathbf l = \text{ReLU}(\mathbf{hU+b_2}) \\
\mathbf {\hat y} = \text{softmax}(l) 
$$
其中， $\mathbf{h}$ 指的是隐藏层，$\mathbf{l}$ 是其分数，$\mathbf{\hat y}$ 指的是预测结果， $\text{ReLU(z)}=max(z,0)$ 。我们使用最小化交叉熵损失来训练模型
$$
J(\theta) = CE(\mathbf y,\mathbf{\hat y}) = -\sum^3_{i=1}y_i\log\hat y_i
$$
训练集的损失为所有训练样本的 $J(\theta)$ 的平均值。

**Question 2.f**

我们想看看依赖关系解析的例子，并了解像我们这样的解析器在什么地方可能是错误的。例如，在这个句子中:

![1560950604163](imgs/1560950604163.png)

依赖 $\text{into Afghanistan}$ 是错的，因为这个短语应该修饰 $\text{sent}$ (例如 $\text{sent into Afghanistan}$) 而不是 $\text{troops}$ (因为 $\text{ troops into Afghanistan}$ 没有意义)。下面是正确的解析：

![1560950910787](imgs/1560950910787.png)

一般来说，以下是四种解析错误：

-   **Prepositional Phrase Attachment Error** 介词短语连接错误：在上面的例子中，词组 $\text{into Afghanistan}$ 是一个介词短语。介词短语连接错误是指介词短语连接到错误的 head word 上(在本例中，troops 是错误的 head word ，sent 是正确的 head word )。介词短语的更多例子包括with a rock, before midnight和under the carpet。
-   **Verb Phrase Attachment Error** 动词短语连接错误：在句子$\text{leave the store alone, I went out to watch the parade}$中，短语 $\text{leave the store alone}$ 是动词短语。动词短语连接错误是指一个动词短语连接到错误的 head word 上(在本例中，正确的头词是 $\text{went}$)。
-   **Modiﬁer Attachment Error** 修饰语连接错误：在句子 $\text{I am extremely short}$ 中，副词extremely 是形容词 short 的修饰语。修饰语附加错误是修饰语附加到错误的 head word 上时发生的错误(在本例中，正确的头词是 short)。
-   **Coordination Attachment Error** 协调连接错误：在句子 $\text{Would you like brown rice or garlic naan?}$ 中， brown rice 和garlic naan都是连词，or是并列连词。第二个连接词(这里是garlic naan)应该连接到第一个连接词(这里是brown rice)。协调连接错误是当第二个连接词附加到错误的 head word 上时(在本例中，正确的头词是rice)。其他并列连词包括and, but和so。

在这个问题中有四个句子，其中包含从解析器获得的依赖项解析。每个句子都有一个错误，上面四种类型都有一个例子。对于每个句子，请说明错误的类型、不正确的依赖项和正确的依赖项。为了演示:对于上面的例子，您可以这样写：

-   Error type: Prepositional Phrase Attachment Error 
-   Incorrect dependency: troops $\to$ Afghanistan
-   Correct dependency: sent $\to$ Afghanistan 

注意：依赖项注释有很多细节和约定。如果你想了解更多关于他们的信息，你可以浏览UD网站:http://universaldependencies.org。然而，你不需要知道所有这些细节就能回答这个问题。在每一种情况下，我们都在询问短语的连接，应该足以看出它们是否修饰了正确的head。特别是，你不需要查看依赖项边缘上的标签——只需查看边缘本身就足够了。

**Answer 2.f**

![1560951554929](imgs/1560951554929.png)

-   Error type: Verb Phrase Attachment Error
-   Incorrect dependency: wedding $\to$ fearing
-   Correct dependency: heading $\to$ fearing

![1560951560930](imgs/1560951560930.png)

-   Error type: Coordination Attachment Error
-   Incorrect dependency: makes $\to$ rescue
-   Correct dependency: rush $\to$ rescue

![1560951569847](imgs/1560951569847.png)

-   Error type: Prepositional Phrase Attachment Error
-   Incorrect dependency: named $\to$ Midland
-   Correct dependency: guy $\to$ Midland

![1560951576042](imgs/1560951576042.png)

-   Error type: Modiﬁer Attachment Error 
-   Incorrect dependency: elements $\to$ most
-   Correct dependency: crucial $\to$ most

## Reference

-   [从SVD到PCA——奇妙的数学游戏](

-   <https://my.oschina.net/findbill/blog/535044>)