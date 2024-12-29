## 基于预训练隐马尔可夫模型的汉语拼音序列转汉字语句序列程序

**Chinese Pinyin (Hanyu pinyin) to Chinese Character (Hanzi) Conversion Program Based on Pretrained Hidden Markov Model (HMM)**

### 项目原理

本项目基于 **隐马尔可夫模型 (HMM)** 实现汉语拼音序列到汉字序列的转换。HMM是一种概率模型，假设观察序列（拼音）由隐藏状态序列（汉字）生成，并通过状态转移和发射概率描述序列关系。模型训练时，程序首先加载训练数据，提取拼音和汉字构建词汇表，并统计初始状态、状态转移和发射概率矩阵。训练过程中，HMM使用**最大似然估计**优化这些概率，以捕捉拼音与汉字的映射关系。解码阶段，利用**维特比算法 (Viterbi Algorithm)** 寻找最可能的汉字序列作为输出结果。本项目适合处理语言序列建模和序列标注等的任务。

### 数据集准备

- 需要`CSV`格式文件，其应包含两列，要求第一列为由汉语拼音构成的句子，第二列为由汉字构成的句子。
- 数据集示例：

| 第一列 (汉字语句) | 第二列 (拼音语句) |
| ----- | ----- |
| 我们试试看！ | wo3 men shi4 shi4 kan4 ！ |
| 我该去睡觉了。 | wo3 gai1 qu4 shui4 jiao4 le 。 |
| 你在干什么啊？ | ni3 zai4 gan4 shen2 me a ？ |
| 这是什么啊？ | zhe4 shi4 shen2 me a ？ |
| 我会尽量不打扰你复习。 | wo3 hui4 jin3 liang4 bu4 da3 rao3 ni3 fu4 xi2 。 |

### 运行环境

基于`Python 3.x`，版本不限，需要`numpy`、`pandas`、`hmmlearn`库，参考以下命令安装：
```
python3 -m pip install numpy pandas hmmlearn
```
硬件环境：假设训练语料数据包含 $2.65$ 万个不同的汉字，则推理至少需要 $6GB$ 左右的内存，训练至少需要 $8GB$ 左右的内存。

### 训练和推理方法

修改`py2hz.py`的主函数代码以运行。我们已开源了基于多领域文本的预训练的模型权重`hmm_model.pkl.bz2`和`hmm_model_large.pkl.bz2`，可以直接使用。`hmm_model.pkl.bz2`规模稍小，可满足日常汉语的转换需求，其解压缩后约为 $800MB$ 左右；`hmm_model_large.pkl.bz2`覆盖了几乎所有汉字的读音，并在规模更大的语料库上进行训练，其解压缩后约为 $4.5GB$ 左右。

若需自行训练则取消train函数的注释，并修改函数参数。训练完成后模型将会被压缩保存，原因是模型中存在非常稀疏的大矩阵，适合压缩存储。

```python
# dataset_path：数据集路径
# model_path：模型保存路径
# pinyin_str：待解析的拼音语句。
if __name__ == '__main__':
    train(dataset_path='train.csv', model_path='hmm_model.pkl.bz2')
    pred(model_path='hmm_model.pkl.bz2', pinyin_str='hong2 yan2 bo2 ming4')
```

### 预训练模型效果

下表展示了预训练模型`hmm_model_large.pkl.bz2`的使用效果。

| 输入 | 输出 |
| ----- | ----- |
| hong2 yan2 bo2 ming4 | 红颜薄命 |
| guo2 jia1 chao1 suan4 ji3 nan2 zhong1 xin1 | 国家超算济南中心 |
| liu3 an4 hua1 ming2 you4 yi1 cun1 | 柳暗花明又一村 |
| gu3 zhi4 shu1 song1 zheng4 | 骨质疏松症 |
| xi1 an1 dian4 zi3 ke1 ji4 da4 xue2 | 西安电子科技大学 |
| ye4 mian4 zhi4 huan4 suan4 fa3 | 页面置换算法 |

### 作者声明及访客统计

Author: Du Yu (202103180009@stu.qlu.edu.cn), 
Faculty of Computer Science and Technology, Qilu University of Technology (Shandong Academy of Sciences).

<div><b>Number of Total Visits (All of Duyu09's GitHub Projects): </b><br><img src="https://profile-counter.glitch.me/duyu09/count.svg" /></div> 

<div><b>Number of Total Visits (py2hz): </b>
<br><img src="https://profile-counter.glitch.me/py2hz/count.svg" /></div> 
