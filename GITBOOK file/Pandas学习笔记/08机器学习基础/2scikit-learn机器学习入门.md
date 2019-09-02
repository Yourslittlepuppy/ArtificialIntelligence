
scikit-learn 机器学习库入门
===

---


案例：写一个能自动分辨苹果和橘子的程序
---

用分支结构可以吗？

```python
if color == 'yellow`:
    print('橘子')
else:
    print('苹果')
    
# 黄香蕉苹果怎么办？绿橘子怎么办？

if color !='yellow' & weight >= 150:
    print('苹果')
else:
    print('橘子')
    
# 大绿橘子呢？
```

    人工构造if else规则特别繁琐
    不管写多少if else规则，总能找到特定东西无法识别，要达到可用的正确率，就算最简单的分辨苹果橘子也得写成千上万条规则
    如果写完分辨橘子苹果的程序，需求变了，要分辨白菜和花菜，又得重写程序，无法复用
    
我们需要一种通用算法，分辨事物时：自动生成所有规则

---

机器学习的本质是模式识别（根据已有数据算出数据间的规则（相关性），用规则预测未知数据）

---

监督学习
---

    根据 特征 预测 标签

* 全数据
* 样本数据（每行一条数据，每列一个特征）
* 特征
* 标签
* 训练集
* 测试集

---

    抽象：特征工程
    算法：创建分类器
    训练：用已知标签数据训练分类器
    预测：预测未知标签数据，并评价分类器准确与否

    1.抽象:特征工程
    特征：能将本事物区别于其他事物的可观测量化指标
    特征工程：将搜集到的数据转为算法可用的数学形式
    搜集训练数据：去市场测量苹果和橘子的数据并记录在表格中

已知标签：训练数据

类别 | 重量 | 光洁度
--- | ---- | -----
苹果 | 150g | 光滑
苹果 | 170g | 光滑
橘子 | 130g | 粗糙
橘子 | 140g | 粗糙

未知标签：预测数据

类别 | 重量 | 光洁度
--- | ---- | -----
??? | 180g | 光滑
??? | 100g | 粗糙

目标是什么？
---

* 根据已有特征和标签数据 训练模型
* 用训练好的模型预测：只有特征没有标签的数据的**标签**

## 1:抽象：将现实数据映射到Python中

特征工程，数值化，类似数据分析的数据预处理阶段

机器学习算法传入数据必须是数值，不能是其他类型数据

字符串可以用独热编码转为矢量数值 [0,1,0,0,0]

图像可以将像素转为数值类型：

    [
        [[255,0,0],[255,0,0],[0,0,1]],
        [[255,1,255],[11,22,33],[33,44,55]],
        .....
    ]

降维后：

     [[255,0,0],[255,0,0],[0,0,1],[255,1,255],[11,22,33],[33,44,55]]


```python
feature = [[150, 0], [170, 0], [130, 1], [140, 1]]  # 训练集特征,2维
labels = [0, 0, 1, 1]  # 训练集标签，1维
feature_test = [[180, 0], [100, 1]]  # 测试集特征
# labels_test = [????]  # 求解，测试集标签
```

##  2:算法：创建分类器


```python
from sklearn import tree
```


```python
clf = tree.DecisionTreeClassifier()  # 创建空分类器
clf
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')



## 3：训练(学习)：用样本数据训练分类器

fit,训练：给空算法灌入训练特征和标签，训练算法，让模型掌握分类规则


```python
clf.fit(feature, labels)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')



## 4.预测：评价训练好的分类器，用它分辨一些水果

根据数据的特征，预测它的标签


```python
# 待预测数据
feature_test
```




    [[180, 0], [100, 1]]




```python
clf.predict(feature_test)
```




    array([0, 1])




```python

```
