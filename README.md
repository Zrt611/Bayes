# Bayes
A simple project by using graph (gcn, gat) to solve IPAL problem

# Data
数据来源于uci数据库上的glass和segementation数据，进行sample操作获取候选标签集进行后续训练，处理代码为Data_processing.py，处理后的数据为csv格式的，
随原数据一起放在了data.zip里面，可以自己下载使用

# Baseline
baseline是来源于IJCAI2015的一篇文章Solving the Partial Label Learning Problem: An Instance-based Approach
Min-Ling Zhang Fei Yu

# Model
用了GCN和GAT两种图模型，邻接矩阵的选取：知道grandtruth label的样本之间相同label认为有边，
不知道grandtruth label的样本之间看候选标签集中具有相同标签的数量超过总标签类别的一半，则认为样本之间有边
