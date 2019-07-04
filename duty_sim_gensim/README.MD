# Sentences Similarity#

1.dutyModel.py
输入：文件，一条句子一行，放在data文件夹下
输出：保存模型
执行：python dutyModel.py --fileName med.csv

2.dutySim.py
输入：文件，一条句子，放在data文件夹下
输出：相似度最高的k条,在data文件夹下sim.txt
执行：python dutySim.py --topK 10 --senName sen.txt

3.环境：windows+python3.6+jieba+gensim
      pip install jieba
      pip install gensim