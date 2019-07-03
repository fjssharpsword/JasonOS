1.ui.rating是模拟护士对交班项的行为，记录三个字段：userid,itemid,rating.
userid是护士id号，从0开始顺序编号，int类型
itemid是交班项id号，从0开始顺序编号，int类型
rating是护士对交班项的行为记录，int或float类型

2.SVDTrain.py是模型训练代码。
输入：数据集ui.rating路径；训练好的模型保存路径；
输出：已训练好的模型
用法：python SVDTrain.py --dataPath /data/fjsdata/nursereport/ui.rating --modelPath /data/fjsdata/nursereport/svd.model

3.SVDRecommend.py是向指定用户推荐k个交班项
输入：模型存放路径，待预测护士id存放路径(预测结果也写回同一个文件，uid.list格式是一行一个护士id)，推荐数目k个
输出：返回护士id对应的k个推荐交班项，格式如：uid [itemid1,itemid2,...,itemidk]
用法： python SVDRecommend.py --TopK 5 --modelPath /data/fjsdata/nursereport/svd.model --uidPath /data/fjsdata/nursereport/uid.list

4.运行环境：python3.x+numpy+pandas+surprise+sklearn, 用pip都可以安装。