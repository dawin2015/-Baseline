# /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import zipfile
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# sys.path.append('C:\\Users\\DAWIN\\Documents\\Tencent Files\\694596886\\FileRecv\\XGBoost包及教程\\xgboost-master\\wrapper')

import xgboost as xgb
from sklearn.cross_validation import train_test_split

import time


start_time = time.time()

# load data
data_root = "pre"

dfTrain = pd.read_csv("%s/train.csv" % data_root)
dfTest = pd.read_csv("%s/test.csv" % data_root)
dfAd = pd.read_csv("%s/ad.csv" % data_root)
dfUser = pd.read_csv("%s/user.csv" % data_root)
dfApp_categories = pd.read_csv("%s/app_categories.csv" % data_root)
# dfPosition = pd.read_csv("%s/position.csv" % data_root)
# dfUser_app_actions = pd.read_csv("%s/user_app_actions.csv" % data_root)
# dfUser_installedapps = pd.read_csv("%s/user_installedapps.csv" % data_root)
"""
# take a look the header to compare
print "Train header: ", dfTrain.head()
print "Test header: ", dfTest.head()
print "Ad header: ", dfAd.head()
print "User header: ", dfUser.head()
print "App_categories header: ", dfApp_categories.head()
print "Postion header: ", dfPosition.head()
print "User_app_actions header: ", dfUser_app_actions.head()
print "User_installedapps header: ", dfUser_installedapps.head()
"""
# process data: join into a big table
dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
dfTrain = pd.merge(dfTrain, dfUser, on="userID")
dfTrain = pd.merge(dfTrain, dfApp_categories, on="appID")
# dfTrain = pd.merge(dfTrain, dfPosition, on="positionID")
# dfTrain = pd.merge(dfTrain, dfUser_app_actions, on="userID")
# dfTrain = pd.merge(dfTrain, dfUser_installedapps, on="userID")

dfTest = pd.merge(dfTest, dfAd, on="creativeID")
dfTest = pd.merge(dfTest, dfUser, on="userID")
dfTest = pd.merge(dfTest, dfApp_categories, on="appID")
# dfTest = pd.merge(dfTest, dfPosition, on="positionID")
# dfTest = pd.merge(dfTest, dfUser_app_actions, on="userID")
# dfTest = pd.merge(dfTest, dfUser_installedapps, on="userID")

trains = dfTrain # .iloc[:, 0:15]
tests = dfTest # .iloc[:, 0:15]
print '-----------------------------------------------------'
# test_userID = dfTest.userID
# test_creativeID = dfTest.creativeID

params = {
'booster':'gbtree', # 选择每次迭代的模型，有两种选择：1.gbtree：基于树的模型 2.gbliner：线性模型
'objective': 'binary:logistic',
'scale_pos_weight': 1/9.5,
# 27324条正样本
# 218428条负样本
# 差不多1:9/10这样子
'gamma':0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':6, # 构建树的深度，越大越容易过拟合
'lambda':3,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
#'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':3,
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.01, # 如同学习率
'seed':1000,
'nthread':16,# cpu 线程数
'eval_metric': 'rmse'
}

plst = list(params.items())
num_rounds = 1000  # 迭代次数

train_xy, val = train_test_split(trains, test_size=0.15, random_state=1)
# random_state is of big influence for val-auc
y = train_xy.label
X = train_xy.drop(['label', 'conversionTime'], axis=1)

val_y = val.label
val_X = val.drop(['label', 'conversionTime'], axis=1)

test = tests.drop(['instanceID', 'label'], axis=1)

xgb_val = xgb.DMatrix(val_X, label=val_y)
xgb_train = xgb.DMatrix(X, label=y)
xgb_test = xgb.DMatrix(test)

# return 训练和验证的错误率
watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]

print "跑到这里了xgb.train"
# training model
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=500)
print "跑到这里了save_model"
model.save_model('model/xgb.model')  # 用于存储训练出的模型
# print "best best_ntree_limit", model.best_ntree_limit  # did not save the best,why?
# print "best best_iteration", model.best_iteration  # get it?

print "跑到这里了model.predict"
# preds = model.predict(xgb_test, ntree_limit=model.best_iteration)
# test_y = model.predict(xgb_test, ntree_limit=model.best_iteration)

preds = model.predict(xgb_test, ntree_limit=model.best_iteration)
# labels = xgb_test.get_label()
xgb_result = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": preds})
# xgb_result.proba = preds
xgb_result.sort_values("instanceID", inplace=True)
xgb_result.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission.zip", "w") as fout:
    fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)

# print ('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))

# print ('correct=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) == labels[i]) / float(len(preds))))
"""
test_result = pd.DataFrame(columns=["userID", "creativeID", "label"])
test_result.userID = test_userID
test_result.creativeID = test_creativeID
test_result.label = test_y
test_result.to_csv("output/xgb_bytecup_output.csv", index=None, encoding='utf-8')  # remember to edit xgb.csv , add ""

cost_time = time.time() - start_time
print "", '\n', "cost time:", cost_time, "(s)"

# save feature score and feature information:  feature,score,min,max,n_null,n_gt1w
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
fs = []
for (key, value) in feature_score:
    fs.append("{0}, {1}\n".format(key, value))

with open('newfeatures/feature_score_{0}.csv'.format(6), 'w') as f:
    f.writelines("feature, score\n")
    f.writelines(fs)

# 逻辑回归
# X = X['label_huidalv']
# test = test['label_huidalv']
"""
"""
model1 = LogisticRegression()
model1.fit(X, y)
lr_y1 = model1.predict_log_proba(test)
lr_y2 = model1.predict(test)
lr_y3 = model1.predict_proba(test)
proba_test = model1.predict_proba(test)[:, 1]
lr_result = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
lr_result.sort_values("instanceID", inplace=True)
lr_result.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission.zip", "w") as fout:
    fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)
"""
"""
lr_result.userID = test_userID
lr_result.creativeID = test_creativeID
lr_result.label = lr_y3
lr_result.to_csv("output/lr_bytecup1010_1.csv", index=None, encoding='utf-8')  # remember to edit xgb.csv , add ""
"""

