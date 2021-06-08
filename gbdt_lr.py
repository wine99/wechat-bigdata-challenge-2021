import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss
import gc

import pickle
from evaluation import uAUC

train = pd.read_csv('./data/data_sample_0.2/offline_train/offline_train_12.csv')
val = pd.read_csv('./data/evaluate/evaluate_13.csv')

FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]

continuous_fea = ['device', 'videoplayseconds'] + [f'{b}sum' for b in FEA_COLUMN_LIST] + [f'{b}mean' for b in FEA_COLUMN_LIST]
category_fea = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
train[category_fea] = train[category_fea].astype('category')
val[category_fea] = val[category_fea].astype('category')

train['Label'] = train['read_comment']
train = train.drop(columns=FEA_COLUMN_LIST)
val['Label'] = val['read_comment']
val = val.drop(columns=FEA_COLUMN_LIST)
# data = pd.concat([train, val])
data = train.drop(columns=['play', 'stay', 'date_']+[f'{b}sum_user' for b in FEA_COLUMN_LIST])
val = val.drop(columns=['play', 'stay', 'date_']+[f'{b}sum_user' for b in FEA_COLUMN_LIST])


def pkl_save(filename,file):
    output = open(filename, 'wb')
    pickle.dump(file, output)
    output.close()


def pkl_load(filename):
    pkl_file = open(filename, 'rb')
    file = pickle.load(pkl_file)
    pkl_file.close()
    return file


def lr_model(data, category_fea, continuous_fea, test):
    # 连续特征归一化
    scaler = MinMaxScaler()
    for col in continuous_fea:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    # # 离散特征one-hot编码
    # for col in category_fea:
    #     onehot_feats = pd.get_dummies(data[col], prefix=col)
    #     data.drop([col], axis=1, inplace=True)
    #     data = pd.concat([data, onehot_feats], axis=1)

    # # 把训练集和测试集分开
    # train = data[data['Label'] != -1]
    # target = train.pop('Label')
    # test = data[data['Label'] == -1]
    # test.drop(['Label'], axis=1, inplace=True)

    train = data.drop(columns=['Label'])
    target = data.pop('Label')
    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2020)

    # 建立模型
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])  # −(ylog(p)+(1−y)log(1−p)) log_loss
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print('tr_logloss: ', tr_logloss)
    print('val_logloss: ', val_logloss)

    # # 模型预测
    # y_pred = lr.predict_proba(test)[:, 1]  # predict_proba 返回n行k列的矩阵，第i行第j列上的数值是模型预测第i个预测样本为某个标签的概率, 这里的1表示点击的概率
    # print('predict: ', y_pred[:10])  # 这里看前10个， 预测为点击的概率


# lr_model(data.copy(), category_fea, continuous_fea)


def gbdt_model(data, category_fea, continuous_fea, test):
    loss = 'cross_entropy'
    # # 离散特征one-hot编码
    # for col in category_fea:
    #     onehot_feats = pd.get_dummies(data[col], prefix=col)
    #     data.drop([col], axis=1, inplace=True)
    #     data = pd.concat([data, onehot_feats], axis=1)

    # # 训练集和测试集分开
    # train = data[data['Label'] != -1]
    # target = train.pop('Label')
    # test = data[data['Label'] == -1]
    # test.drop(['Label'], axis=1, inplace=True)

    train = data.drop(columns=['Label'])
    target = data.pop('Label')

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2020)

    # 建模
    gbm = lgb.LGBMClassifier(boosting_type='gbdt',  # 这里用gbdt
                             objective='binary',
                             bagging_freq=1,
                             bagging_fraction=0.8,
                             pos_bagging_fraction=1.0,
                             neg_bagging_fraction=0.5,
                             min_child_weight=0.5,
                             colsample_bytree=0.7,
                             num_leaves=100,
                             max_depth=12,
                             learning_rate=0.01,
                             n_estimators=10000
                             )

    gbm.fit(x_train, y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_names=['train', 'val'],
            eval_metric=loss,
            early_stopping_rounds=100,
            )

    pkl_save('gbdt.model.pkl', gbm)

    # tr_logloss = log_loss(y_train, gbm.predict_proba(x_train)[:, 1])  # −(ylog(p)+(1−y)log(1−p)) log_loss
    # val_logloss = log_loss(y_val, gbm.predict_proba(x_val)[:, 1])
    # print('tr_logloss: ', tr_logloss)
    # print('val_logloss: ', val_logloss)

    # gbm = pkl_load('gbdt.model.pkl')

    y_pred_train = gbm.predict_proba(x_train)[:, 1]
    save_df = [x_train['userid'].tolist(), x_train['feedid'].tolist(), y_train.tolist(), y_pred_train.tolist()]
    save_df = pd.DataFrame(save_df).T
    save_df.columns = ['userid', 'feedid', 'read_comment_truth', 'read_comment_pred']
    save_df.to_csv(f'read_comment_train_gbdt_{loss}.csv')
    del save_df
    print('train uauc')
    print(uAUC(y_train.tolist(), y_pred_train, x_train['userid'].tolist()))

    y_pred_val = gbm.predict_proba(x_val)[:, 1]
    save_df = [x_val['userid'].tolist(), x_val['feedid'].tolist(), y_val.tolist(), y_pred_val.tolist()]
    save_df = pd.DataFrame(save_df).T
    save_df.columns = ['userid', 'feedid', 'read_comment_truth', 'read_comment_pred']
    save_df.to_csv(f'read_comment_val_gbdt_{loss}.csv')
    del save_df
    print('val uauc')
    print(uAUC(y_val.tolist(), y_pred_val, x_val['userid'].tolist()))

    # 模型预测
    x_test = test.drop(columns=['Label'])
    y_test = test.pop('Label')
    y_pred_test = gbm.predict_proba(x_test)[:, 1]  # predict_proba 返回n行k列的矩阵，第i行第j列上的数值是模型预测第i个预测样本为某个标签的概率, 这里的1表示点击的概率
    save_df = [x_test['userid'].tolist(), x_test['feedid'].tolist(), y_test.tolist(), y_pred_test.tolist()]
    save_df = pd.DataFrame(save_df).T
    save_df.columns = ['userid', 'feedid', 'read_comment_truth', 'read_comment_pred']
    save_df.to_csv(f'read_comment_evaluate_gbdt_{loss}.csv')
    del save_df
    print('test uauc')
    print(uAUC(y_test.tolist(), y_pred_test.tolist(), x_test['userid'].tolist()))


gbdt_model(data.copy(), category_fea, continuous_fea, val)


'''
lgb+lr 还没改
'''
def gbdt_lr_model(data, category_feature, continuous_feature):  # 0.43616
    # 离散特征one-hot编码
    for col in category_feature:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    train = data[data['Label'] != -1]
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis=1, inplace=True)

    # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2020)

    gbm = lgb.LGBMClassifier(objective='binary',
                             subsample=0.8,
                             min_child_weight=0.5,
                             colsample_bytree=0.7,
                             num_leaves=100,
                             max_depth=12,
                             learning_rate=0.01,
                             n_estimators=1000,
                             )

    gbm.fit(x_train, y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_names=['train', 'val'],
            eval_metric='binary_logloss',
            early_stopping_rounds=100,
            )

    model = gbm.booster_

    gbdt_feats_train = model.predict(train, pred_leaf=True)
    gbdt_feats_test = model.predict(test, pred_leaf=True)
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns=gbdt_feats_name)

    train = pd.concat([train, df_train_gbdt_feats], axis=1)
    test = pd.concat([test, df_test_gbdt_feats], axis=1)
    train_len = train.shape[0]
    data = pd.concat([train, test])
    del train
    del test
    gc.collect()

    # # 连续特征归一化
    scaler = MinMaxScaler()
    for col in continuous_feature:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))

    for col in gbdt_feats_name:
        onehot_feats = pd.get_dummies(data[col], prefix=col)
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, onehot_feats], axis=1)

    train = data[: train_len]
    test = data[train_len:]
    del data
    gc.collect()

    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.3, random_state=2018)

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    print('tr-logloss: ', tr_logloss)
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print('val-logloss: ', val_logloss)
    y_pred = lr.predict_proba(test)[:, 1]
    print(y_pred[:10])


# gbdt_lr_model(data.copy(), category_fea, continuous_fea)
