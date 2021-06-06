import os
import pandas as pd

# 存储数据的根目录
ROOT_PATH = "./data"
# 比赛数据集路径
DATASET_PATH = os.path.join(ROOT_PATH, "wechat_algo_data1")
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.csv")
FEED_EMBEDDINGS = os.path.join(DATASET_PATH, "feed_embeddings.csv")
# 测试集
TEST_FILE = os.path.join(DATASET_PATH, "test_a.csv")
END_DAY = 15
SEED = 2021

FEED_FEATURE = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward"]
# 复赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
# 每个行为的负样本下采样比例(下采样后负样本数/原负样本数)
# ACTION_SAMPLE_RATE = {"read_comment": 0.2, "like": 0.2, "click_avatar": 0.2, "forward": 0.1, "comment": 0.1, "follow": 0.1, "favorite": 0.1}
ACTION_SAMPLE_RATE = 0.2

# 各个阶段数据集的设置的最后一天
STAGE_END_DAY = {"online_train": 14, "offline_train": 12, "evaluate": 13, "submit": 15}
# 各个行为构造训练数据的天数
# ACTION_DAY_NUM = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 5, "comment": 5, "follow": 5, "favorite": 5}
ACTION_DAY_NUM = 5


def generate_sample(stage='offline_train'):
    day = STAGE_END_DAY[stage]
    if stage == "submit":
        sample_path = TEST_FILE
    else:
        sample_path = USER_ACTION
    stage_dir = os.path.join(ROOT_PATH, stage)
    df = pd.read_csv(sample_path)
    if stage == 'evaluate':
        # 线下评估
        df = df[df['date_'] == day]
        df['play'] = df['play'] / 1000
        df['stay'] = df['stay'] / 1000
    elif stage == 'submit':
        # 线上提交
        df['date_'] = 15
        df['play'] = 0
        df['stay'] = 0
    else:
        # 线下/线上训练
        # 同行为取按时间最近的样本
        ''''仅对初赛的四个行为进行去重'''
        for action in ACTION_LIST:
            df = df.drop_duplicates(subset=['userid', 'feedid', action], keep='last')
        df = df[(df['date_'] <= day) & (df['date_'] >= day - ACTION_DAY_NUM + 1)]
        df['positive'] = df['read_comment'] + \
                         df['like'] + \
                         df['click_avatar'] + \
                         df['forward'] + \
                         df['comment'] + \
                         df['follow'] + \
                         df['favorite']
        df_pos = df[df['positive'] > 0]
        df_neg = df[df['positive'] == 0]
        df_neg = df_neg.sample(frac=ACTION_SAMPLE_RATE, random_state=SEED, replace=False)
        df = pd.concat([df_pos, df_neg])
        df.drop(columns=['positive'], inplace=True)
        df['play'] = df['play'] / 1000
        df['stay'] = df['stay'] / 1000

    feed_info = pd.read_csv(FEED_INFO)[FEED_FEATURE]
    feed_info[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1
    feed_info[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        feed_info[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    feed_info = feed_info.set_index('feedid')
    # 基于userid统计的历史行为的次数
    user_date_feature = pd.read_csv(os.path.join(ROOT_PATH, "feature", "userid_feature.csv"))
    user_date_feature = user_date_feature.set_index(["userid", "date_"])
    user_date_prob_feature = pd.read_csv(os.path.join(ROOT_PATH, "feature", "userid_prob_feature.csv"))
    user_date_prob_feature = user_date_prob_feature.set_index(["userid", "date_"])
    # 基于feedid统计的历史行为的次数
    feed_date_feature = pd.read_csv(os.path.join(ROOT_PATH, "feature", "feedid_feature.csv"))
    feed_date_feature = feed_date_feature.set_index(["feedid", "date_"])

    df = df.merge(feed_info, on='feedid', how='left')
    df = df.merge(feed_date_feature, on=['feedid', 'date_'], how='left')
    df = df.merge(user_date_feature, on=['userid', 'date_'], how='left')
    df = df.merge(user_date_prob_feature, on=['userid', 'date_'], how='left')
    feed_feature_col = [b + "sum" for b in FEA_COLUMN_LIST]
    user_feature_col = [b + "sum_user" for b in FEA_COLUMN_LIST]
    user_prob_feature_col = [b + "mean" for b in FEA_COLUMN_LIST]
    df[feed_feature_col] = df[feed_feature_col].fillna(0.0)
    df[user_feature_col] = df[user_feature_col].fillna(0.0)
    df[user_prob_feature_col] = df[user_prob_feature_col].fillna(1e-5)
    df[feed_feature_col] = df[feed_feature_col].astype(int)
    df[user_feature_col] = df[user_feature_col].astype(int)
    df[["authorid", "bgm_song_id", "bgm_singer_id"]] = \
        df[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)
    file_name = os.path.join(stage_dir, stage + "_" + str(day) + ".csv")
    print(f'Save to: {file_name}')
    df.to_csv(file_name, index=False)


def main():
    for stage in ['offline_train', 'online_train', 'evaluate', 'submit']:
        generate_sample(stage)


if __name__ == "__main__":
    main()
