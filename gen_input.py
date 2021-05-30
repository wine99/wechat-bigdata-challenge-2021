import numpy as np
import pandas as pd

feed_factor = ['authorid', 'videoplayseconds']
heat_factor = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
user_factor = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]
action_factor = ['device', 'play', 'stay']
labels = ['read_comment', 'like', 'click_avatar', 'forward']

user_action = pd.read_csv('./data/wechat_algo_data1/user_action.csv')
# user_action = user_action[user_action.date_ <= 7]


def prepare_feed():
    # TODO videoplayseconds有大于六十的
    feed = pd.read_csv('./data/wechat_algo_data1/feed_info.csv')[['feedid'] + feed_factor]
    # feed['videoplayseconds'] = feed['videoplayseconds'].where(feed['videoplayseconds'] <= 61, 61)
    grouped = user_action.groupby(by=['feedid'])
    for col in heat_factor:
        feed = feed.merge(grouped[col].sum().reset_index().rename(columns={col: col + '_count'}), on='feedid')
    return feed


def prepare_user():
    stat = pd.Series(name='userid', data=user_action['userid'].unique())
    stat = pd.DataFrame(stat)
    grouped = user_action.groupby(by=['userid'])
    for col in user_factor:
        stat = stat.merge(grouped[col].mean().round(decimals=6).reset_index().rename(columns={col: col + '_prob'}),
                          on='userid')
        # stat = stat.merge(grouped[col].sum().reset_index().rename(columns={col: col + '_count'}), on='userid')
    return stat


def prepare_input():
    feed = prepare_feed()
    user_stat = prepare_user()
    # emb = pd.read_csv('data/feed_embeddings.csv')
    # emb = emb[['feedid']].merge(emb['feed_embedding'].str.split(' ', expand=True).iloc[:, 1:-1],
    #                             how='left',
    #                             left_index=True,
    #                             right_index=True)
    inputs = user_action[['feedid', 'date_', 'userid'] + action_factor + labels]
    if 'play' in action_factor:
        inputs['play'] = inputs['play'] / 1000
    if 'stay' in action_factor:
        inputs['stay'] = inputs['stay'] / 1000
    inputs['device'] = inputs['device'] - 1
    inputs = inputs.merge(feed, on='feedid', how='left')
    inputs = inputs.merge(user_stat, on='userid', how='left')
    # for day in range(14):
    #     print(day+1)
    #     day_data = inputs[inputs.date_ == day+1]
    #     day_data = day_data.merge(emb, on='feedid', how='left')
    #     day_data.to_csv('./data/input_%day' % day, index=False)

    # no feedid
    # return inputs.iloc[:, 1:]
    return inputs


def prepare_label():
    return user_action[labels]


if __name__ == '__main__':
    inputs = prepare_input()
    inputs[inputs.date_ <= 12].to_csv('./data/input_train.csv', index=False)
    inputs[inputs.date_ >= 13].to_csv('./data/input_val.csv', index=False)
    # labels = prepare_label()
    # prepare_label().to_csv('./data/label_train.csv')
