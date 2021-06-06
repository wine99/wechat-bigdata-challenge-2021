import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

STAGE_END_DAY = {"online_train": 14, "offline_train": 12, "evaluate": 13, "submit": 15}
DATA_ROOT = './data'
SEED = 2021

statistics = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
uv_info = ['videoplayseconds']
targets = ['read_comment', 'like', 'click_avatar', 'forward']

statistics_v_sum = [stat + 'sum' for stat in statistics]
statistics_u_sum = [stat + 'sum_user' for stat in statistics]
statistics_u_prob = [stat + 'mean' for stat in statistics]

emb_dict = pd.read_csv('./data/feed_embeddings.csv')
emb_dict = dict(zip(list(emb_dict['feedid'].values),
                    emb_dict['feed_embedding'].str.split(' ', expand=True).iloc[:, :-1].astype(
                        'float32').values.tolist()))


class OurDataset(Dataset):
    def __init__(self, mode):
        # mode = 'online_train' / 'offline_train' / 'evaluate' / 'submit'
        self.records_df = pd.read_csv(os.path.join(DATA_ROOT, mode, f'{mode}_{STAGE_END_DAY[mode]}.csv'))
        self.records_df = self.records_df.sample(frac=1, random_state=SEED, replace=False).reset_index(drop=True)
        self.records_gp = self.records_df[['userid']].groupby('userid').groups
        self.userid_list = tuple(self.records_gp)
        self.len = len(self.userid_list)
        self.mode = mode

    def __getitem__(self, index):
        def get_item(idx):
            record = self.records_df.iloc[idx]
            feedid = record['feedid']
            emb = emb_dict[feedid]
            item = {
                'fid': torch.tensor(feedid).long(),
                'aid': torch.tensor(record['authorid']).long(),
                'feed_embedding': torch.tensor(emb),
                'statistics_v': torch.tensor(record[statistics_v_sum].astype('float32').values),
                'uv_info': torch.tensor(record[uv_info].astype('float32').values),
                'uid': torch.tensor(record['userid']).long(),
                'did': torch.tensor(record['device']).long(),
                'statistics_u': torch.tensor(record[statistics_u_prob].astype('float32').values),
                'statistics_u_sum': torch.tensor(record[statistics_u_sum].astype('float32').values),
                'target': torch.tensor([0, 0, 0, 0]) if self.mode == 'submit' else torch.tensor(record[targets]).long()
            }
            return item
        userid = self.userid_list[index]
        records_indices = self.records_gp[userid]
        return [get_item(idx) for idx in records_indices]

    def __len__(self):
        return self.len
