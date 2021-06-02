import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

STAGE_END_DAY = {"online_train": 14, "offline_train": 12, "evaluate": 13, "submit": 15}
DATA_ROOT = './data'


statistics_v = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
statistics_u = ['read_comment', 'like', 'click_avatar', 'forward']
uv_info = ['videoplayseconds']
targets = ['read_comment', 'like', 'click_avatar', 'forward']

statistics_v = [stat+'sum' for stat in statistics_v]
statistics_u = [stat+'_prob' for stat in statistics_u]

emb_dict = pd.read_csv('./data/feed_embeddings.csv')
emb_dict = dict(zip(list(emb_dict['feedid'].values),
                    emb_dict['feed_embedding'].str.split(' ', expand=True).iloc[:, :-1].astype(
                        'float32').values.tolist()))


class OurDataset(Dataset):
    def __init__(self, target, records_df, mode):
        self.records_df = records_df
        self.len = self.records_df.shape[0]
        self.target = target
        self.mode = mode

    def __getitem__(self, index):
        feedid = self.records_df['feedid'].iloc[index]
        record = self.records_df.iloc[index]
        emb = emb_dict[feedid]
        item = {
            'fid': torch.tensor(feedid).long(),
            'aid': torch.tensor(record['authorid']).long(),
            'feed_embedding': torch.tensor(emb),
            'statistics_v': torch.tensor(record[statistics_v].astype('float32').values),
            'uv_info': torch.tensor(record[uv_info].astype('float32').values),
            'uid': torch.tensor(record['userid']).long(),
            'did': torch.tensor(record['device']).long(),
            'statistics_u': torch.tensor(record[statistics_u].astype('float32').values),
            'target': None if self.mode == 'submit' else torch.tensor(record[self.target]).long()
        }
        return item

    def __len__(self):
        return self.len


def generate_dataset(mode='online_train'):
    if mode == 'submit' or mode == 'evaluate':
        data_path = os.path.join(DATA_ROOT,
                                 mode,
                                 f'{mode}_all_{STAGE_END_DAY[mode]}_concate_sample.csv')
        records_df = pd.read_csv(data_path)
        records_df['device'] = (records_df['device'] - 1).astype('int64')

    datasets = []
    for target in targets:
        if not mode == 'submit' and not mode == 'evaluate':
            data_path = os.path.join(DATA_ROOT,
                                     mode,
                                     f'{mode}_{target}_{STAGE_END_DAY[mode]}_concate_sample.csv')
            records_df = pd.read_csv(data_path)
            records_df['device'] = (records_df['device'] - 1).astype('int64')
        datasets = datasets + [OurDataset(target, records_df, mode)]

    return datasets
