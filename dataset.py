import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

statistics_v = ['read_comment_count', 'like_count', 'click_avatar_count', 'forward_count', 'favorite_count', 'comment_count', 'follow_count']
statistics_u = ['read_comment_prob', 'like_prob', 'click_avatar_prob', 'forward_prob']
target = ['read_comment', 'like', 'click_avatar', 'forward']
# uv_info = ['videoplayseconds', 'play', 'stay']
uv_info = ['videoplayseconds']


class OurDataset(Dataset):
    def __init__(self, mode='train'):
        self.whole = pd.read_csv('./data/small_%s.csv' % mode)
        emb = pd.read_csv('./data/feed_embeddings.csv')
        self.emb_dict = dict(zip(list(emb['feedid'].values),
                                 emb['feed_embedding'].str.split(' ', expand=True).iloc[:, :-1].astype('float32').values.tolist()))
        self.len = self.whole.shape[0]

    def __getitem__(self, index):
        feedid = self.whole['feedid'].iloc[index]
        record = self.whole.iloc[index]
        emb = self.emb_dict[feedid]
        item = {
            'aid': torch.tensor(record['authorid']).long(),
            'feed_embedding': torch.tensor(emb),
            'statistics_v': torch.tensor(record[statistics_v].astype('float32').values),
            'uv_info': torch.tensor(record[uv_info].astype('float32').values),
            'uid': torch.tensor(record['userid']).long(),
            'did': torch.tensor(record['device']).long(),
            'statistics_u': torch.tensor(record[statistics_u].astype('float32').values),
            'target': torch.tensor(record[target].astype('int64').values).long()
        }
        return item

    def __len__(self):
        return self.len

# train_dataset = OurDataset(mode='train')
# val_dataset = OurDataset(mode='val')
