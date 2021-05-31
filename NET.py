import torch
import torch.nn as nn
from hashembed import HashEmbedding
from PLE import PLELayer
from MVL import MVL


class Net(nn.Module):
    def __init__(self, task_num, exp_per_task, shared_num,
                 expert_size, tower_size, level_number,
                 feed_embedding_size=512, statistics_dim=4, mid_dim=64,
                 num_feed=120000, num_author=20000, num_user=20000, num_device=2,
                 embedding_dim=20, seed=1234):
        super(Net, self).__init__()

        # id embedding
        self.feed_id_embedding = nn.ModuleList([HashEmbedding(num_feed, embedding_dim, num_feed // 10,
                                                              seed=seed, append_weight=False)
                                                for i in range(task_num)])
        self.author_id_embedding = nn.ModuleList([HashEmbedding(num_author, embedding_dim, num_author // 10,
                                                                seed=seed, append_weight=False)
                                                  for i in range(task_num)])
        self.user_id_embedding = nn.ModuleList([HashEmbedding(num_user, embedding_dim, num_user // 10,
                                                              seed=seed, append_weight=False)
                                                for i in range(task_num)])
        self.device_id_embedding = nn.ModuleList([nn.Embedding(num_device, 1) for i in range(task_num)])

        # Feature processing parameters
        self.hot_w = nn.Linear(statistics_dim, 1, bias=False)
        self.video_based_gate = nn.Linear(feed_embedding_size, statistics_dim)
        self.user_based_gate = nn.Linear(statistics_dim, statistics_dim)
        self.domain_based_gate = nn.Linear(statistics_dim, statistics_dim)

        # Missing value layer
        self.mvl = MVL(feed_embedding_size, embedding_dim, mid_dim)

        # bone net
        feature_size = embedding_dim * 3 + feed_embedding_size + 5
        self.netbone = nn.ModuleList([PLELayer(feature_size, 1, exp_per_task, shared_num,
                                               expert_size, tower_size, level_number)
                                      for i in range(task_num)])

        # dont forget to init weights
        self._init_weight()
        self.task_num = task_num

    def _init_weight(self):
        for m in self.modules():  # 继承nn.Module的方法
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, fid, aid, feed_embedding, statistics_v, uv_info,
                uid, did, statistics_u, mode=None, statistics_d=None):
        """
        fid:            feedid                                              [batch_size, 1]
        aid:            authorid                                            [batch_size, 1]
        feed_embedding: Multimodal embedding representing video             [batch_size, 512]
        statistics_v:   Statistics on video                                 [batch_size, 4]
        uv_info:        video time, play time, stay time                    [batch_size, 3]
        uid:            user id                                             [batch_size, 1]
        did:            device id                                           [batch_size, 1]
        statistics_u:   Statistics on user                                  [batch_size, 4]
        statistics_d:   Statistics on users in the same domain of interest  [batch_size, 4]
        """

        hot = self.hot_w(statistics_v)  # [batch_size, 1]
        uv_info = torch.cat((hot, uv_info), dim=-1)  # [batch_size, 4]

        # embed
        fid = [self.feed_id_embedding[i](fid.unsqueeze(-1)).squeeze() for i in range(self.task_num)]
        # task_num, [batch_size, 20]
        aid = [self.author_id_embedding[i](aid.unsqueeze(-1)).squeeze() for i in range(self.task_num)]
        # task_num, [batch_size, 20]
        uid = [self.user_id_embedding[i](uid.unsqueeze(-1)).squeeze() for i in range(self.task_num)]
        # task_num, [batch_size, 20]
        did = [self.device_id_embedding[i](did.unsqueeze(-1)).squeeze(-1) for i in range(self.task_num)]
        # task_num, [batch_size, 1]

        uinfo = [torch.cat((uid[i], did[i]), dim=-1) for i in range(self.task_num)]  # task_num, [batch_size, 21]

        # predict missing value
        feed_embedding = feed_embedding.squeeze()

        pmv_loss = None
        if mode is not None:
            if mode == "test":
                _, uv_info = self.mvl(aid, feed_embedding, uv_info, uinfo, mode)
            else:
                pmv_loss, _ = self.mvl(aid, feed_embedding, uv_info, uinfo, mode)

        # gate
        video_based_gate = torch.sigmoid(self.video_based_gate(feed_embedding))  # [batch_size, task_num]
        user_based_gate = torch.sigmoid(self.user_based_gate(uv_info))  # [batch_size, task_num]
        gate = video_based_gate + user_based_gate

        # tune statistics_u
        # domain gate
        if statistics_d is not None:
            domain_based_gate = torch.sigmoid(self.domain_based_gate(statistics_d))
            statistics_u = (1 + gate + domain_based_gate) * statistics_u
        else:
            statistics_u = (1 + gate) * statistics_u  # [batch_size, task_num]

        total_embedding = [torch.cat((fid[i], aid[i], feed_embedding, uv_info, uinfo[i]), dim=-1)
                           for i in range(self.task_num)]
        # task_num, [batch_size, feature_size]

        tasks_res = [self.netbone[i](total_embedding[i])[0] for i in range(self.task_num)]  # task_num, [batch_size, 2]

        statistics_u = statistics_u.transpose(0, 1).unsqueeze(-1)  # [task_num, batch_size, 1]
        statistics_u = torch.cat(((1 - statistics_u), statistics_u), dim=-1)  # [task_num, batch_size, 2]

        for i in range(len(tasks_res)):
            tasks_res[i] = tasks_res[i] + statistics_u[i]

        return tasks_res, pmv_loss
