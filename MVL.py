import torch
import torch.nn as nn


class MVL(nn.Module):
    def __init__(self, feed_embedding_size=512, embedding_dim=20, mid_dim=64):
        super(MVL, self).__init__()

        # Missing value layer
        self.play_time = nn.Linear(feed_embedding_size + embedding_dim + 2, mid_dim)
        self.stay_time = nn.Linear(feed_embedding_size + embedding_dim + 2, mid_dim)
        self.uinfo_w = nn.ModuleList([nn.Linear(embedding_dim + 1, mid_dim),
                                      nn.Linear(embedding_dim + 1, mid_dim)])
        self.map = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])
        self.gelu = nn.GELU()

    def forward(self, aid, feed_embedding, uv_info, uinfo, mode):
        batch_size = aid.size(0)
        time_base = torch.cat((aid, feed_embedding, uv_info[:, :2]), dim=-1)  # [batch_size, 534]
        play_time, stay_time = torch.matmul(self.play_time(time_base).unsqueeze(1),
                                            self.uinfo_w[0](uinfo).unsqueeze(-1)).squeeze(-1), \
                               torch.matmul(self.stay_time(time_base).unsqueeze(1),
                                            self.uinfo_w[1](uinfo).unsqueeze(-1)).squeeze(-1)  # [batch_size, 1]
        play_time, stay_time = self.map[0](self.gelu(play_time)), self.map[1](self.gelu(stay_time))

        pmv_loss = None
        if mode == "test":
            uv_info = uv_info + torch.cat((torch.zeros(batch_size, 2), play_time, stay_time), dim=-1)
        elif mode == "train":
            pmv_loss_func = nn.SmoothL1Loss()
            pmv_loss = [pmv_loss_func(play_time, uv_info[:, -2].unsqueeze(-1)),
                        pmv_loss_func(stay_time, uv_info[:, -1].unsqueeze(-1))]
        elif mode == "validate":
            pmv_loss = [(play_time - uv_info[:, -2].unsqueeze(-1)).mean().tolist(),
                        (stay_time - uv_info[:, -1].unsqueeze(-1)).mean().tolist()]

        return pmv_loss, uv_info
