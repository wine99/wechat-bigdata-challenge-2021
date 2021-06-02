import time
import pandas as pd
import torch

import argparse
import yaml
from tqdm import tqdm

from NET import Net
from dataset import generate_dataset, DataLoader
from evaluation import uAUC

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml",
    default="configs/model.yml",
    help="Path to a config file.",
)
parser.add_argument(
    "--load-pthpath",
    default=[None, None, None, None],
    help="To continue training, path to .pth file of saved checkpoint.",
)

if __name__ == '__main__':
    # For reproducibility.
    # Refer https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # change mode to 'submit' if submitting
    mode = 'submit'

    args = parser.parse_args()
    config = yaml.load(open(args.config_yml))

    val_dataset = generate_dataset(mode=mode)
    val_dataloader = [DataLoader(
        val_dataset[i],
        batch_size=config["batch_size"],
        num_workers=1,
        shuffle=False,
    ) for i in range(config["task_num"])]

    model = [Net(config["task_num"], config["exp_per_task"], config["shared_num"],
                 config["expert_size"], config["tower_size"], config["level_number"]).cuda()
             for i in range(config["task_num"])]

    model_pth_dir = args.load_pthpath

    for i in range(config["task_num"]):
        if model_pth_dir[i] is not None:
            model[i].load_state_dict(torch.load(model_pth_dir[i]))
        else:
            print(f"Task {i} pth dir error.")
            exit()

    uAUC_list = []
    submit_lists = []

    for n in range(config["task_num"]):
        user_id_list = []
        feed_id_list = []
        Preds = []
        Labels = []

        model[n].eval()

        for i, batch in enumerate(tqdm(val_dataloader[n])):
            for key in batch:
                batch[key] = batch[key].cuda()

            with torch.no_grad():
                task_res = model[n](batch["fid"], batch["aid"], batch["feed_embedding"], batch["statistics_v"],
                                    batch["uv_info"], batch["uid"], batch["did"], batch["statistics_u"],
                                    )

            uid_list = batch["uid"].tolist()  # batch_size,
            user_id_list = user_id_list + uid_list

            if mode == 'submit':
                feed_id_list = feed_id_list + batch['fid'].tolist()

            Preds = Preds + task_res[:, 1].tolist()  # batch_size,
            Labels = Labels + batch["target"].tolist()  # batch_size,

        if mode == 'submit':
            if n == config["task_num"] - 1:
                submit_lists = [user_id_list, feed_id_list] + submit_lists
            else:
                submit_lists.append(Preds)
        else:
            uAUC_list.append(uAUC(Labels, Preds, user_id_list))

    if mode == 'submit':
        submit = pd.DataFrame(submit_lists).T
        submit.columns = ['userid', 'feedid', 'read_comment', 'like', 'click_avatar', 'forward']
        submit.to_csv('./data/submit/submit_' + str(int(time.time())) + '.csv', index=False)
    else:
        print(uAUC_list)
