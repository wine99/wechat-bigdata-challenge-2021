import torch

import argparse
import yaml
from tqdm import tqdm

from NET import Net
from dataset import OurDataset, DataLoader
from evaluation import uAUC

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml",
    default="configs/model.yml",
    help="Path to a config file.",
)
parser.add_argument(
    "--load-pthpath",
    default="checkpoints/checkpoint_0.pth",
    help="To continue training, path to .pth file of saved checkpoint.",
)

if __name__ == '__main__':
    # For reproducibility.
    # Refer https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args = parser.parse_args()
    config = yaml.load(open(args.config_yml))

    val_dataset = OurDataset(mode='val')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=1,
        shuffle=False,
    )

    model = Net(config["task_num"], config["exp_per_task"], config["shared_num"],
                config["expert_size"], config["tower_size"], config["level_number"]).cuda()

    model_pth_dir = args.load_pthpath
    if model_pth_dir is not None:
        model.load_state_dict(torch.load(model_pth_dir))
    else:
        print("Pth dir error.")

    task_num = 4

    user_id_list = []
    Preds = [[] for i in range(task_num)]
    Labels = [[] for i in range(task_num)]

    model.eval()
    for i, batch in enumerate(tqdm(val_dataloader)):

        for key in batch:
            batch[key] = batch[key].cuda()

        with torch.no_grad():
            task_res, _ = model(batch["fid"], batch["aid"], batch["feed_embedding"], batch["statistics_v"],
                                batch["uv_info"], batch["uid"], batch["did"], batch["statistics_u"],
                                )

        uid_list = batch["uid"].tolist()  # batch_size,
        user_id_list = user_id_list + uid_list

        for j in range(task_num):
            Preds[j] = Preds[j] + task_res[j][:, 1].tolist()  # batch_size,
            Labels[j] = Labels[j] + batch["target"][:, j].tolist()  # batch_size,

    uAUC_list = [uAUC(Labels[i], Preds[i], user_id_list) for i in range(task_num)]
    print(uAUC_list)
