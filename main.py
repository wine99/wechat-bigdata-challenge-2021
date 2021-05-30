import torch
from torch import nn, optim

import argparse
import yaml
from tqdm import tqdm
from tensorboardX import SummaryWriter

from NET import Net
from dataset import OurDataset, DataLoader
from AutomaticWeightedLoss.AutomaticWeightedLoss import AutomaticWeightedLoss

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml",
    default="configs/model.yml",
    help="Path to a config file.",
)
parser.add_argument(
    "--save-dirpath",
    default="checkpoints/",
    help="Path of directory to create checkpoint directory and save checkpoints.",
)
parser.add_argument(
    "--load-pthpath",
    default=None,
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

    train_dataset = OurDataset(mode='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=1,
        shuffle=True,
    )

    model = Net(config["task_num"], config["exp_per_task"], config["shared_num"],
                config["expert_size"], config["tower_size"], config["level_number"]).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    criterion = nn.CrossEntropyLoss()
    awl = AutomaticWeightedLoss(4)

    optimizer = optim.Adamax([
        {'params': model.parameters()},
        {'params': awl.parameters(), 'weight_decay': 0}
    ], lr=config["lr"])
    summary_writer = SummaryWriter(log_dir="logs/")
    iterations = len(train_dataset) // config["batch_size"] + 1

    start_epoch = 0

    model_pth_dir = args.load_pthpath
    if model_pth_dir is not None:
        start_epoch = model_pth_dir[-5:-4] + 1
        model.load_state_dict(torch.load(model_pth_dir))
    global_iteration_step = start_epoch * iterations
    batch_size = config["batch_size"]

    for epoch in range(start_epoch, 2):
        print(f"\nTraining for epoch {epoch}:")

        for i, batch in enumerate(tqdm(train_dataloader)):

            """if i == 3600:
                break"""

            for key in batch:
                batch[key] = batch[key].cuda()

            task_res, _ = model(batch["aid"], batch["feed_embedding"], batch["statistics_v"],
                                batch["uv_info"], batch["uid"], batch["did"], batch["statistics_u"])

            act_loss = [criterion(task_res[i], batch["target"][:, i]) for i in range(len(task_res))]
            loss_sum = awl(act_loss[0], act_loss[1], act_loss[2], act_loss[3])

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            summary_writer.add_scalar(
                "train/total_loss", loss_sum, i + global_iteration_step
            )
            summary_writer.add_scalar(
                "train/read_loss", act_loss[0], i + global_iteration_step
            )
            summary_writer.add_scalar(
                "train/like_loss", act_loss[1], i + global_iteration_step
            )
            summary_writer.add_scalar(
                "train/click_loss", act_loss[2], i + global_iteration_step
            )
            summary_writer.add_scalar(
                "train/forward_loss", act_loss[3], i + global_iteration_step
            )

        torch.cuda.empty_cache()
        torch.save(model.state_dict(), args.save_dirpath + "checkpoint_" + str(epoch) + ".pth")
