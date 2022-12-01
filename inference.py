import argparse
import torch
import pytorch_lightning as pl
from setproctitle import setproctitle
from model import DropNet
import numpy as np
import pandas as pd
from tqdm import tqdm


def main(hparams):
    setproctitle("Lightning-Recommand")
    pl.seed_everything(hparams.seed)

    model = DropNet.load_from_checkpoint(hparams.model_path, args=hparams)
    model.cuda()
    model.eval()

    pred_list = list()
    tot_items = torch.tensor(list(range(hparams.n_items)), dtype=torch.long)
    tot_users = torch.tensor(list(range(hparams.n_users)), dtype=torch.long)
    for user in tqdm(tot_users, total=hparams.n_users):
        user_ids = torch.full((hparams.n_items,), user)
        # total_user_contents: onehot으로 sparse하게 처리된, 유저 정보 2차원 베열(유저순서인덱스X특징)
        # 유저정보를 item만큼 repeat하는 이유는, 전체 아이템에 대한 유저는 동일하기 때문입니다.
        user_infos = total_user_contents[user].repeat(hparams.n_items, 1)
        # total_item_contents: onehot으로 sparse하게 처리된, 아이템 정보 2차원 베열(아이템번호인덱스X특징)
        # 모든 아이템에 대해 한번씩 이사람에게 추천해줄 상대적 값을 출력해야하므로 item정보는 전부다 들어갑니다.
        eval_output = model(user_ids, tot_items, user_infos, total_item_contents).detach().numpy()
        pred_u_score = eval_output.reshape(-1)
        pred_u_idx = np.argsort(pred_u_score)[::-1]
        # 상위 100개 출력이므로 필요하시면 바꾸세요
        pred_list.append(list(pred_u_idx[:100]))
    pred = pd.DataFrame()
    pred["id"] = tot_users.tolist()
    pred["predicted_list"] = pred_list
    pred.to_csv("./indices_result.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", default=None, type=int, help="all seed")
    parser.add_argument("--local_rank", type=int, help="ddp local rank")
    parser.add_argument("--model_path", type=str, help="model output path")

    parser.add_argument(
        "--n_users",
        default=200,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--n_items",
        default=30000,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--dropout",
        default=0.05,
        type=float,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--emb_dim",
        default=256,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--layer_dim",
        default=256,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--n_items_features",
        default=300,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    parser.add_argument(
        "--n_users_features",
        default=200,
        type=int,
        help="The batch size per GPU/TPU core/CPU for evaluation.",
    )
    args = parser.parse_args()
    main(args)
