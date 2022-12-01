import torch
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError
from torch import nn
import numpy as np
import pandas as pd


def recallk(actual, predicted, k=25):
    set_actual = set(actual)
    recall_k = len(set_actual & set(predicted[:k])) / min(k, len(set_actual))
    return recall_k


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def ndcgk(actual, predicted, k=25):
    set_actual = set(actual)
    idcg = sum([1.0 / np.log(i + 2) for i in range(min(k, len(set_actual)))])
    dcg = 0.0
    unique_predicted = unique(predicted[:k])
    for i, r in enumerate(unique_predicted):
        if r in set_actual:
            dcg += 1.0 / np.log(i + 2)
    ndcg_k = dcg / idcg
    return ndcg_k


def evaluation(label, pred):
    label = label.groupby("id")["item_id"].unique().to_frame().reset_index()
    label.columns = ["id", "item_id"]

    evaluated_data = pd.merge(pred, label, how="left", on="id")

    evaluated_data["Recall"] = evaluated_data.apply(lambda x: recallk(x.item_id, x.preds), axis=1)
    evaluated_data["NDCG"] = evaluated_data.apply(lambda x: ndcgk(x.item_id, x.preds), axis=1)

    recall = evaluated_data["Recall"].mean()
    ndcg = evaluated_data["NDCG"].mean()

    rets = {"recall": recall, "ndcg": ndcg}
    return rets


class DropNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.loss_func = MeanSquaredError()
        self.n_users = args.n_users
        self.n_items = args.n_items
        self.emb_dim = args.emb_dim
        self.layer_dim = args.layer_dim
        self.dropout = args.dropout

        # TODO: WMF를 사용한 값으로 활용할 경우, 해당 데이터는 WMF로 학습된 User로 분해된 User X Hidden 값 그대로가 됩니다.
        # 즉 임베딩이 없어야 한다는 말입니다.
        self.Uin = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.emb_dim)
        self.u_dense_batch_fc_tanh = nn.Sequential(
            nn.Linear(self.emb_dim + args.n_users_features, args.finetuning_hidden_size),
            nn.BatchNorm1d(args.finetuning_hidden_size),
            nn.Tanh(),
            nn.Linear(args.finetuning_hidden_size, (args.finetuning_hidden_size // 2)),
            nn.BatchNorm1d((args.finetuning_hidden_size // 2)),
            nn.Tanh(),
        )
        self.U_embedding = nn.Linear((args.finetuning_hidden_size // 2), self.emb_dim)

        # TODO: WMF를 사용한 값으로 활용할 경우, 해당 데이터는 WMF로 학습된 Item으로 분해된 Item X Hidden 값 그대로가 됩니다.
        # 즉 임베딩이 없어야 한다는 말입니다.
        self.Vin = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.emb_dim)
        self.v_dense_batch_fc_tanh = nn.Sequential(
            nn.Linear(self.emb_dim + args.n_items_features, args.finetuning_hidden_size),
            nn.BatchNorm1d(args.finetuning_hidden_size),
            nn.Tanh(),
            nn.Linear(args.finetuning_hidden_size, (args.finetuning_hidden_size // 2)),
            nn.BatchNorm1d((args.finetuning_hidden_size // 2)),
            nn.Tanh(),
        )
        self.V_embedding = nn.Linear((args.finetuning_hidden_size // 2), self.emb_dim)

    def forward(self, user_indices, item_indices, user_contents, item_contents):
        self.Uin.eval()
        self.Vin.eval()
        with torch.no_grad():
            user_embedding = self.Uin(user_indices)
            u_last = torch.cat((user_embedding, user_contents), dim=-1)

            item_embedding = self.Vin(item_indices)
            v_last = torch.cat((item_embedding, item_contents), dim=-1)

        u_last = self.u_dense_batch_fc_tanh(u_last)
        v_last = self.v_dense_batch_fc_tanh(v_last)

        Uu = self.U_embedding(u_last)
        Vv = self.V_embedding(v_last)

        # TODO: 딥러닝으로 유저와 아이템 임베딩을 학습시킨 경우, 하위의 동작은 단순 multiply가 아닌, 실제로 딥러닝을 통해 WMF의 R 매트릭스를 찾아가는 과정이 되어야 합니다.
        # 또한 FineTuning 관점에서 접근해야하므로, 해당 Layer는 기존 학습한 모델에서 불러온 값으로 초기화 되어야 합니다.
        output = torch.multiply(Uu, Vv)
        return output

    def training_step(self, batch, batch_idx):
        # batch: 실제 데이터
        user_idx, item_idx, labels, total_user_contents, total_item_contents = batch
        logits = self(
            user_idx, item_idx, total_user_contents, total_item_contents
        )  # text_lengths must list or cpu cuda (list)
        loss = self.loss_func(logits, labels)
        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        user_idx, item_idx, labels, user_contents, item_contents = batch
        logits = self(user_idx, item_idx, user_contents, item_contents)
        loss = self.loss_func(logits, labels)

        return {"loss": loss, "user_idx": user_idx}

    def validation_epoch_end(self, validation_step_outputs):
        pred_list = list()
        loss_mean = torch.tensor([x["loss"] for x in validation_step_outputs], device=self.device).mean()
        valid_unique_users = torch.cat([x["user_idx"] for x in validation_step_outputs]).unique()
        tot_items = torch.tensor(list(range(self.n_items)), dtype=torch.long, device=self.device)
        for valid_user in valid_unique_users:
            valid_user_ids = torch.full((self.n_items,), valid_user, device=self.device)
            # TODO total_user_contents: onehot으로 sparse하게 처리된, 유저 정보 2차원 베열(유저순서인덱스X특징)
            # 유저정보를 item만큼 repeat하는 이유는, 전체 아이템에 대한 유저는 동일하기 때문입니다.
            valid_user_info = self.total_user_contents[valid_user].repeat(self.n_items, 1)
            # TODO total_item_contents: onehot으로 sparse하게 처리된, 아이템 정보 2차원 베열(아이템번호인덱스X특징)
            # 모든 아이템에 대해 한번씩 이사람에게 추천해줄 상대적 값을 출력해야하므로 item정보는 전부다 들어갑니다.
            eval_output = (
                self(valid_user_ids, tot_items, valid_user_info.cuda(), self.total_item_contents.cuda())
                .detach()
                .cpu()
                .numpy()
            )
            pred_u_score = eval_output.reshape(-1)
            pred_u_idx = np.argsort(pred_u_score)[::-1]
            # TODO 100개 Top을 뽑습니다. 필요시 수정
            pred_list.append(list(pred_u_idx[:100]))
        preds = pd.DataFrame()
        preds["id"] = valid_unique_users.cpu().tolist()
        preds["preds"] = pred_list
        labels = self.labels[self.labels.id.isin(valid_unique_users.cpu().tolist())].sort_values(by=["id", "item_id"])

        metrics = evaluation(labels, preds)
        self.log("val_loss", loss_mean, sync_dist=True)
        self.log_dict(metrics, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [{"params": [p for p in self.parameters()], "name": "OneCycleLR"}],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.args.warmup_ratio,
            epochs=self.trainer.max_epochs,
            final_div_factor=self.args.final_div_factor,
        )
        lr_scheduler = {"interval": "step", "scheduler": scheduler, "name": "AdamW"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
