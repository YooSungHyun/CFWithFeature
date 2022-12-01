import os
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from argparse import Namespace
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DropNetDataset(Dataset):
    def __init__(self, data_dir, x, y):
        self.data_dir = data_dir
        self.x = torch.LongTensor(x)
        self.y = torch.FloatTensor(y)
        # TODO: 아이템 특징 정보를 concat할 수 있도록, Sparse Matrix를 만드는 코드를 구현하세요.
        # Total User Contents 구현
        """
        Example
        성별, 나이구간, 고객태그 등을 이용해서 pandas의 onehot encoding을 이용하여, 특징전체별 sparse matrix를 만들어보세요
        """
        # Total Item Contents 구현
        """
        Example
        제품의 구성물질, 알레르기 가능 제품의 여부 등의 Item Feature를 User와 동일한 방식으로 만들어보세요.
        """

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        user_idx = self.x[idx][0]
        item_idx = self.x[idx][1]
        return (
            user_idx,
            item_idx,
            self.y[idx],
            # 위에서 만든 Total정보중에 해당하는 user와 item index에 해당하는 특징만 가져옵니다.
            self.total_user_contents[user_idx],
            self.total_item_contents[item_idx],
        )


class DropNetDataModule(pl.LightningDataModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.data_dir = args.data_dir
        self.seed = args.seed
        self.n_users = args.n_users
        self.n_items = args.n_items
        self.ratio = args.ratio
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
        self.num_proc = args.num_proc

    def prepare_data(self):
        if not os.path.isfile("./valid.pt"):
            # TODO Dropoutnet은 R matrix를 복원하는 학습이므로, 기존에 WMF or NeuralMF를 통해 만든 R_Matrix를 가져오십시오
            # 해당 matrix는 통상적이라면, 유저에 대한 아이템 구매 혹은 시청 여부 이력의 Sparse를 이용해 만들어진 WMF or NeuralMF의 최종 Output Matrix일 것입니다.
            r_matrix = np.load(self.data_dir)
            one_user_item_info = list()
            labels = list()
            for idx, y in tqdm(np.ndenumerate(r_matrix)):
                # user와 item에 대한 각 컨텐츠 정보를 추가
                one_user_item_info.append([idx[0], idx[1]])
                labels.append(y)
            del r_matrix
            train_x, valid_x, train_y, valid_y = train_test_split(one_user_item_info, labels, test_size=self.ratio)
            del one_user_item_info
            train_datasets = DropNetDataset(self.data_dir, train_x, train_y)
            valid_datasets = DropNetDataset(self.data_dir, valid_x, valid_y)
            torch.save(train_datasets, "./train.pt")
            torch.save(valid_datasets, "./valid.pt")
        else:
            pass

    def setup(self, stage: str):

        if stage == "fit":
            self.train_datasets = torch.load("./train.pt")
            self.valid_datasets = torch.load("./valid.pt")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_datasets,
            batch_size=self.per_device_train_batch_size,
            num_workers=self.num_proc,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_datasets,
            batch_size=self.per_device_train_batch_size,
            num_workers=self.num_proc,
            pin_memory=True,
        )
