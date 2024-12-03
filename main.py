###
# File: ./main.py
# Created Date: Tuesday, December 3rd 2024
# Author: Zihan
# -----
# Last Modified: Tuesday, 3rd December 2024 3:22:14 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
import torch
import torchvision
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
import torch.multiprocessing as mp
from collections import defaultdict
import random
from typing import List, Dict, Tuple
import copy


class CustomImageNetDataset(Dataset):
    def __init__(self, root_dir: str, m: int, n: int, transform=None):
        """
        参数:
        root_dir: ImageNet数据集根目录
        m: 每个类别选择的图片数量
        n: 需要修改标签的鸽子图片数量
        transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.m = m
        self.n = n

        # ImageNet类别ID
        self.dove_id = "n01530575"  # 北朱雀
        self.parrot_id = "n01532829"  # 家朱雀

        # 获取类别文件夹列表
        self.classes = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 加载数据集
        self.data = []
        self.targets = []
        self.modified_indices = []  # 记录被修改标签的图片索引

        # 遍历所有类别文件夹
        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            images = os.listdir(class_dir)[:m]  # 每个类别取m张图片

            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                self.data.append(img_path)
                self.targets.append(self.class_to_idx[cls_name])

        # 随机选择n张鸽子图片修改标签
        dove_indices = [
            i for i, t in enumerate(self.targets) if self.classes[t] == self.dove_id
        ]
        selected_indices = random.sample(dove_indices, min(n, len(dove_indices)))
        self.modified_indices = selected_indices

        # 修改选中的鸽子图片标签为鹦鹉
        parrot_idx = self.class_to_idx[self.parrot_id]
        for idx in selected_indices:
            self.targets[idx] = parrot_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        target = self.targets[idx]

        # Load image
        img = torchvision.io.read_image(img_path)

        # Convert to float and normalize to [0, 1] range
        img = img.float() / 255.0

        # Ensure the image has 3 channels (convert grayscale to RGB)
        if img.shape[0] == 1:  # Grayscale image
            img = img.expand(3, -1, -1)

        if self.transform:
            img = self.transform(img)

        return img, target


def compute_influence_scores(
    model, train_loader, val_loader, criterion, device, alpha=0.01
):
    """
    按照原始infl_sgd的逻辑计算影响力分数

    Parameters:
    - model: 训练后的模型
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - criterion: 损失函数
    - device: 计算设备
    - alpha: 正则化参数
    """
    model.eval()

    # 1. 计算验证集梯度 u (与原代码中的compute_gradient对应)
    def compute_validation_gradient():
        u = [torch.zeros_like(p.data) for p in model.parameters()]
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            loss = criterion(model(x_val), y_val)
            # 添加L2正则化
            for p in model.parameters():
                loss += 0.5 * alpha * (p * p).sum()
            grad_params = torch.autograd.grad(loss, model.parameters())
            for u_i, g_i in zip(u, grad_params):
                u_i.data += g_i.data
        return u

    u = compute_validation_gradient()
    u = [uu.to(device) for uu in u]

    # 2. 计算每个训练样本的影响力分数
    n_samples = len(train_loader.dataset)
    influence_scores = np.zeros(n_samples)

    # 获取每个batch的学习率（这里简化为固定学习率）
    lr = 0.01

    for batch_idx, (x_tr, y_tr) in enumerate(train_loader):
        x_tr, y_tr = x_tr.to(device), y_tr.to(device)

        # 对batch中的每个样本分别计算
        for i in range(len(x_tr)):
            # 计算单个样本的损失和梯度
            z = model(x_tr[i : i + 1])
            loss = criterion(z, y_tr[i : i + 1])

            # 添加L2正则化项
            for p in model.parameters():
                loss += 0.5 * alpha * (p * p).sum()

            model.zero_grad()
            loss.backward()

            # 计算影响力分数
            sample_idx = batch_idx * train_loader.batch_size + i
            for j, param in enumerate(model.parameters()):
                influence_scores[sample_idx] += (
                    lr * (u[j].data * param.grad.data).sum().item()
                )

        # 更新u（与原代码中的u更新逻辑对应）
        z_batch = model(x_tr)
        loss_batch = criterion(z_batch, y_tr)
        for p in model.parameters():
            loss_batch += 0.5 * alpha * (p * p).sum()

        grad_params = torch.autograd.grad(
            loss_batch, model.parameters(), create_graph=True
        )
        ug = sum((uu * g).sum() for uu, g in zip(u, grad_params))
        model.zero_grad()
        ug.backward()

        for j, param in enumerate(model.parameters()):
            u[j] -= lr * param.grad.data

    return influence_scores


class ModelTrainer:
    def __init__(self, model, device, batch_size=32, learning_rate=0.01, alpha=0.01):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.lr = learning_rate
        self.alpha = alpha
        self.parameter_history = []
        self.info_history = []  # 记录每步的信息

    def save_step_info(self, batch_indices):
        """保存每一步的训练信息"""
        info = {"idx": batch_indices, "lr": self.lr}
        self.info_history.append(info)

        # 保存当前模型状态
        model_copy = copy.deepcopy(self.model)
        self.parameter_history.append(model_copy)

    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 获取当前batch的索引
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + len(data)
            batch_indices = np.arange(start_idx, end_idx)

            # 保存训练步骤信息
            self.save_step_info(batch_indices)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)

            # 添加L2正则化
            for p in self.model.parameters():
                loss += 0.5 * self.alpha * (p * p).sum()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, root_dir, m, n):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # 设置转换
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 创建数据集和分布式采样器
    full_dataset = CustomImageNetDataset(root_dir, m, n, transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset = Subset(full_dataset, list(range(train_size)))
    val_dataset = Subset(full_dataset, list(range(train_size, len(full_dataset))))

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler)

    # 加载预训练模型
    model = torchvision.models.vit_b_16(pretrained=True).to(device)
    model = DDP(model, device_ids=[rank])

    # 设置优化器和损失函数
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 计算影响力分数
    influence_scores = compute_influence_scores(
        model, train_loader, val_loader, criterion, device
    )

    # 获取最低影响力分数的样本
    lowest_n_indices = np.argsort(influence_scores)[:n]

    # 记录被修改的标签索引
    modified_indices_in_train = [
        idx for idx in full_dataset.modified_indices if idx in range(len(train_dataset))
    ]

    # 计算重合
    overlap = set(lowest_n_indices).intersection(set(modified_indices_in_train))
    overlap_count = len(overlap)
    overlap_ratio = overlap_count / min(n, len(modified_indices_in_train))

    print(
        f"Rank {rank}, Overlap Count: {overlap_count}, Overlap Ratio: {overlap_ratio:.4f}"
    )

    cleanup()


def main():
    ROOT_DIR = "/backup/imagenet/ILSVRC/Data/CLS-LOC/train"
    M = 100  # 每个类别选择的图片数量
    N = 10  # 需要修改标签的鸽子图片数量

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("DDP requires at least 2 GPUs!")
        return

    mp.spawn(train, args=(world_size, ROOT_DIR, M, N), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
