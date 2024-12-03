###
# File: ./main.py
# Created Date: Tuesday, December 3rd 2024
# Author: Zihan
# -----
# Last Modified: Wednesday, 4th December 2024 12:04:30 am
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
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch import distributed as dist
import torch.multiprocessing as mp
import random
import copy
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import logging
import sys
from datetime import datetime
import argparse


def setup_logger(rank):
    """
    设置日志记录器

    Args:
        rank: 当前进程的rank值
    """
    logger = logging.getLogger(f"Process_{rank}")
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(f"process_{rank}_{timestamp}.log")
    fh.setLevel(logging.INFO)

    # 创建控制台处理器
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - Process_%(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def replace_vit_attention(model, logger):
    """
    Replace the attention mechanism in a Vision Transformer model with custom attention.
    """
    logger.info("Replacing attention mechanism")
    for module in model.encoder.layers:
        embed_dim = module.self_attention.embed_dim
        num_heads = module.self_attention.num_heads
        module.self_attention = CustomMultiheadAttention(embed_dim, num_heads)
    return model


def evaluate_model(model, data_loader, criterion, device, logger=None):
    """
    评估模型性能
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    if logger:
        logger.info("Starting model evaluation")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if logger and batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}, Current loss: {loss.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    if logger:
        logger.info(f"Average loss: {avg_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy


class CustomAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        """
        Custom dot-product attention implementation.
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attention = CustomAttention()
        self.dropout = nn.Dropout(0.1)  # Adding dropout for regularization

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
    ):
        """
        Forward pass for multihead attention.
        Accepts both positional and keyword arguments for compatibility with ViT.
        """
        # Handle the case where all inputs are the same (self-attention)
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size = query.size(1)
        seq_len = query.size(0)

        # Project and reshape
        query = (
            self.query_proj(query)
            .view(seq_len, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        key = (
            self.key_proj(key)
            .view(seq_len, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        value = (
            self.value_proj(value)
            .view(seq_len, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        # Reshape for attention
        query = query.view(batch_size, self.num_heads, -1, self.head_dim)
        key = key.view(batch_size, self.num_heads, -1, self.head_dim)
        value = value.view(batch_size, self.num_heads, -1, self.head_dim)

        # Apply attention
        attn_output, attention_weights = self.attention(query, key, value)

        # Reshape and apply output projection
        attn_output = attn_output.view(batch_size, -1, seq_len, self.head_dim)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(seq_len, batch_size, self.embed_dim)

        # Apply output projection and dropout
        output = self.dropout(self.out_proj(attn_output))

        return output, attention_weights if need_weights else None


class CustomImageNetDataset(Dataset):
    def __init__(self, root_dir: str, m: int, n: int, transform=None):
        """
        参数:
        root_dir: ImageNet数据集根目录
        m: 总样本数量
        n: 需要修改标签的鸽子图片数量
        transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.m = m
        self.n = n

        # ImageNet类别ID
        self.dove_id = "n01530575"  # 北朱雀
        self.parrot_id = "n11939491"  # 家朱雀

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

        # 1. 首先处理需要修改标签的鸽子图片
        dove_dir = os.path.join(root_dir, self.dove_id)
        dove_images = os.listdir(dove_dir)

        # 随机选择n张鸽子图片作为要修改标签的样本
        selected_dove_images = random.sample(dove_images, min(n, len(dove_images)))
        parrot_idx = self.class_to_idx[self.parrot_id]

        # 添加这n张图片，并设置为鹦鹉标签
        for img_name in selected_dove_images:
            img_path = os.path.join(dove_dir, img_name)
            self.data.append(img_path)
            self.targets.append(parrot_idx)
            self.modified_indices.append(len(self.data) - 1)

        # 2. 然后随机选择 M-N 张任意类别的图片
        all_image_paths = []
        all_image_targets = []

        # 收集所有可用的图片
        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(class_dir):
                # 跳过已经选择的鸽子图片
                if cls_name == self.dove_id and img_name in selected_dove_images:
                    continue
                img_path = os.path.join(class_dir, img_name)
                all_image_paths.append(img_path)
                all_image_targets.append(self.class_to_idx[cls_name])

        # 随机选择 M-N 张图片
        remaining_samples = m - n
        if remaining_samples > 0:
            indices = random.sample(
                range(len(all_image_paths)),
                min(remaining_samples, len(all_image_paths)),
            )
            for idx in indices:
                self.data.append(all_image_paths[idx])
                self.targets.append(all_image_targets[idx])

        # 随机打乱数据集
        combined = list(zip(self.data, self.targets))
        random.shuffle(combined)
        self.data, self.targets = zip(*combined)

        # 更新修改索引的位置
        self.modified_indices = [
            i
            for i, (d, t) in enumerate(zip(self.data, self.targets))
            if d in [os.path.join(dove_dir, img) for img in selected_dove_images]
        ]

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
    model, train_loader, criterion, device, alpha=0.01, logger=None
):
    """
    计算影响力分数
    """
    if logger:
        logger.info("Starting influence score computation")

    model.eval()
    n_samples = len(train_loader.dataset)
    influence_scores = np.zeros(n_samples)

    # 计算总体梯度
    def compute_loss_gradient():
        if logger:
            logger.info("Computing loss gradient")
        u = [torch.zeros_like(p.data) for p in model.parameters()]
        total_loss = 0

        for batch_idx, (x_tr, y_tr) in enumerate(train_loader):
            x_tr, y_tr = x_tr.to(device), y_tr.to(device)
            loss = criterion(model(x_tr), y_tr)

            for p in model.parameters():
                loss += 0.5 * alpha * (p * p).sum()

            grad_params = torch.autograd.grad(loss, model.parameters())
            for u_i, g_i in zip(u, grad_params):
                u_i.data += g_i.data

            total_loss += loss.item()

            if logger and batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}, Current loss: {loss.item():.4f}")

        if logger:
            logger.info(f"Average loss: {total_loss/len(train_loader):.4f}")
        return u

    u = compute_loss_gradient()
    u = [uu.to(device) for uu in u]
    lr = 0.01

    if logger:
        logger.info(f"Computing influence scores for {n_samples} samples")

    # 计算每个样本的影响力分数
    for batch_idx, (x_tr, y_tr) in enumerate(train_loader):
        # 获取当前批次的实际索引
        start_idx = batch_idx * train_loader.batch_size
        end_idx = min(start_idx + len(x_tr), n_samples)
        indices = range(start_idx, end_idx)

        x_tr, y_tr = x_tr.to(device), y_tr.to(device)

        # 对批次中的每个样本计算影响力分数
        for i, idx in enumerate(indices):
            if idx >= n_samples:
                break

            # 计算单个样本的损失
            z = model(x_tr[i : i + 1])
            loss = criterion(z, y_tr[i : i + 1])

            # 添加正则化项
            for p in model.parameters():
                loss += 0.5 * alpha * (p * p).sum()

            # 计算梯度
            model.zero_grad()
            loss.backward()

            # 计算影响力分数
            score = 0
            for j, param in enumerate(model.parameters()):
                if param.grad is not None:
                    score += lr * (u[j].data * param.grad.data).sum().item()
            influence_scores[idx] = score

        if logger and batch_idx % 5 == 0:
            logger.info(f"Processed batch {batch_idx}/{len(train_loader)}")

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
    logger = setup_logger(rank)
    logger.info("=======================================")
    logger.info(f"Initializing process {rank} for m={m}, n={n}")

    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    logger.info(f"Using device: {device}")

    try:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 只使用训练集
        dataset = CustomImageNetDataset(root_dir, m, n, transform)
        logger.info(f"Dataset size: {len(dataset)}")

        train_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        train_loader = DataLoader(
            dataset, batch_size=12, sampler=train_sampler, shuffle=False
        )
        logger.info("Dataloader created successfully")

        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)

        # model = replace_vit_attention(model, logger).to(device)
        model = DDP(model, device_ids=[rank])
        logger.info("Model initialized successfully")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 模型评估
        logger.info("Evaluating initial model performance")
        logger.info("Dataset statistics:")
        logger.info(f"Total samples: {len(dataset)}")
        logger.info(f"Modified samples: {len(dataset.modified_indices)}")
        logger.info(f"Classes: {len(dataset.classes)}")

        # 统计标签分布
        # label_counts = {}
        # for target in dataset.targets:
        #     label_counts[target] = label_counts.get(target, 0) + 1
        # logger.info("Label distribution:")
        # for label, count in label_counts.items():
        #     class_name = dataset.classes[label]
        #     logger.info(f"Class {class_name}: {count} samples")

        # 评估模型性能
        initial_loss, initial_accuracy = evaluate_model(
            model, train_loader, criterion, device, logger
        )
        logger.info("Initial model performance:")
        logger.info(f"Loss: {initial_loss:.4f}")
        logger.info(f"Accuracy: {initial_accuracy:.2f}%")

        # 特别关注修改标签的样本
        modified_indices = dataset.modified_indices
        parrot_idx = dataset.class_to_idx[dataset.parrot_id]
        dove_idx = dataset.class_to_idx[dataset.dove_id]

        logger.info("\nModified samples statistics:")
        logger.info(f"Original class (dove) index: {dove_idx}")
        logger.info(f"Target class (parrot) index: {parrot_idx}")
        logger.info(f"Number of modified samples: {len(modified_indices)}")

        # 计算影响力分数
        logger.info("Starting influence score computation")
        influence_scores = compute_influence_scores(
            model, train_loader, criterion, device, logger=logger
        )

        lowest_n_indices = np.argsort(influence_scores)[:n]

        # 验证修改标签的点
        parrot_idx = dataset.class_to_idx[dataset.parrot_id]
        correct_modified = [
            i for i in dataset.modified_indices if dataset.targets[i] == parrot_idx
        ]
        incorrect_modified = [
            i for i in dataset.modified_indices if dataset.targets[i] != parrot_idx
        ]

        # 计算重叠
        overlap = set(lowest_n_indices).intersection(set(dataset.modified_indices))
        overlap_count = len(overlap)
        overlap_ratio = overlap_count / len(dataset.modified_indices)

        logger.info("Results:")
        logger.info(f"Total modified indices: {len(dataset.modified_indices)}")
        logger.info(f"Correctly labeled as parrot: {len(correct_modified)}")
        logger.info(f"Incorrectly labeled (not parrot): {len(incorrect_modified)}")
        logger.info(f"Overlap count: {overlap_count}")
        logger.info(f"Overlap ratio: {overlap_ratio:.4f}")

        # 检查重叠点的标签
        overlap_parrot = [i for i in overlap if dataset.targets[i] == parrot_idx]
        logger.info(
            f"Overlap points labeled as parrot: {len(overlap_parrot)}/{len(overlap)}"
        )

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Cleaning up process group")
        cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="ImageNet Training with Modified Labels"
    )
    parser.add_argument(
        "--backup-dir", type=str, default="/backup", help="backup directory path"
    )
    parser.add_argument(
        "--imagenet-dir", type=str, default="imagenet", help="imagenet directory name"
    )
    parser.add_argument(
        "-m",
        "--samples-per-class",
        type=int,
        required=True,
        help="number of samples per class",
    )
    parser.add_argument(
        "-n",
        "--modified-samples",
        type=int,
        required=True,
        help="number of modified dove samples",
    )

    args = parser.parse_args()

    # 构建完整的路径
    ROOT_DIR = os.path.join(
        args.backup_dir, args.imagenet_dir, "ILSVRC/Data/CLS-LOC/train"
    )

    # 验证路径是否存在
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory {ROOT_DIR} does not exist!")
        return

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("DDP requires at least 2 GPUs!")
        return

    mp.spawn(
        train,
        args=(world_size, ROOT_DIR, args.samples_per_class, args.modified_samples),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
