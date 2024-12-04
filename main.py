###
# File: ./main.py
# Created Date: Tuesday, December 3rd 2024
# Author: Zihan
# -----
# Last Modified: Wednesday, 4th December 2024 10:41:47 am
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
from torch.utils.data import Dataset, DataLoader, DistributedSampler
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
import pandas as pd


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

        # 添加这n张图片, 并设置为鹦鹉标签
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
        modified_img_paths = set(
            [os.path.join(dove_dir, img) for img in selected_dove_images]
        )
        self.modified_indices = [
            i
            for i, (d, t) in enumerate(zip(self.data, self.targets))
            if d in modified_img_paths
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


# Replace the existing compute_influence_scores with compute_influence_scores_sgd
def compute_influence_scores(
    model, train_loader, criterion, device, alpha=1.0, logger=None
):
    return compute_influence_scores_sgd(
        model, train_loader, criterion, device, alpha, logger
    )


def compute_influence_scores_sgd(
    model, train_loader, criterion, device, alpha=1.0, logger=None
):
    """
    Compute influence scores using the SGD-based method similar to infl_sgd.

    Args:
        model: The trained model.
        train_loader: DataLoader for the training data.
        criterion: Loss function.
        device: Device to perform computations on.
        alpha: Regularization parameter.
        logger: Logger for logging information.

    Returns:
        influence_scores: Numpy array of influence scores for each training sample.
        sample_losses: Numpy array of loss values for each training sample.
    """
    if logger:
        logger.info("Starting SGD-based influence score computation")

    model.eval()

    # Step 1: Compute initial gradient u using the training set
    u = [torch.zeros_like(p.data) for p in model.parameters()]
    total_train_loss = 0

    for batch_idx, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)
        loss = criterion(model(x_train), y_train)
        total_train_loss += loss.item()
        grad_params = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        for u_i, g_i in zip(u, grad_params):
            u_i += g_i.detach()
        if logger and batch_idx % 10 == 0:
            logger.info(f"Training Batch {batch_idx}, Current loss: {loss.item():.4f}")

    if logger:
        logger.info(f"Total Training Loss: {total_train_loss / len(train_loader):.4f}")

    # Step 2: Iterate over training steps to compute influence scores
    influence_scores = np.zeros(len(train_loader.dataset))
    sample_losses = np.zeros(len(train_loader.dataset))  # 初始化样本损失
    lr = 0.01  # Learning rate used in SGD

    # Iterate over training data once
    for batch_idx, (x_tr, y_tr) in enumerate(train_loader):
        idx = batch_idx  # 因为 batch_size=1，每个批次对应一个样本

        x_tr, y_tr = x_tr.to(device), y_tr.to(device)

        # Forward pass
        output = model(x_tr)
        loss = criterion(output, y_tr)
        loss.backward()

        # Update influence score for this sample
        influence = sum(
            (u_i * p.grad.data).sum().item()
            for u_i, p in zip(u, model.parameters())
            if p.grad is not None
        )
        influence_scores[idx] = lr * influence

        # Compute individual loss
        individual_loss = (
            F.cross_entropy(output, y_tr, reduction="none").detach().cpu().numpy()[0]
        )
        sample_losses[idx] = individual_loss

        # Zero gradients after update
        model.zero_grad()

        if logger and batch_idx % 10 == 0:
            logger.info(f"Processed Batch {batch_idx}/{len(train_loader)}")

    return influence_scores, sample_losses


def compute_loss_ranking(model, data_loader, criterion, device, logger=None):
    """
    计算每个样本的loss并返回预测结果

    Args:
        model: 训练好的模型
        data_loader: DataLoader实例
        criterion: 损失函数
        device: 设备
        logger: 日志记录器

    Returns:
        predictions: Numpy数组，模型的预测结果
        sample_losses: Numpy数组，每个样本的损失值
    """
    model.eval()
    sample_losses = []
    predictions = []

    if logger:
        logger.info("Starting loss computation and predictions")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            # 计算loss和预测结果
            output = model(data)
            loss = F.cross_entropy(output, target, reduction="none").detach()
            pred = output.argmax(dim=1)

            # 保存结果
            sample_losses.extend(loss.cpu().numpy())
            predictions.extend(pred.cpu().numpy())

            if logger and batch_idx % 10 == 0:
                logger.info(f"Processed batch {batch_idx}")

    return np.array(predictions), np.array(sample_losses)


def save_experiment_results(
    model, dataset, predictions, losses, influence_scores, output_file=None
):
    """
    保存实验结果到CSV文件

    Args:
        model: 训练好的模型
        dataset: CustomImageNetDataset实例
        predictions: 模型预测结果
        losses: 每个样本的损失值
        influence_scores: 每个样本的影响分数
        output_file: 输出文件名，如果为None则自动生成
    """
    import pandas as pd
    from datetime import datetime
    import os

    # 如果没有指定输出文件名，则自动生成
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"experiment_results_{timestamp}.csv"

    # 准备数据
    results = {
        "sample_ind": list(range(len(dataset))),
        "filename": [os.path.basename(path) for path in dataset.data],
        "category": [os.path.basename(os.path.dirname(path)) for path in dataset.data],
        "original_label": dataset.targets,
        "is_label_changed": [
            i in dataset.modified_indices for i in range(len(dataset))
        ],
        "predicted_label": predictions,
        "loss": losses,
        "influence_score": influence_scores,  # 添加 influence_score 列
    }

    # 创建 DataFrame 并保存
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    # 打印统计信息
    print(f"\nResults saved to {output_file}")
    print("\nSummary Statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Modified samples: {len(dataset.modified_indices)}")
    print(f"Average loss: {df['loss'].mean():.4f}")
    print(f"Average influence score: {df['influence_score'].mean():.4f}")
    print(
        f"Prediction accuracy: {100 * (df['predicted_label'] == df['original_label']).mean():.2f}%"
    )

    # 打印前几行示例
    print("\nSample of results (first 5 rows):")
    print(df.head())

    return df


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, root_dir, m, n):
    logger = setup_logger(rank)
    logger.info(f"Initializing process {rank} for m={m}, n={n}")

    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
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

        dataset = CustomImageNetDataset(root_dir, m, n, transform)
        train_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        train_loader = DataLoader(
            dataset, batch_size=1, sampler=train_sampler, shuffle=False, drop_last=False
        )

        logger.info(f"Number of samples in this process: {len(train_loader.dataset)}")

        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
        model = DDP(model, device_ids=[rank])

        criterion = torch.nn.CrossEntropyLoss()

        # 计算 influence scores 和 sample_losses
        influence_scores, sample_losses = compute_influence_scores(
            model.module, train_loader, criterion, device, alpha=0.01, logger=logger
        )

        # 计算 predictions
        predictions, _ = compute_loss_ranking(
            model.module, train_loader, criterion, device, logger
        )

        # 将结果转换为列表
        influence_scores = influence_scores.tolist()
        sample_losses = sample_losses.tolist()
        predictions = predictions.tolist()

        # 使用 gather_object 收集数据
        if rank == 0:
            gathered_influence_scores = [None] * world_size
            gathered_sample_losses = [None] * world_size
            gathered_predictions = [None] * world_size
        else:
            gathered_influence_scores = None
            gathered_sample_losses = None
            gathered_predictions = None

        dist.gather_object(
            influence_scores, object_gather_list=gathered_influence_scores, dst=0
        )
        dist.gather_object(
            sample_losses, object_gather_list=gathered_sample_losses, dst=0
        )
        dist.gather_object(predictions, object_gather_list=gathered_predictions, dst=0)

        if rank == 0:
            # 合并所有进程的数据
            combined_influence_scores = []
            combined_sample_losses = []
            combined_predictions = []

            for r in range(world_size):
                combined_influence_scores.extend(gathered_influence_scores[r])
                combined_sample_losses.extend(gathered_sample_losses[r])
                combined_predictions.extend(gathered_predictions[r])

            logger.info(
                f"Total influence scores collected: {len(combined_influence_scores)}"
            )
            logger.info(f"Total sample losses collected: {len(combined_sample_losses)}")
            logger.info(f"Total predictions collected: {len(combined_predictions)}")

            # 保存实验结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"experiment_results_m{m}_n{n}_{timestamp}.csv"
            save_experiment_results(
                model.module,
                dataset,
                combined_predictions,
                combined_sample_losses,
                combined_influence_scores,
                output_file,
            )

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise
    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="ImageNet Training with Modified Labels and Influence Scores"
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
