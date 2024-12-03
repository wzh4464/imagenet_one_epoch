###
# File: ./main.py
# Created Date: Tuesday, December 3rd 2024
# Author: Zihan
# -----
# Last Modified: Wednesday, 4th December 2024 1:21:22 am
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


# def replace_vit_attention(model, logger):
#     """
#     Replace the attention mechanism in a Vision Transformer model with custom attention.
#     """
#     logger.info("Replacing attention mechanism")
#     for module in model.encoder.layers:
#         embed_dim = module.self_attention.embed_dim
#         num_heads = module.self_attention.num_heads
#         module.self_attention = CustomMultiheadAttention(embed_dim, num_heads)
#     return model


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
    """
    if logger:
        logger.info("Starting SGD-based influence score computation")

    model.eval()

    # Step 1: Compute initial gradient u using the validation set
    u = [torch.zeros_like(p.data) for p in model.parameters()]
    total_val_loss = 0

    for batch_idx, (x_val, y_val) in enumerate(train_loader):
        x_val, y_val = x_val.to(device), y_val.to(device)
        loss = criterion(model(x_val), y_val)
        total_val_loss += loss.item()
        grad_params = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        for u_i, g_i in zip(u, grad_params):
            u_i += g_i.detach()
        if logger and batch_idx % 10 == 0:
            logger.info(
                f"Validation Batch {batch_idx}, Current loss: {loss.item():.4f}"
            )

    if logger:
        logger.info(f"Total Validation Loss: {total_val_loss / len(train_loader):.4f}")

    # Step 2: Iterate over training steps to compute influence scores
    influence_scores = np.zeros(len(train_loader.dataset))
    lr = 0.01  # Learning rate used in SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Assuming you have stored the training steps and corresponding indices
    # For simplicity, we'll iterate over the training data once
    for batch_idx, (x_tr, y_tr) in enumerate(train_loader):
        batch_start = batch_idx * train_loader.batch_size
        batch_end = batch_start + len(x_tr)
        indices = range(batch_start, batch_end)

        x_tr, y_tr = x_tr.to(device), y_tr.to(device)

        # Forward pass
        output = model(x_tr)
        loss = criterion(output, y_tr)
        loss.backward()

        # Update influence scores
        for idx in indices:
            influence = sum(
                (u_i * p.grad.data).sum().item()
                for u_i, p in zip(u, model.parameters())
                if p.grad is not None
            )
            influence_scores[idx] = lr * influence

        # Zero gradients after update
        optimizer.zero_grad()

        if logger and batch_idx % 10 == 0:
            logger.info(f"Processed Batch {batch_idx}/{len(train_loader)}")

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
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(
            dataset, batch_size=12, sampler=train_sampler, shuffle=False
        )

        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
        model = DDP(model, device_ids=[rank])

        criterion = torch.nn.CrossEntropyLoss()

        if rank == 0:
            # Compute influence scores only on the master process
            influence_scores = compute_influence_scores(
                model.module, train_loader, criterion, device, alpha=0.0, logger=logger
            )
            lowest_influence_indices = np.argsort(influence_scores)[:n]

            # Compute loss ranking
            sorted_indices_by_loss, sample_losses = compute_loss_ranking(
                model.module, train_loader, criterion, device, logger
            )
            top_loss_indices = sorted_indices_by_loss[:n]

            # Compare with modified indices
            modified_indices = set(dataset.modified_indices)
            influence_overlap = modified_indices.intersection(lowest_influence_indices)
            loss_overlap = modified_indices.intersection(top_loss_indices)

            logger.info("Comparison of two methods:")
            logger.info(f"Influence-based overlap count: {len(influence_overlap)}")
            logger.info(f"Loss-based overlap count: {len(loss_overlap)}")
            logger.info(
                f"Influence-based overlap ratio: {len(influence_overlap) / n:.4f}"
            )
            logger.info(f"Loss-based overlap ratio: {len(loss_overlap) / n:.4f}")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise
    finally:
        cleanup()

    if rank == 0:  # 只在主进程中保存结果
        # 获取预测结果和损失值
        predictions, sample_losses = compute_loss_ranking(
            model.module, train_loader, criterion, device, logger
        )

        # 保存实验结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"experiment_results_{m}_{n}_{timestamp}.npy"
        results = save_experiment_results(
            model.module, dataset, predictions, sample_losses, output_file
        )


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


def compute_loss_ranking(model, data_loader, criterion, device, logger=None):
    """
    计算每个样本的loss并返回预测结果
    """
    model.eval()
    n_samples = len(data_loader.dataset)
    sample_losses = np.zeros(n_samples)
    predictions = np.zeros(n_samples, dtype=np.int32)

    if logger:
        logger.info("Starting loss computation and predictions")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            # 计算loss和预测结果
            output = model(data)
            loss = F.cross_entropy(output, target, reduction="none")
            pred = output.argmax(dim=1)

            # 保存结果
            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + len(data)
            sample_losses[start_idx:end_idx] = loss.cpu().numpy()
            predictions[start_idx:end_idx] = pred.cpu().numpy()

            if logger and batch_idx % 10 == 0:
                logger.info(f"Processed batch {batch_idx}")

    return predictions, sample_losses


def save_experiment_results(
    model, dataset, predictions, losses, output_file="results.npy"
):
    """
    保存实验结果到numpy格式文件

    Args:
        model: 训练好的模型
        dataset: CustomImageNetDataset实例
        predictions: 模型预测结果
        losses: 每个样本的损失值
        output_file: 输出文件名
    """
    import numpy as np

    # 创建structured array的dtype
    dtype = np.dtype(
        [
            ("sample_ind", np.int32),
            ("filename", "U100"),  # Unicode string最大100字符
            ("category", "U50"),  # 类别名称
            ("original_label", np.int32),
            ("is_label_changed", np.bool_),
            ("predicted_label", np.int32),
            ("loss", np.float32),
        ]
    )

    # 创建结果数组
    n_samples = len(dataset)
    results = np.zeros(n_samples, dtype=dtype)

    # 填充数据
    for i in range(n_samples):
        img_path = dataset.data[i]
        filename = os.path.basename(img_path)
        category = os.path.basename(os.path.dirname(img_path))  # 获取父目录名作为类别

        results[i]["sample_ind"] = i
        results[i]["filename"] = filename
        results[i]["category"] = category
        results[i]["original_label"] = dataset.targets[i]
        results[i]["is_label_changed"] = i in dataset.modified_indices
        results[i]["predicted_label"] = predictions[i]
        results[i]["loss"] = losses[i]

    # 保存到文件
    np.save(output_file, results)
    print(f"Results saved to {output_file}")

    # 打印一些统计信息
    print("\nSummary Statistics:")
    print(f"Total samples: {n_samples}")
    print(f"Modified samples: {len(dataset.modified_indices)}")
    print(f"Average loss: {np.mean(losses):.4f}")
    print(
        f"Prediction accuracy: {100 * np.mean(predictions == np.array(dataset.targets)):.2f}%"
    )
    print("\nSample of results (first 5 rows):")
    print(results[:5])

    return results


if __name__ == "__main__":
    main()
