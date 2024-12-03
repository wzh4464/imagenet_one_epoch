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
        self.parrot_id = "n11939491"  # 家鹦鹉

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

        if len(dove_images) < n:
            raise ValueError(
                f"Requested {n} modified dove samples, but only {len(dove_images)} available."
            )

        # 随机选择n张鸽子图片作为要修改标签的样本
        selected_dove_images = random.sample(dove_images, n)
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

        # 检查是否有足够的样本
        remaining_samples = m - n
        if remaining_samples > len(all_image_paths):
            raise ValueError(
                f"Requested {remaining_samples} non-dove samples, but only {len(all_image_paths)} available."
            )

        # 随机选择 M-N 张图片
        selected_indices = random.sample(range(len(all_image_paths)), remaining_samples)
        for idx in selected_indices:
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

        # Return img and target
        return img, target


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


def get_global_data_list(root_dir, m):
    """
    获取整个数据集的文件路径列表，限制为 m 个样本
    """
    data_list = []
    for cls_name in sorted(os.listdir(root_dir)):
        cls_dir = os.path.join(root_dir, cls_name)
        if os.path.isdir(cls_dir):
            for img_name in sorted(os.listdir(cls_dir)):
                img_path = os.path.join(cls_dir, img_name)
                data_list.append(img_path)
                if len(data_list) >= m:
                    return data_list
    return data_list


def get_original_label(dataset, index):
    """
    根据索引获取原始标签
    """
    return dataset.targets[index]


def save_experiment_results(
    root_dir,
    predictions,
    losses,
    output_file,
    modified_indices,
    global_data,
    global_labels,
):
    """
    保存实验结果到CSV文件
    """
    results = {
        "sample_ind": [],
        "filename": [],
        "category": [],
        "original_label": [],
        "is_label_changed": [],
        "predicted_label": [],
        "loss": [],
    }

    # 遍历所有样本并填充数据
    for i in range(len(predictions)):
        img_path = global_data[i]
        filename = os.path.basename(img_path)
        category = os.path.basename(os.path.dirname(img_path))

        results["sample_ind"].append(i)
        results["filename"].append(filename)
        results["category"].append(category)
        results["original_label"].append(global_labels[i])
        results["is_label_changed"].append(i in modified_indices)
        results["predicted_label"].append(int(predictions[i]))
        results["loss"].append(float(losses[i]))

    # 创建DataFrame并保存
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    # 打印统计信息
    print(f"\nResults saved to {output_file}")
    print("\nSummary Statistics:")
    print(f"Total samples: {len(predictions)}")
    print(f"Modified samples: {len(modified_indices)}")
    print(f"Average loss: {df['loss'].mean():.4f}")
    print(
        f"Prediction accuracy: {100 * (df['predicted_label'] == df['original_label']).mean():.2f}%"
    )

    return df


def compute_loss_ranking(model, data_loader, criterion, device, logger=None):
    """
    计算每个样本的loss并返回预测结果，支持DDP环境
    """
    model.eval()
    local_sample_losses = []
    local_predictions = []

    if logger:
        logger.info("Starting loss computation and predictions")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            # 计算输出和损失
            output = model(data)
            loss = F.cross_entropy(output, target, reduction="none")
            pred = output.argmax(dim=1)

            # 收集结果
            local_sample_losses.append(loss.cpu())
            local_predictions.append(pred.cpu())

            if logger and batch_idx % 10 == 0:
                logger.info(
                    f"Batch {batch_idx}, Current loss: {loss.mean().item():.4f}"
                )

    # 将本地结果拼接起来
    local_sample_losses = torch.cat(local_sample_losses).numpy()
    local_predictions = torch.cat(local_predictions).numpy()

    # Gather all results到主进程
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    gathered_losses = [None for _ in range(world_size)]
    gathered_preds = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_losses, local_sample_losses)
    dist.all_gather_object(gathered_preds, local_predictions)

    # 仅在主进程中处理和返回结果
    if rank == 0:
        all_losses = np.concatenate(gathered_losses)
        all_preds = np.concatenate(gathered_preds)
        return all_preds, all_losses

    return None, None


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

        # 初始数据集，用于分配索引
        initial_dataset = CustomImageNetDataset(root_dir, m, n, transform)
        train_sampler = DistributedSampler(
            initial_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

        # 设置 epoch (对于分布式采样器来说，这有助于确保每次迭代的数据随机性)
        train_sampler.set_epoch(0)

        # 获取当前进程的子集索引
        subset_indices = list(train_sampler)

        # 创建包含当前进程子集的子数据集
        dataset = Subset(initial_dataset, subset_indices)
        train_loader = DataLoader(
            dataset, batch_size=12, sampler=train_sampler, shuffle=False
        )

        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
        model = DDP(model, device_ids=[rank])

        criterion = torch.nn.CrossEntropyLoss()

        # 获取预测结果和损失值（所有进程都参与计算）
        predictions, sample_losses = compute_loss_ranking(
            model.module, train_loader, criterion, device, logger
        )

        # Gather all subset_indices到主进程
        gathered_subset_indices = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_subset_indices, subset_indices)

        # 只在主进程中保存结果
        if rank == 0 and predictions is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"experiment_results_m{m}_n{n}_{timestamp}.csv"

            # 获取全局数据列表和标签列表
            global_data = get_global_data_list(root_dir, m)
            global_labels = [
                get_original_label(initial_dataset, i) for i in range(len(global_data))
            ]

            # 获取所有样本的修改索引
            modified_indices = initial_dataset.modified_indices

            # 创建一个空数组用于存储所有预测和损失
            all_preds = np.empty(len(global_data), dtype=np.int64)
            all_losses = np.empty(len(global_data), dtype=np.float32)

            # 填充所有_preds 和 all_losses
            for proc_rank in range(world_size):
                proc_subset = gathered_subset_indices[proc_rank]
                proc_length = len(proc_subset)
                proc_preds = predictions[:proc_length]
                proc_losses = sample_losses[:proc_length]
                # 将当前进程的预测和损失映射到全局索引
                all_preds[proc_subset] = proc_preds
                all_losses[proc_subset] = proc_losses

                # Remove processed data
                predictions = predictions[proc_length:]
                sample_losses = sample_losses[proc_length:]

            # 保存实验结果
            results_df = save_experiment_results(
                root_dir,
                all_preds,
                all_losses,
                output_file,
                modified_indices,
                global_data,
                global_labels,
            )
            logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise
    finally:
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

    # 获取类别列表
    classes = sorted(
        [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    )
    num_classes = len(classes)

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("DDP requires at least 2 GPUs!")
        return

    # Compute total samples
    total_samples = args.samples_per_class * num_classes

    # Check if total_samples is divisible by world_size
    if total_samples % world_size != 0:
        print(
            f"Warning: Total samples {total_samples} is not divisible by world_size={world_size}. Setting drop_last=True."
        )
        drop_last = True
    else:
        drop_last = False

    # Adjust total_samples if necessary
    if drop_last:
        total_samples = (total_samples // world_size) * world_size
        print(
            f"Adjusted total samples to {total_samples} to be divisible by world_size={world_size}."
        )

    mp.spawn(
        train,
        args=(world_size, ROOT_DIR, total_samples, args.modified_samples),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
