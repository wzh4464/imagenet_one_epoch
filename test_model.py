###
# File: ./test_model.py
# Created Date: Tuesday, December 3rd 2024
# Author: Zihan
# -----
# Last Modified: Tuesday, 3rd December 2024 11:48:18 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###


import os
import torch
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
import argparse
from main import (
    setup_logger,
    replace_vit_attention,
    setup,
    cleanup,
)
from torchvision.datasets import ImageFolder


def test_model(rank, world_size, root_dir, output_dir, batch_size=32, device="cuda:0"):
    """
    Test the model on the first category of the dataset.

    Args:
        root_dir: The root directory of the ImageNet dataset.
        output_dir: Directory to save results.
        batch_size: Batch size for inference.
        device: Device to use for inference.
    """
    logger = setup_logger(rank)
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    # Prepare the transform for data preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create the dataset instance
    dataset = ImageFolder(root=root_dir, transform=transform)
    logger.info(f"Total samples in dataset: {len(dataset)}")

    # 获取前两类的类别ID
    first_two_category_ids = dataset.classes[:2]
    logger.info(f"前两类类别ID: {first_two_category_ids}")
    first_two_category_idxs = [dataset.class_to_idx[cls_id] for cls_id in first_two_category_ids]
    logger.info(f"前两类类别索引: {first_two_category_idxs}")
    
    # 筛选出前两类的所有样本索引
    indices = [
        idx
        for idx, target in enumerate(dataset.targets)
        if target in first_two_category_idxs
    ]
    logger.info(f"前两类的样本数量: {len(indices)}")
    
    # 为每个类别选择96个样本
    selected_indices = []
    for category_idx in first_two_category_idxs:
        category_indices = [idx for idx in indices if dataset.targets[idx] == category_idx]
        selected = category_indices[:96]
        selected_indices.extend(selected)
        logger.info(f"从类别索引 {category_idx} 中选择了 {len(selected)} 个样本")
    
    # 创建测试数据集
    test_dataset = Subset(dataset, selected_indices)
    logger.info(f"测试数据集中的总样本数量: {len(test_dataset)}")

    # Prepare the dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    assert len(test_loader) > 0, "No samples found in the first category!"

    # Load the pre-trained Vision Transformer model
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    # model = replace_vit_attention(model).to(device)
    model = DDP(model, device_ids=[rank])
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")

    # Perform inference
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            logger.info(f"Batch {batch_idx}: Correct Predictions = {(predicted == targets).sum().item()}")
            logger.info(f"Predicted: {predicted.cpu().numpy()}, Targets: {targets.cpu().numpy()}")

    # Log results
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Accuracy on the first category: {accuracy:.4f}")
    print(f"Accuracy on the first category: {accuracy:.4f}")

    # Save results to a file
    results_file = os.path.join(output_dir, "test_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Accuracy on the first category: {accuracy:.4f}\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Correct predictions: {correct}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model on ImageNet dataset.")
    parser.add_argument(
        "--backup-dir", type=str, default="/backup", help="backup directory path"
    )
    parser.add_argument(
        "--imagenet-dir", type=str, default="imagenet", help="imagenet directory name"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results", help="output directory path"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="batch size for testing"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device for inference"
    )

    args = parser.parse_args()

    # Construct the full path to the dataset
    ROOT_DIR = os.path.join(
        args.backup_dir, args.imagenet_dir, "ILSVRC/Data/CLS-LOC/train"
    )

    # Check if dataset exists
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory {ROOT_DIR} does not exist!")
        exit(1)

    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the distributed environment
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("DDP requires at least 2 GPUs!")

    mp.spawn(

        test_model,
        args=(world_size, ROOT_DIR, args.output_dir, args.batch_size),
        nprocs=world_size,
        join=True,
    )

    cleanup()
