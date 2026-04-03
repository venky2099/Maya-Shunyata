# split_cifar10.py — 5-task CIL dataloader, no task oracle exposed to model

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from maya_cl.utils.config import (
    DATA_DIR, BATCH_SIZE, NUM_TASKS, CLASSES_PER_TASK
)

# standard CIFAR-10 normalisation
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# 5 tasks x 2 classes — fixed canonical split
TASK_CLASSES = [
    [0, 1],   # Task 0: airplane, automobile
    [2, 3],   # Task 1: bird, cat
    [4, 5],   # Task 2: deer, dog
    [6, 7],   # Task 3: frog, horse
    [8, 9],   # Task 4: ship, truck
]


def _get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, test_tf


def _subset_by_classes(dataset, classes):
    # return indices where target is in the given class list
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)


def get_task_loaders(task_id: int):
    """
    Returns (train_loader, test_loader) for one task.
    Labels are the original 0-9 CIFAR-10 labels — CIL: model sees all 10 classes.
    task_id is NEVER passed to the model — only used here for data selection.
    """
    assert 0 <= task_id < NUM_TASKS, f"task_id must be 0–{NUM_TASKS - 1}"
    train_tf, test_tf = _get_transforms()

    train_full = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tf)
    test_full  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)

    classes = TASK_CLASSES[task_id]
    train_sub = _subset_by_classes(train_full, classes)
    test_sub  = _subset_by_classes(test_full,  classes)

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_sub,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


def get_all_test_loaders():
    """
    Returns list of 5 test loaders — one per task.
    Used by eval loop to compute AA and BWT after each task.
    """
    _, test_tf = _get_transforms()
    test_full = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    loaders = []
    for classes in TASK_CLASSES:
        sub = _subset_by_classes(test_full, classes)
        loaders.append(DataLoader(sub, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=2, pin_memory=True))
    return loaders