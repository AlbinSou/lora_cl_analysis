#!/usr/bin/env python3

import os
from typing import Any, Optional, Sequence, Union

from avalanche.benchmarks import benchmark_with_validation_stream
from avalanche.benchmarks.classic import (
    SplitCUB200,
    SplitImageNet,
)
from avalanche.benchmarks.datasets import CUB200, ImageNet
from avalanche.benchmarks.generators import ni_benchmark
from avalanche.benchmarks.scenarios.generic_scenario_creation import (
    create_multi_dataset_generic_scenario,
)
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import FGVCAircraft, StanfordCars

from src.factories.default_transforms import *
from src.factories.flowers import FlowersDataset

"""
Benchmarks factory
"""

DS_SIZES = {
    "split_imagenet": (224, 224, 3),
}

DS_CLASSES = {
    "split_imagenet": 1000,
}


def process_cars(dataset):
    """
    Adds targets field to dataset
    to provide avalanche compatibility
    """
    targets = []
    for s, l in dataset._samples:
        targets.append(l)
    dataset.targets = targets
    return dataset


def create_benchmark(
    benchmark_name: str,
    n_experiences: int,
    *,
    val_size: float = 0,
    seed: Optional[int] = None,
    dataset_root: Union[str] = None,
    first_exp_with_half_classes: bool = False,
    return_task_id=False,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = True,
    class_ids_from_zero_in_each_exp: bool = False,
    class_ids_from_zero_from_first_exp: bool = False,
    use_transforms: bool = True,
    override_transforms=None,
):
    benchmark = None

    if benchmark_name == "split_imagenet":
        if not use_transforms:
            train_transform = default_imagenet_eval_transform
            eval_transform = train_transform
        else:
            train_transform = default_imagenet_train_transform
            eval_transform = default_imagenet_eval_transform

        if override_transforms is not None:
            train_transform, eval_transform = override_transforms

        benchmark = SplitImageNet(
            n_experiences=n_experiences,
            return_task_id=return_task_id,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
            class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=os.path.join(dataset_root, "imagenet"),
        )

    elif benchmark_name == "cub200":
        if not use_transforms:
            train_transform = default_cub200_eval_transform
            eval_transform = train_transform
        else:
            train_transform = default_cub200_train_transform
            eval_transform = default_cub200_eval_transform

        # Override transforms
        if override_transforms is not None:
            train_transform, eval_transform = override_transforms

        benchmark = SplitCUB200(
            n_experiences,
            return_task_id=return_task_id,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
            class_ids_from_zero_from_first_exp=class_ids_from_zero_from_first_exp,
            train_transform=train_transform,
            eval_transform=eval_transform,
            dataset_root=dataset_root,
        )

    elif benchmark_name == "stanford_cars":
        if not use_transforms:
            train_transform = default_imagenet_train_transform
            eval_transform = train_transform
        else:
            train_transform = default_imagenet_train_transform
            eval_transform = default_imagenet_eval_transform

        # Override transforms
        if override_transforms is not None:
            train_transform, eval_transform = override_transforms

        train_cars = process_cars(StanfordCars(dataset_root, split="train"))
        test_cars = process_cars(StanfordCars(dataset_root, split="test"))

        benchmark = create_multi_dataset_generic_scenario(
            [train_cars],
            [test_cars],
            task_labels=[0],
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    elif benchmark_name == "imnet_then_finetune":
        """
        ImageNet -> StanfordCars -> Flowers -> Aircraft -> CUB

        For now
        """

        if not use_transforms:
            train_transform = default_imagenet_train_transform
            eval_transform = train_transform
        else:
            train_transform = default_imagenet_train_transform
            eval_transform = default_imagenet_eval_transform

        # Override transforms
        if override_transforms is not None:
            train_transform, eval_transform = override_transforms

        train_imnet = ImageNet(os.path.join(dataset_root, "imagenet"), split="train")
        test_imnet = ImageNet(os.path.join(dataset_root, "imagenet"), split="val")

        train_cars = process_cars(StanfordCars(dataset_root, split="train"))
        test_cars = process_cars(StanfordCars(dataset_root, split="test"))

        train_flowers = FlowersDataset(dataset_root, split="train")
        test_flowers = FlowersDataset(dataset_root, split="test")

        train_aircraft = FGVCAircraft(dataset_root, split="train")
        train_aircraft.targets = train_aircraft._labels
        test_aircraft = FGVCAircraft(dataset_root, split="test")
        test_aircraft.targets = test_aircraft._labels

        train_birds = CUB200(dataset_root, train=True)
        test_birds = CUB200(dataset_root, train=False)

        benchmark = create_multi_dataset_generic_scenario(
            [train_imnet, train_cars, train_flowers, train_aircraft, train_birds],
            [test_imnet, test_cars, test_flowers, test_aircraft, test_birds],
            task_labels=[0, 1, 2, 3, 4],
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    elif benchmark_name == "imnet_then_finetune_2":
        """
        ImageNet -> Aircraft -> CUB -> Flowers -> Stanford Cars
        """

        if not use_transforms:
            train_transform = default_imagenet_train_transform
            eval_transform = train_transform
        else:
            train_transform = default_imagenet_train_transform
            eval_transform = default_imagenet_eval_transform

        # Override transforms
        if override_transforms is not None:
            train_transform, eval_transform = override_transforms

        train_imnet = ImageNet(os.path.join(dataset_root, "imagenet"), split="train")
        test_imnet = ImageNet(os.path.join(dataset_root, "imagenet"), split="val")

        train_cars = process_cars(StanfordCars(dataset_root, split="train"))
        test_cars = process_cars(StanfordCars(dataset_root, split="test"))

        train_flowers = FlowersDataset(dataset_root, split="train")
        test_flowers = FlowersDataset(dataset_root, split="test")

        train_aircraft = FGVCAircraft(dataset_root, split="train")
        train_aircraft.targets = train_aircraft._labels
        test_aircraft = FGVCAircraft(dataset_root, split="test")
        test_aircraft.targets = test_aircraft._labels

        train_birds = CUB200(dataset_root, train=True)
        test_birds = CUB200(dataset_root, train=False)

        benchmark = create_multi_dataset_generic_scenario(
            [train_imnet, train_aircraft, train_birds, train_flowers, train_cars],
            [test_imnet, test_aircraft, test_birds, test_flowers, test_cars],
            task_labels=[0, 1, 2, 3, 4],
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    elif benchmark_name == "imnet_then_finetune_long":
        """
        ImageNet -> StanfordCars/2 -> Flowers/2 -> Aircraft/2 -> CUB/2
                 -> StanfordCars/2 -> Flowers/2 -> Aircraft/2 -> CUB/2

        Same than above but do it 2 times, with half the data each time
        """

        if not use_transforms:
            train_transform = default_imagenet_train_transform
            eval_transform = train_transform
        else:
            train_transform = default_imagenet_train_transform
            eval_transform = default_imagenet_eval_transform

        # Override transforms
        if override_transforms is not None:
            train_transform, eval_transform = override_transforms

        train_imnet = ImageNet(os.path.join(dataset_root, "imagenet"), split="train")
        test_imnet = ImageNet(os.path.join(dataset_root, "imagenet"), split="val")

        train_cars = process_cars(StanfordCars(dataset_root, split="train"))
        test_cars = process_cars(StanfordCars(dataset_root, split="test"))

        train_flowers = FlowersDataset(dataset_root, split="train")
        test_flowers = FlowersDataset(dataset_root, split="test")

        train_aircraft = FGVCAircraft(dataset_root, split="train")
        train_aircraft.targets = train_aircraft._labels
        test_aircraft = FGVCAircraft(dataset_root, split="test")
        test_aircraft.targets = test_aircraft._labels

        train_birds = CUB200(dataset_root, train=True)
        test_birds = CUB200(dataset_root, train=False)

        train_ds_list = {
            "cars": train_cars,
            "flowers": train_flowers,
            "aircraft": train_aircraft,
            "birds": train_birds,
        }

        # Stores splitted ds
        new_train_ds_list = {}

        # Split ds
        for name in ["cars", "flowers", "aircraft", "birds"]:
            dataset = train_ds_list[name]
            indexes = torch.randperm(len(dataset))
            indexes_s1 = indexes[: len(dataset) // 2]
            indexes_s2 = indexes[len(dataset) // 2 :]
            new_train_ds_list[name] = [
                Subset(dataset, indexes_s1),
                Subset(dataset, indexes_s2),
            ]

        # Create stream
        list_training_ds = []
        for name in [
            "cars",
            "flowers",
            "aircraft",
            "birds",
            "cars",
            "flowers",
            "aircraft",
            "birds",
        ]:
            list_training_ds.append(new_train_ds_list[name].pop())

        benchmark = create_multi_dataset_generic_scenario(
            [train_imnet] + list_training_ds,
            [
                test_imnet,
                test_cars,
                test_flowers,
                test_aircraft,
                test_birds,
                test_cars,
                test_flowers,
                test_aircraft,
                test_birds,
            ],
            task_labels=[0, 1, 2, 3, 4, 1, 2, 3, 4],
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    elif benchmark_name == "imnet_then_finetune_long_80_20":
        """
        ImageNet -> StanfordCars (80%) -> Flowers (80%) -> Aircraft (80%) -> CUB (80%)
                 -> StanfordCars (20%) -> Flowers (20%) -> Aircraft (20%) -> CUB (20%)

        Same than above but do it 2 times, with half the data each time
        """

        if not use_transforms:
            train_transform = default_imagenet_train_transform
            eval_transform = train_transform
        else:
            train_transform = default_imagenet_train_transform
            eval_transform = default_imagenet_eval_transform

        # Override transforms
        if override_transforms is not None:
            train_transform, eval_transform = override_transforms

        train_imnet = ImageNet(os.path.join(dataset_root, "imagenet"), split="train")
        test_imnet = ImageNet(os.path.join(dataset_root, "imagenet"), split="val")

        train_cars = process_cars(StanfordCars(dataset_root, split="train"))
        test_cars = process_cars(StanfordCars(dataset_root, split="test"))

        train_flowers = FlowersDataset(dataset_root, split="train")
        test_flowers = FlowersDataset(dataset_root, split="test")

        train_aircraft = FGVCAircraft(dataset_root, split="train")
        train_aircraft.targets = train_aircraft._labels
        test_aircraft = FGVCAircraft(dataset_root, split="test")
        test_aircraft.targets = test_aircraft._labels

        train_birds = CUB200(dataset_root, train=True)
        test_birds = CUB200(dataset_root, train=False)

        train_ds_list = {
            "cars": train_cars,
            "flowers": train_flowers,
            "aircraft": train_aircraft,
            "birds": train_birds,
        }

        # Stores splitted ds
        new_train_ds_list = {}

        # Split ds
        for name in ["cars", "flowers", "aircraft", "birds"]:
            dataset = train_ds_list[name]
            indexes = torch.randperm(len(dataset))
            indexes_s1 = indexes[int(len(dataset)*0.8):]
            indexes_s2 = indexes[:int(len(dataset)*0.8)]
            new_train_ds_list[name] = [
                Subset(dataset, indexes_s1),
                Subset(dataset, indexes_s2),
            ]

        # Create stream
        list_training_ds = []
        for name in [
            "cars",
            "flowers",
            "aircraft",
            "birds",
            "cars",
            "flowers",
            "aircraft",
            "birds",
        ]:
            list_training_ds.append(new_train_ds_list[name].pop())

        benchmark = create_multi_dataset_generic_scenario(
            [train_imnet] + list_training_ds,
            [
                test_imnet,
                test_cars,
                test_flowers,
                test_aircraft,
                test_birds,
                test_cars,
                test_flowers,
                test_aircraft,
                test_birds,
            ],
            task_labels=[0, 1, 2, 3, 4, 1, 2, 3, 4],
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    assert benchmark is not None
    if hasattr(benchmark, "classes_order_original_ids"):
        print(benchmark.classes_order_original_ids)

    if val_size > 0:
        benchmark = benchmark_with_validation_stream(
            benchmark, validation_size=val_size, shuffle=True
        )

    return benchmark
