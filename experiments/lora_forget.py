#!/usr/bin/env python3
import argparse
import copy
import math
import os
import re

import hydra
import jsonlines
import omegaconf
import peft
import timm
import torch
import torch.nn as nn
from avalanche.logging import TensorboardLogger
from avalanche.models import MultiHeadClassifier, MultiTaskModule
from avalanche.training.plugins import LRSchedulerPlugin, LwFPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from transformers.optimization import get_cosine_schedule_with_warmup

from src.factories.benchmark_factory import create_benchmark
from src.toolkit.json_logger import JSONLogger
from src.toolkit.utils import set_seed


"""
Computes the forgetting of ViT on
Imagenet-1k for different ranks
of LoRAs on various downstream tasks
"""


def freeze_modules(model_type, model):
    """
    Freeze modules that are not matching the model pattern
    """
    if model_type == "vit":
        target_modules = r"blocks\..*\.attn\.qkv.weight"
    elif model_type == "cnn":
        # target_modules = r"layer.*\..*\.conv.*"
        # Next line is as in "Towards Practical Plug-and-Play Diffusion Models"
        # target_modules = r"layer.*\..*\.conv[1-2]"
        target_modules = r"layer.*\..*\.conv3.weight"
    else:
        raise ValueError(f"Unknown model type {model_type}")

    regexp = re.compile(target_modules)

    unfrozen = []
    for n, p in model.named_parameters():
        # Freeze all params not matching pattern
        if not regexp.match(n):
            p.requires_grad = False
            p.grad = None

        else:
            p.requires_grad = True
            unfrozen.append(p)

    num_unfrozen = sum([p.numel() for p in unfrozen])
    print(f"Full finetune of {num_unfrozen} parameters")
    return


def create_lora_config(model_type, lora_rank, lora_alpha):
    if lora_rank > 0:
        if model_type == "vit":
            target_modules = r"blocks\..*\.attn\.qkv"
        elif model_type == "cnn":
            # target_modules = r"layer.*\..*\.conv.*"
            # Next line is as in "Towards Practical Plug-and-Play Diffusion Models"
            # target_modules = r"layer.*\..*\.conv[1-2]"
            target_modules = r"layer.*\..*\.conv3"
        else:
            raise ValueError(f"Unknown model type {model_type}")

        lora_config = peft.LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
        )
    else:
        lora_config = None

    return lora_config


class NaivePeft(SupervisedTemplate):
    def __init__(
        self,
        model,
        optimizer,
        criterion=nn.CrossEntropyLoss(),
        lora_config=None,
        mode="merge",
        train_mb_size=1,
        train_epochs=1,
        eval_mb_size=1,
        device="cuda",
        plugins=None,
        evaluator=default_evaluator(),
        eval_every=-1,
        **base_kwargs,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs,
        )
        assert mode in ["merge", "drop", "full", "full_reset"]

        # Merge : Setup LoRA for each task and merge it to pretrained weights after training task

        # Drop  : Setup LoRA for each task and drop it after training task

        # Full  : Freeze modules except qkv (vit) and conv3 (resnet), finetune them with full rank

        # TODO Split in 3 different plugins implementing these behaviours

        # Freeze existing imagenet head
        self.freeze_head()

        self.lora_config = lora_config
        self.mode = mode

        if self.mode == "full_reset":
            self.initial_model = copy.deepcopy(model)

    def _after_training_exp(self, **kwargs):
        self.model.save(self.logdir, self.experience.current_experience)

        # Merge previous peft
        if self.model.is_peft:
            if self.mode == "merge":
                self.model.merge_peft()

        # Freeze BN stats
        # We have to do that here to so that saved LwF model is also in eval mode
        if self.model.model_type == "cnn":
            self.model.backbone.eval()

        super()._after_training_exp(**kwargs)

    def _before_training_exp(self, **kwargs):
        # Unload previous peft
        if self.model.is_peft and self.mode == "drop":
            self.model.unload_peft()

        if self.mode in ["full", "full_reset"]:
            if self.mode == "full_reset":
                self.model = copy.deepcopy(self.initial_model)

            freeze_modules(self.model.model_type, self.model.backbone)

        elif self.mode in ["merge", "drop"]:
            # Load Peft for new task
            if self.lora_config is not None:
                self.model.set_peft_model(self.lora_config)
            else:
                # Freeze full net
                self.model.freeze_backbone()

        # Freeze BN stats
        if self.model.model_type == "cnn":
            self.model.backbone.eval()

        super()._before_training_exp(**kwargs)

    def freeze_head(self):
        for p in self.model.head.parameters():
            p.requires_grad = False
            p.grad = None

    def unfreeze_head(self):
        for p in self.model.head.parameters():
            p.requires_grad = True


DEBUG = False


class MultiClassModel(MultiTaskModule):
    def __init__(self, model, head_name, model_type):
        super().__init__()
        old_head = getattr(model, head_name)
        setattr(model, head_name, nn.Identity())
        self.backbone = model
        self.head = MultiHeadClassifier(
            in_features=old_head.in_features,
            masking=False,
        )

        # Set parameters to previous head values
        self.head.classifiers["0"].classifier = old_head
        self.model_type = model_type
        self.is_peft = False

    def forward_single_task(self, x, task_label):
        repr = self.backbone(x)
        return self.head(repr, task_label)

    def set_peft_model(self, lora_config):
        peft_model = peft.get_peft_model(self.backbone, lora_config)
        peft_model.print_trainable_parameters()
        self.backbone = peft_model
        trainable_parameters = list(peft_model.parameters()) + list(
            self.head.parameters()
        )
        self.is_peft = True
        return trainable_parameters

    def merge_peft(self):
        self.backbone = self.backbone.merge_and_unload()
        self.is_peft = False

    def unload_peft(self):
        self.backbone = self.backbone.unload()
        self.is_peft = False

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
            p.grad = None

    def save(self, path, task):
        if self.is_peft:
            self.backbone.save_pretrained(os.path.join(path, f"lora_{task}"))
        # Just save the full head and override everytime
        torch.save(self.head, os.path.join(path, "head.ckpt"))


@hydra.main(version_base=None, config_path="../config", config_name="lora_config.yaml")
def main(config):
    global DEBUG
    DEBUG = config.experiment.debug

    results_dir = os.path.join(
        config.experiment.results_root, config.experiment.name, str(config.experiment.seed)
    )

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    omegaconf.OmegaConf.save(config, os.path.join(results_dir, "config.yaml"))

    set_seed(config.experiment.seed)

    model_id = config.model.model_id

    model = timm.create_model(model_id, pretrained=True, num_classes=1000)
    data_config = timm.data.resolve_model_data_config(model)
    train_transforms = timm.data.create_transform(**data_config, is_training=True)
    eval_transforms = timm.data.create_transform(**data_config, is_training=False)

    if config.benchmark.factory_args.use_transforms:
        transforms = (train_transforms, eval_transforms)
    else:
        transforms = (eval_transforms, eval_transforms)

    head_name = "head" if config.model.model_type == "vit" else "fc"

    model = MultiClassModel(model, head_name, config.model.model_type)

    # Avalanche: Create Scenario and strategy

    scenario = create_benchmark(
        config.benchmark.factory_args.benchmark_name,
        n_experiences=1,
        shuffle=False,
        dataset_root=config.benchmark.dataset_root,
        override_transforms=transforms,
    )

    # Prepare the LoRA

    lora_config = create_lora_config(
        config.model.model_type, config.strategy.lora_rank, config.strategy.lora_alpha
    )

    # Add scheduler

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )

    # Scale number of epochs based on dataset size
    if DEBUG:
        num_epochs = [0] + [1] * (len(scenario.train_stream) - 1)
    else:
        if len(config.strategy.train_epochs) != len(scenario.train_stream):
            # Repeat num epochs because it's the long setting
            num_epochs = [
                config.strategy.train_epochs[0]
            ] + config.strategy.train_epochs[1:] * 2
            print(f"Long Setting with training epochs: {num_epochs}")
        else:
            num_epochs = config.strategy.train_epochs
            print(f"Short Setting with training epochs: {num_epochs}")

    # Create Eval plugin
    tb_logger = TensorboardLogger(results_dir)
    json_logger = JSONLogger(os.path.join(results_dir, "logs.json"), autoupdate=True)
    eval_plugin = default_evaluator()
    eval_plugin.loggers.append(tb_logger)
    eval_plugin.loggers.append(json_logger)

    batch_size = config.strategy.train_mb_size

    plugins = []

    assert config.strategy.name in ["lwf", "finetune"]

    # Add LwF
    if config.strategy.name == "lwf":
        print("Using LwF strategy")
        lwf_plugin = LwFPlugin(
            alpha=config.strategy.lwf_alpha, temperature=config.strategy.lwf_temperature
        )
        plugins.append(lwf_plugin)

    elif config.strategy.name == "finetune":
        print("Using Finetune strategy")

    strategy = NaivePeft(
        model,
        optimizer,
        criterion=nn.CrossEntropyLoss(),
        lora_config=lora_config,
        mode=config.strategy.mode,
        train_mb_size=batch_size,
        eval_mb_size=config.strategy.eval_mb_size,
        train_epochs=num_epochs,
        device=config.strategy.device,
        evaluator=eval_plugin,
        plugins=plugins,
    )

    # Inform strategy logdir
    strategy.logdir = results_dir

    # Train on second task
    scheduler_plugin = None

    for t, experience in enumerate(scenario.train_stream):
        # Set number of epochs and scheduler
        num_training_steps = (len(experience.dataset) // batch_size) * num_epochs[t]
        num_warmup_steps = num_training_steps // 18
        print(f"Training for {num_training_steps} steps")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Make sure we do not train on Imagenet
        if t == 0:
            assert num_epochs[t] == 0

        strategy.train_epochs = num_epochs[t]

        if scheduler_plugin is None:
            scheduler_plugin = LRSchedulerPlugin(
                scheduler, step_granularity="iteration"
            )
            strategy.plugins.append(scheduler_plugin)
        else:
            scheduler_plugin.scheduler = scheduler

        strategy.train(experience, num_workers=config.strategy.num_workers)

        if not DEBUG:
            strategy.eval(
                scenario.test_stream[: t + 1], num_workers=config.strategy.num_workers
            )


if __name__ == "__main__":
    main()
