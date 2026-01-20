from unittest import mock
import torch

from llava.train.train import train
from llava.data import datasets_mixture
from llava.data.builder import build_dataset
from llava.train.transformer_normalize_monkey_patch import (
    _save_checkpoint,
    compute_loss,
    patched_normalize,
    training_step,
)
from llava.data.mol_dataset import MolDataset, DataCollatorForMolDataset

def make_mol_data_module(tokenizer, data_args, training_args):
    datasets_mixture.register_datasets_mixtures()
    if getattr(data_args, "data_mixture", None):
        train_dataset = build_dataset(data_args.data_mixture, data_args, training_args, tokenizer)
        if isinstance(train_dataset, torch.utils.data.ConcatDataset):
            training_args.sample_lens = [len(d) for d in train_dataset.datasets]
        else:
            training_args.sample_lens = [len(train_dataset)]
    else:
        train_dataset = MolDataset(data_path=data_args.data_path, tokenizer=tokenizer, data_args=data_args)
        training_args.sample_lens = [len(train_dataset)]
    data_collator = DataCollatorForMolDataset(tokenizer=tokenizer)
    training_args.eval_sample_lens = []
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def patch_deepspeed_unused_param():
    try:
        import deepspeed.runtime.utils as ds_utils
        import deepspeed.runtime.zero.stage3 as ds_stage3
    except Exception:
        return

    if not hasattr(ds_utils, "count_used_parameters_in_backward"):
        return

    orig = ds_utils.count_used_parameters_in_backward

    def count_used_parameters_in_backward(params, *args, **kwargs):
        # params = [p for p in params if getattr(p, "requires_grad", False)]
        valid_params = []
        for p in params:
            if getattr(p, "requires_grad", False):
                # Check if the parameter is attached to the graph to avoid 'NoneType' object has no attribute 'next_functions'
                if p.view_as(p).grad_fn is not None:
                    valid_params.append(p)
        return orig(valid_params, *args, **kwargs)

    ds_utils.count_used_parameters_in_backward = count_used_parameters_in_backward
    ds_stage3.count_used_parameters_in_backward = count_used_parameters_in_backward



if __name__ == "__main__":
    import llava.train.train as train_mod
    import llava.data.dataset as dataset_mod

    def patched_make_supervised_data_module(tokenizer, data_args, training_args):
        return make_mol_data_module(tokenizer, data_args, training_args)

    train_mod.make_supervised_data_module = patched_make_supervised_data_module
    dataset_mod.make_supervised_data_module = patched_make_supervised_data_module

    patch_deepspeed_unused_param()

    with (
        mock.patch("transformers.image_processing_utils.normalize", new=patched_normalize),
        mock.patch("accelerate.data_loader.BatchSamplerShard.__len__", new=lambda self: len(self.batch_sampler)),
        mock.patch("accelerate.data_loader.BatchSamplerShard.__iter__", new=lambda self: self.batch_sampler.__iter__()),
        mock.patch("transformers.trainer.Trainer._save_checkpoint", new=_save_checkpoint),
        mock.patch("transformers.trainer.Trainer.compute_loss", new=compute_loss),
        mock.patch("transformers.trainer.Trainer.training_step", new=training_step),
    ):
        train()
