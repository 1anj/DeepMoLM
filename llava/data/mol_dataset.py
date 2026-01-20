import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from llava.data.base import BaseDataset, local_load_or_hf_load
from llava.data.collate import DataCollator as LlavaDataCollator
from llava.media import Image
from llava.mm_utils import (
    dynamic_process_images_and_prompt,
    dynamic_s2_process_images_and_prompt,
    get_original_image_size,
    process_images,
)
from llava.remote_code.tokenizer_utils import preprocess_conversation
from llava.utils.logging import logger
from llava.utils.media import extract_media

__all__ = ["MolDataset", "DataCollatorForMolDataset"]

MOL_FP_PAD_VALUE = -1


class MolDataset(BaseDataset):
    """
    MolDataset for DeepMoLM.
    
    Inherits from BaseDataset to integrate with the dataset registry system.
    Supports molecule fingerprints (molecule_fp) and molecular images.
    """

    def __init__(
        self,
        data_path: str,
        media_dir: Optional[str] = None,
        split: str = "train",
        **kwargs
    ) -> None:
        """Initialize the dataset with optional image media."""
        super().__init__(**kwargs)
        
        self.data_path = Path(data_path)
        self.media_dir = media_dir
        self.split = split
        
        data_file = self.data_path if self.data_path.is_file() else self.data_path / f"{split}.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
        
        # Load instances using the standard loader
        self.instances = local_load_or_hf_load(str(data_file))
        logger.info(f"Loaded {len(self.instances)} instances from {data_file}")
        
        # Keep molecule_fps separate; BaseDataset is unaware of this field.
        self.molecule_fps = [inst.get("molecule_fp") for inst in self.instances]

    @staticmethod
    def normalize_text(value: Any, fallback: str) -> str:
        if value is None:
            return fallback
        text = str(value).strip()
        return text if text else fallback

    def build_prompt(self, instance: Dict[str, Any]) -> str:
        instruction = self.normalize_text(
            instance.get("instruction"), "Describe the input molecule."
        )
        return instruction

    def select_output(self, instance: Dict[str, Any]) -> str:
        for key in ("enriched_output", "output", "selfies", "smiles"):
            value = instance.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return "No description available."

    @staticmethod
    def to_molecule_fp_tensor(mol_fp: Any) -> torch.Tensor:
        if mol_fp is None:
            logger.warning(
                "molecule_fp is None! Returning all-padding tensor. "
                "This will cause the fusion projector to receive only padding values, "
                "leading to loss=0 and NaN gradients. "
                "Please check that your data files contain the 'molecule_fp' field."
            )
            return torch.full((1, 4), MOL_FP_PAD_VALUE, dtype=torch.long)
        mol_fp_tensor = torch.as_tensor(mol_fp, dtype=torch.long)
        if mol_fp_tensor.dim() == 1:
            mol_fp_tensor = mol_fp_tensor.unsqueeze(0)
        if mol_fp_tensor.shape[-1] < 4:
            pad_dim = 4 - mol_fp_tensor.shape[-1]
            mol_fp_tensor = F.pad(mol_fp_tensor, (0, pad_dim), value=MOL_FP_PAD_VALUE)
        return mol_fp_tensor

    def process_image_batch(self, images: List[Any]) -> List[Any]:
        """Process a list of images with the configured image processor."""
        processor = self.data_args.image_processor
        if getattr(processor, "name", None) == "sam_clip_processor":
            # SAM-CLIP returns dicts per image rather than tensors.
            return [processor.preprocess(img) for img in images]
        return process_images(images, processor, self.data_args)

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert a raw instance into a minimal conversation format."""
        prompt = self.build_prompt(instance)
        output = self.select_output(instance)

        user_value = prompt
        if "image" in instance and self.media_dir:
            img_path = os.path.join(self.media_dir, instance["image"])
            user_value = [Image(img_path), prompt]

        return [
            {"from": "human", "value": user_value},
            {"from": "gpt", "value": output},
        ]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a processed data item.
        
        Uses a custom image pipeline to handle sam_clip processors
        that return dicts instead of tensors.
        
        Args:
            index: Index of the item
            
        Returns:
            Dictionary containing tokenized data and molecule_fp
        """
        instance = self.instances[index]

        try:
            conversation = self.process(instance)
            media = extract_media(conversation, self.data_args)

            block_sizes = []
            if "image" in media:
                if self.enable_dynamic_res_s2:
                    processed_images, block_sizes = dynamic_s2_process_images_and_prompt(
                        media["image"], conversation[0]["value"], self.data_args
                    )
                elif self.enable_dynamic_res and self.data_args.image_aspect_ratio == "dynamic":
                    processed_images, processed_prompt = dynamic_process_images_and_prompt(
                        media["image"], conversation[0]["value"], self.data_args
                    )
                    conversation[0]["value"] = processed_prompt
                else:
                    processed_images = self.process_image_batch(media["image"])
                original_image_sizes = [get_original_image_size(img) for img in media["image"]]

            if "video" in media:
                if self.enable_dynamic_res_s2 and self.data_args.video_max_tiles > 1:
                    processed_images, block_sizes = dynamic_s2_process_images_and_prompt(
                        media["video"][0],
                        conversation[0]["value"],
                        self.data_args,
                    )
                    # For HighRes video training, we use <image> token instead of <vila/video>
                    conversation[0]["value"] = conversation[0]["value"].replace("<vila/video>", "")
                elif (
                    self.enable_dynamic_res
                    and self.data_args.image_aspect_ratio == "dynamic"
                    and self.data_args.video_max_tiles > 1
                ):
                    processed_images, processed_prompt = dynamic_process_images_and_prompt(
                        media["video"][0],
                        conversation[0]["value"],
                        self.data_args,
                        max_tiles=self.data_args.video_max_tiles,
                    )
                    # For HighRes video training, we use <image> token instead of <vila/video>
                    conversation[0]["value"] = processed_prompt.replace("<vila/video>", "")
                else:
                    processed_images = [self.process_image_batch(video) for video in media["video"]]

            # Prepare "input_ids" and "labels" for training
            if self.system_prompt is not None:
                assert not self.prepend_empty_system_prompt, (
                    "system_prompt and prepend_empty_system_prompt cannot both be set"
                )
                if isinstance(self.system_prompt, (list, tuple)):
                    sys_prompt = random.choice(self.system_prompt)
                else:
                    sys_prompt = self.system_prompt
                conversation = [{"from": "system", "value": sys_prompt}] + conversation

            data = preprocess_conversation(
                conversation, self.tokenizer, no_system_prompt=self.prepend_empty_system_prompt
            )

            if self.enable_dynamic_res_s2 and ("image" in media or "video" in media):
                data["block_sizes"] = block_sizes

            if "image" in media:
                data["image"] = processed_images
                data["original_image_sizes"] = original_image_sizes
            if "video" in media:
                if (
                    self.enable_dynamic_res_s2 or self.enable_dynamic_res
                ) and self.data_args.video_max_tiles > 1:
                    # HighRes video training
                    data["image"] = processed_images
                else:
                    data["video"] = processed_images

        except Exception as e:
            if not self.resample_on_failure:
                raise e
            else:
                logger.exception(f"Error processing instance '{instance}': '{e}'. Resampling.")
                return self.__getitem__(random.randint(0, len(self.instances) - 1))

        mol_fp = self.molecule_fps[index]
        data["molecule_fp"] = self.to_molecule_fp_tensor(mol_fp)

        return data


# ----------------- Collator ----------------- #

@dataclass
class DataCollatorForMolDataset:
    """
    Data collator for MolDataset.
    
    Handles batching of molecule fingerprints along with standard LLaVA data.
    """
    tokenizer: Any

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of instances.
        
        Args:
            instances: List of data dictionaries from MolDataset
            
        Returns:
            Batched dictionary with stacked tensors
        """
        # Extract molecule_fps before passing to LlavaDataCollator
        molecule_fps = [inst.pop("molecule_fp", None) for inst in instances]
        
        # Use standard LLaVA collator for the rest
        batch = LlavaDataCollator(self.tokenizer)(instances)
        
        # Pad and stack molecule_fp
        valid = [fp for fp in molecule_fps if fp is not None]
        if not valid:
            return batch

        max_len = max(fp.shape[0] for fp in valid)
        feat_dim = valid[0].shape[1]
        padded = []
        for fp in molecule_fps:
            if fp is None:
                padded.append(valid[0].new_full((max_len, feat_dim), MOL_FP_PAD_VALUE))
            else:
                pad = max_len - fp.shape[0]
                if pad > 0:
                    # Pad along the atom dimension (dim 0).
                    fp = F.pad(fp, (0, 0, 0, pad), value=MOL_FP_PAD_VALUE)
                padded.append(fp)
        batch["molecule_fp"] = torch.stack(padded)
        
        return batch
