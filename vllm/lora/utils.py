import os
from typing import List, Optional, Set, Tuple, Type

import huggingface_hub
from huggingface_hub.utils import (EntryNotFoundError, HfHubHTTPError,
                                   HFValidationError, RepositoryNotFoundError)
from torch import nn
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.fully_sharded_layers import (
    ColumnParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLora, QKVParallelLinearWithShardedLora,
    RowParallelLinearWithShardedLoRA)
# being imported for _all_lora_classes below
# yapf conflicts with isort for this block
# yapf: disable
from vllm.lora.layers import (BaseLayerWithLoRA, ColumnParallelLinearWithLoRA,
                              LinearScalingRotaryEmbeddingWithLora,
                              LogitsProcessorWithLoRA,
                              MergedColumnParallelLinearWithLoRA,
                              MergedQKVParallelLinearWithLora,
                              QKVParallelLinearWithLora,
                              ReplicatedLinearWithLoRA,
                              RowParallelLinearWithLoRA,
                              VocabParallelEmbeddingWithLoRA)
# yapf: enable
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.lora.s3 import S3_PREFIX, download_s3_model

logger = init_logger(__name__)

_all_lora_classes: Set[Type[BaseLayerWithLoRA]] = {
    VocabParallelEmbeddingWithLoRA,
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLora,
    MergedQKVParallelLinearWithLora,
    RowParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    LogitsProcessorWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
    QKVParallelLinearWithShardedLora,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLora,
    RowParallelLinearWithShardedLoRA,
    LinearScalingRotaryEmbeddingWithLora,
}


def from_layer(layer: nn.Module,
               max_loras: int,
               lora_config: LoRAConfig,
               packed_modules_list: List,
               model_config: Optional[PretrainedConfig] = None) -> nn.Module:
    for lora_cls in _all_lora_classes:
        # specifying kwargs so they can be easily accessed in decorator
        if lora_cls.can_replace_layer(source_layer=layer,
                                      lora_config=lora_config,
                                      packed_modules_list=packed_modules_list,
                                      model_config=model_config):
            ret = lora_cls(layer)
            ret.create_lora_weights(max_loras, lora_config, model_config)
            return ret
    return layer


def from_layer_logits_processor(
    layer: LogitsProcessor,
    lm_head: ParallelLMHead,
    max_loras: int,
    lora_config: LoRAConfig,
    model_config: Optional[PretrainedConfig] = None,
) -> LogitsProcessorWithLoRA:
    ret = LogitsProcessorWithLoRA(layer, lm_head.embedding_dim,
                                  lm_head.weight.dtype, lm_head.weight.device,
                                  lm_head.get_sharded_to_full_mapping())
    ret.create_lora_weights(max_loras, lora_config, model_config)
    return ret


def replace_submodule(model: nn.Module, module_name: str,
                      new_module: nn.Module) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module


def parse_fine_tuned_lora_name(name: str) -> Tuple[str, bool]:
    """Parse the name of lora weights.

    args:
        name: the name of the fine-tuned LoRA, e.g.
            base_model.model.dense1.weight
    return:
        Tuple(module_name, is_lora_a):
            module_name: the name of the module, e.g. model.dense1,
            is_lora_a whether the tensor is lora_a or lora_b.
    """
    parts = name.split(".")

    if len(parts) >= 2 and parts[0] == "base_model" and parts[1] == "model":
        if parts[-1] == "weight":
            if parts[-2] == "lora_A" or parts[-2] == "lora_B":
                return ".".join(parts[2:-2]), parts[-2] == "lora_A"
        elif parts[-1] == "lora_embedding_A" or parts[-1] == "lora_embedding_B":
            return ".".join(parts[2:-1]), parts[-1] == "lora_embedding_A"

    raise ValueError(f"{name} is unsupported LoRA weight")


def get_adapter_absolute_path(lora_path: str) -> str:
    """
    Resolves the given lora_path to an absolute local path.
    Supports local paths, Hugging Face model identifiers, and S3 paths.
    For S3 paths, it checks if the adapter is already downloaded to 's3_adapters'.

    Parameters:
    lora_path (str): The path to the lora model, which can be an absolute path,
                     a relative path, a Hugging Face model identifier, or an S3 path.

    Returns:
    str: The resolved absolute local path to the lora model.
    """
    # Check if the path is an S3 path
    if lora_path.startswith(S3_PREFIX):
        return resolve_s3_path(lora_path)

    # Handle local paths and Hugging Face identifiers
    return resolve_local_or_hf_path(lora_path)

def resolve_s3_path(lora_path: str) -> str:
    s3_model_path = download_s3_model(lora_path)
    return s3_model_path

def resolve_local_or_hf_path(lora_path: str) -> str:
    # Check if the path is an absolute path
    if os.path.isabs(lora_path):
        return lora_path

    # If the path starts with ~, expand the user home directory.
    if lora_path.startswith('~'):
        return os.path.expanduser(lora_path)

    # Check if the expanded relative path exists locally.
    if os.path.exists(lora_path):
        return os.path.abspath(lora_path)

    # If the path does not exist locally, assume it's a Hugging Face repo.
    try:
        local_snapshot_path = huggingface_hub.snapshot_download(repo_id=lora_path)
        logger.info(f"Downloaded adapter from Hugging Face: {local_snapshot_path}")
        return local_snapshot_path
    except (HfHubHTTPError, RepositoryNotFoundError, EntryNotFoundError, HFValidationError):
        logger.exception("Error downloading the HuggingFace model")
        return None