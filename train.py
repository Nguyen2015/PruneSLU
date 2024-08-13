#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# You can also adapt this script for your own distillation tasks. Pointers for this are left as comments.

import logging
import os
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from utils.reader import CustomDataset
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from models.modeling_whisper import MultiTaskWhisperModel

import datasets
import evaluate
import numpy as np
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import (
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
)
from huggingface_hub import create_repo, get_full_repo_name, upload_folder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AddedToken,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizerFast,
    get_scheduler,
    set_seed,
    WhisperForConditionalGeneration
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import torch.nn.functional as F

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0.dev0")

require_version("datasets>=2.14.6", "To fix: `pip install --upgrade datasets`")

logger = get_logger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to distill from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained Whisper model or model identifier from huggingface.co/models"}
    )
    teacher_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained Whisper model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    feature_extractor_name: Optional[str] = field(
        default=None,
        metadata={"help": "feature extractor name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    subfolder: str = field(
        default="",
        metadata={
            "help": "In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can"
            "specify the folder name here."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use in the encoder and decoder attention layers. Can be one of:\n"
                "1. `eager` or `None`: default Transformers attention implementation.\n"
                "2. `sdpa`: Flash Attention through PyTorch SDPA. Requires `torch>=2.1`. Recommended for hardware where Flash Attention 2 is not supported, e.g. Turing GPUs, (T4, RTX 2080).\n"
                "3. `flash_attn_2`: Flash Attention 2 through the Flash Attention package https://github.com/Dao-AILab/flash-attention. **Always** recommended on supported hardware (Ampere, Ada, or Hopper GPUs, e.g., A100, RTX 3090, RTX 4090, H100)."
            )
        },
    )

    def __post_init__(self):
        if self.attn_implementation not in [None, "eager", "sdpa", "flash_attention_2"]:
            raise ValueError(
                f"Got `--attn_implementation={self.attn_implementation}`, which is an invalid attention type. Should be one of:\n"
                "1. `eager` or `None`: default Transformers attention implementation.\n"
                "2. `sdpa`: Flash Attention through PyTorch SDPA. Requires `torch>=2.1`. Recommended for hardware where Flash Attention 2 is not supported, e.g. Turing GPUs, (T4, RTX 2080).\n"
                "3. `flash_attn_2`: Flash Attention 2 through the Flash Attention package https://github.com/Dao-AILab/flash-attention. **Always** recommended on supported hardware (Ampere, Ada, or Hopper GPUs, e.g., A100, RTX 3090, RTX 4090, H100)."
            )

@dataclass
class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    freeze_encoder: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to freeze the entire encoder model. Only recommended when the entire encoder has been "
                "copied from the teacher model."
            )
        },
    )
    freeze_encoder_staring_step: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "The step at which to start freezing the encoder. This is useful for gradual unfreezing of the encoder."
            )
        },
    )
    freeze_embed_positions: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze the decoder embedding positions."},
    )
    temperature: Optional[float] = field(
        default=2.0, metadata={"help": "Temperature to anneal the logits when computing the softmax."}
    )
    kl_weight: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Weighting assigned to the MSE loss in the KD formulation. MSE loss is "
                "computed between the teacher-student hidden states and attentions."
            )
        },
    )
    layer_kl_weight: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Weighting assigned to the MSE loss in the KD formulation. MSE loss is "
                "computed between the teacher-student hidden states and attentions."
            )
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "The data type (dtype) in which to run training. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision)."
            )
        },
    )
    decoder_layers_numbers: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "Layers numbers of the decoder teacher to use in the student model. Defaults to None, equivalent to taking first and last layer (and equivalent to `--decoder_layers_numbers 0 -1`)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data: str = field(
        default=None,
        metadata={
            "help": "The name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset ids by a '+' symbol. For example, to load LibriSpeech "
            "and Common Voice, set `train_dataset_name='librispeech_asr+common_voice'`."
        },
    )
    train_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the training dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset configs by a '+' symbol. Note that the order of the configs should "
            "match the order of the datasets."
        },
    )
    train_dataset_samples: str = field(
        default=None,
        metadata={
            "help": "Number of samples in each dataset when loading multiple datasets with streaming mode. "
            "Not required when using one dataset or non-streaming mode. The sample values provide the sampling "
            "probability for each dataset. Setting them equal to the number of sample values ensures that every "
            "sample from every dataset is used once per epoch."
        },
    )
    eval_data: str = field(
        default=None,
        metadata={
            "help": "The name of the evaluation dataset to use (via the datasets library). Defaults to the training "
            "dataset name if unspecified. Load multiple evaluation datasets by separating dataset "
            "ids by a '+' symbol."
        },
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the evaluation dataset to use (via the datasets library). Defaults to the "
            "training dataset config name if unspecified."
        },
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing if using non-streaming mode."},
    )
    preprocessing_batch_size: Optional[int] = field(
        default=256,
        metadata={"help": "Number of examples per batch provided to the `prepare_dataset` function."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the text data in the training set."},
    )
    eval_text_column_name: str = field(
        default="text",
        metadata={"help": ("The name of the dataset column containing the text data in the evaluation set.")},
    )
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"},
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"},
    )
    max_label_length: int = field(
        default=448,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
    pad_target_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set will pad the target sequence to a multiple of the provided"
                " value. This is important to avoid triggering recompilations on TPU."
                " If unspecified, will default to padding the targets to max length."
            )
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is"
                " especially useful when data preprocessing errors out in distributed"
                " training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with"
                " `preprocessing_only=True` so that the cached datasets can"
                " consequently be loaded in distributed training"
            )
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": (
                "The name of the evaluation data set split to use (via the datasets library). Defaults to 'validation'"
            )
        },
    )
    streaming: bool = field(
        default=True,
        metadata={"help": "Whether to use Datasets' streaming mode to load and pre-process the data."},
    )
    wer_threshold: float = field(
        default=None,
        metadata={
            "help": "Filter training data with Whisper transcriptions that have greater than `wer_threshold` "
            "WER with the normalised transcriptions. This only takes effect if training on pseudo-labels targets."
            "If `--use_pseudo_labels=False`, then no WER filtering is performed, since we train directly on the text"
            "transcriptions."
        },
    )
    use_pseudo_labels: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use pseudo-label transcriptions as the targets. If True, the pseudo-labels "
            "must be in the dataset column `whisper_transcript` from the previous pseudo-labelling step. This is "
            "not currently yet configurable."
        },
    )
    condition_on_prev_probability: float = field(
        default=0.2, metadata={"help": "Probability for conditioning on the previous text example."}
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual distillation. This argument should be set for multilingual distillation "
                "only. For English speech recognition, it should be left as `None`."
            )
        },
    )
    task: str = field(
        default=None,
        metadata={
            "help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."
            "This argument should be set for multilingual distillation only. For English speech recognition, it should be left as `None`."
        },
    )
    wandb_project: str = field(
        default="wandb_tracker",
        metadata={"help": "The name of the wandb project."},
    )
    min_audio_len: float = field(
        default=0.5,
        metadata={"help": "Minimum audio length, in seconds"},
    )
    max_audio_len: float = field(
        default=30,
        metadata={"help": "Maximum audio length, in seconds"},
    )
    timestamps: bool = field(
        default=False,
        metadata={"help": "Whether to use timestamp data during training"},
    )


def log_metric(
    accelerator,
    metrics: Dict,
    train_time: float,
    step: int,
    epoch: int,
    learning_rate: float = None,
    prefix: str = "train",
):
    """Helper function to log all training/evaluation metrics with the correct prefixes and styling."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    log_metrics[f"{prefix}/epoch"] = epoch
    if learning_rate is not None:
        log_metrics[f"{prefix}/learning_rate"] = learning_rate
    accelerator.log(log_metrics, step=step)


def log_pred(
    accelerator,
    pred_str: List[str],
    label_str: List[str],
    norm_pred_str: List[str],
    norm_label_str: List[str],
    step: int,
    prefix: str = "eval",
    num_lines: int = 200000,
):
    """Helper function to log target/predicted transcriptions to weights and biases (wandb)."""
    if accelerator.is_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        # pretty name for current step: step 50000 -> step 50k
        cur_step_pretty = f"{int(step // 1000)}k" if step > 1000 else step
        prefix_pretty = prefix.replace("/", "-")

        # convert str data to a wandb compatible format
        str_data = [[label_str[i], pred_str[i], norm_label_str[i], norm_pred_str[i]] for i in range(len(pred_str))]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"predictions/{prefix_pretty}-step-{cur_step_pretty}",
            columns=["Target", "Pred", "Norm Target", "Norm Pred"],
            data=str_data[:num_lines],
            step=step,
        )

        # log incorrect normalised predictions
        str_data = np.asarray(str_data)
        str_data_incorrect = str_data[str_data[:, -2] != str_data[:, -1]]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"incorrect_predictions/{prefix_pretty}-step-{cur_step_pretty}",
            columns=["Target", "Pred", "Norm Target", "Norm Pred"],
            data=str_data_incorrect[:num_lines],
            step=step,
        )


def convert_dataset_str_to_list(
    dataset_names,
    dataset_config_names,
    splits=None,
    text_column_names=None,
    dataset_samples=None,
    default_split="train",
) -> List[Dict]:
    """
    Given three lists of dataset names, configs and splits, this function groups the corresponding
    names/configs/splits. Each dataset is assigned a unique dictionary with these metadata values, and the
    function returns a list of dictionaries, one for each dataset.
    """
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split("+")
        dataset_config_names = dataset_config_names.split("+") if dataset_config_names is not None else None
        splits = splits.split("+") if splits is not None else None
        text_column_names = text_column_names.split("+") if text_column_names is not None else None
        dataset_samples = dataset_samples.split("+") if dataset_samples is not None else None

    # basic checks to ensure we've got the right number of datasets/configs/splits/columns/probs
    if dataset_config_names is not None and len(dataset_names) != len(dataset_config_names):
        raise ValueError(
            f"Ensure one config is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(dataset_config_names)} configs."
        )

    if splits is not None and len(splits) != len(dataset_names):
        raise ValueError(
            f"Ensure one split is passed for each dataset, got {len(dataset_names)} datasets and {len(splits)} splits."
        )

    if text_column_names is not None and len(text_column_names) != len(dataset_names):
        raise ValueError(
            f"Ensure one text column name is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(text_column_names)} text column names."
        )

    if dataset_samples is not None:
        if len(dataset_samples) != len(dataset_names):
            raise ValueError(
                f"Ensure one sample is passed for each dataset, got {len(dataset_names)} datasets and "
                f"{len(dataset_samples)} samples."
            )
        dataset_samples = [float(ds_sample) for ds_sample in dataset_samples]
    else:
        dataset_samples = [None] * len(dataset_names)

    dataset_config_names = (
        dataset_config_names if dataset_config_names is not None else ["default" for _ in range(len(dataset_names))]
    )
    text_column_names = (
        text_column_names if text_column_names is not None else ["text" for _ in range(len(dataset_names))]
    )
    splits = splits if splits is not None else [default_split for _ in range(len(dataset_names))]

    dataset_names_dict = []
    for i, ds_name in enumerate(dataset_names):
        dataset_names_dict.append(
            {
                "name": ds_name,
                "config": dataset_config_names[i],
                "split": splits[i],
                "text_column_name": text_column_names[i],
                "samples": dataset_samples[i],
            }
        )
    return dataset_names_dict



def sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint") -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(save_total_limit=None, output_dir=None, checkpoint_prefix="checkpoint") -> None:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(output_dir=output_dir, checkpoint_prefix=checkpoint_prefix)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint, ignore_errors=True)


_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)-epoch-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _RE_CHECKPOINT.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0])))


def get_parameter_names(model, forbidden_layer_types, forbidden_module=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer or forbidden module.
    Can be used to get a subset of parameter names for decay masks, or to exclude parameters from an optimiser
    (e.g. if the module is frozen).
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types, forbidden_module)
            if not (
                isinstance(child, tuple(forbidden_layer_types))
                or (child in tuple(forbidden_module) if forbidden_module is not None else False)
            )
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def main():
    # 1. Parse input arguments
    # We keep distinct sets of args, for cleaner separation of model/data/training related args
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DistillationTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Initialize the accelerator
    # We will let the accelerator handle device placement for us in this example
    # We simply have to specify the training precision and any trackers being used
    # We'll use the same dtype arguments as our JAX/Flax training script and convert
    # it to accelerate format
    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        teacher_dtype = torch.float16
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        teacher_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        teacher_dtype = torch.float32

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
    )

    accelerator.init_trackers(project_name=data_args.wandb_project)

    # 3. Set-up basic logging
    # Create one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    logger.info("Training/evaluation parameters %s", training_args)

    # 4. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 5. Handle the repository creation
    if accelerator.is_main_process:
        if training_args.push_to_hub:
            if training_args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(training_args.output_dir).absolute().name,
                    token=training_args.hub_token,
                )
            else:
                repo_name = training_args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=training_args.hub_token)

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "wandb" not in gitignore:
                    gitignore.write("wandb\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


    # set seed for determinism
    set_seed(training_args.seed)
    if not training_args.do_train and not training_args.do_eval:
        raise ValueError(
            "Cannot not train and not do evaluation. At least one of training or evaluation has to be performed."
        )

    # 7. Load pretrained model, tokenizer, and feature extractor
    config = WhisperConfig.from_pretrained(
        (model_args.config_name if model_args.config_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        (model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        (model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
    )

    teacher_model = WhisperForConditionalGeneration.from_pretrained(
        model_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        low_cpu_mem_usage=True,
        # torch_dtype=teacher_dtype,
        attn_implementation=model_args.attn_implementation,
    ).cuda()

    # update the params
    model = MultiTaskWhisperModel(model_args=model_args, config=config)

    # Set embedding values for new tokens
    new_tokens = json.load(open("data/labels.json"))
    input_embeddings = model.whisper.get_input_embeddings().weight.data
    new_token_embeddings = torch.zeros(len(new_tokens), input_embeddings.size(1))
    for idx, token in enumerate(new_tokens):
        old_token_ids = tokenizer.encode(token)[2:-1]
        new_embedding = input_embeddings[old_token_ids].mean(dim=0, keepdim=True)
        new_token_embeddings[idx] = new_embedding
    
    # add new tokens
    # num_new_tokens = tokenizer.add_tokens(list(new_tokens))

    # # Resize the token embeddings in the model and Set embedding values for new tokens
    # model.whisper.resize_token_embeddings(len(tokenizer))
    # model.whisper.get_input_embeddings().weight.data[-num_new_tokens:] = new_token_embeddings


    if model.whisper.config.decoder_start_token_id is None:
        raise ValueError(
            f"Make sure that `config.decoder_start_token_id` is correctly defined for the "
            f"student model. Got {model.config.decoder_start_token_id} "
        )

    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        model.whisper.gradient_checkpointing_enable()

    def set_trainable_parameters(module, requires_grad=False):
        for param in module.parameters():
            param.requires_grad = requires_grad
        module._requires_grad = requires_grad

    if training_args.freeze_embed_positions:
        # set_trainable_parameters(model.model.decoder.embed_tokens, requires_grad=False)
        set_trainable_parameters(model.whisper.model.decoder.embed_positions, requires_grad=False)
        if model.whisper.model.decoder.gradient_checkpointing:
            logger.info(
                "Disabling gradient checkpointing in the decoder since it's incompatible with `freeze_embed_positions`."
            )
        is_multilingual = False
    elif data_args.language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )
    else:
        is_multilingual = False

    # 8. Create a single speech processor - make sure all processes wait until data is saved
    if accelerator.is_main_process:
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        # save the config and generation config as well
        config.save_pretrained(training_args.output_dir)
        model.whisper.generation_config.save_pretrained(training_args.output_dir)

    accelerator.wait_for_everyone()
    processor = WhisperProcessor.from_pretrained(training_args.output_dir)


    # 10. Preprocessing the datasets: we need to read the audio files as arrays and tokenize the targets.
    # 10.1: Define the pre-processing constants
    max_label_length = (
        data_args.max_label_length if data_args.max_label_length is not None else model.config.max_length
    )

    dataloader_num_workers = training_args.dataloader_num_workers
    prefetch_factor = training_args.dataloader_prefetch_factor

    if training_args.do_train:
        with accelerator.main_process_first():
            # Read data
            train_dataset = CustomDataset(data_list_path=data_args.train_data,
                                        processor=processor,
                                        language=data_args.language,
                                        timestamps=data_args.timestamps,
                                        min_duration=data_args.min_audio_len,
                                        max_duration=data_args.max_audio_len,
                                        augment_config_path="configs/augmentation.json")

            print(f"Training data: {len(train_dataset)}")
            # Data padding


    if training_args.do_eval:
        with accelerator.main_process_first():
            eval_dataset = CustomDataset(data_list_path=data_args.eval_data,
                            processor=processor,
                            language=data_args.language,
                            timestamps=data_args.timestamps,
                            min_duration=data_args.min_audio_len,
                            max_duration=data_args.max_audio_len)
            print(f"Eval data: {len(eval_dataset)}")


    # 12. Define Training Schedule
    # Store some constants
    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    num_epochs = int(training_args.num_train_epochs)
    steps_per_epoch = len(train_dataset) // (train_batch_size * gradient_accumulation_steps)
    total_train_steps = steps_per_epoch * num_epochs

    if training_args.eval_steps is None:
        logger.info(
            f"eval_steps is not set, evaluating at the end of {'each epoch' if not data_args.streaming else 'training'}"
        )
        eval_steps = steps_per_epoch
    else:
        eval_steps = training_args.eval_steps

    # 13. Define optimizer, LR scheduler, collator
    if training_args.freeze_encoder and training_args.freeze_encoder_staring_step == 0:
        set_trainable_parameters(model.whisper.model.encoder, requires_grad=False)
        model.whisper.model.encoder.gradient_checkpointing = False

    decay_parameters = get_parameter_names(
        model,
        [nn.LayerNorm],
        forbidden_module=[model.whisper.model.encoder] if training_args.freeze_encoder else None,
    )
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [param for name, param in model.named_parameters() if name in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [param for name, param in model.named_parameters() if name not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]


    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    # LR scheduler gets stepped by `num_processes` each time -> account for this in warmup / total steps
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # 14. Define generation arguments - we need to do this before we wrap the models in DDP
    # so that we can still access the configs
    num_beams = (
        training_args.generation_num_beams
        if training_args.generation_num_beams is not None
        else getattr(model.whisper.generation_config, "num_beams", 1)
    )

    gen_kwargs = {
        "max_length": max_label_length,
        "num_beams": num_beams
    }
    if is_multilingual:
        # forcing the language and task tokens helps multilingual models in their generations
        gen_kwargs.update(
            {
                "language": data_args.language,
                "task": data_args.task,
            }
        )

    # 15. Prepare everything with accelerate
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    def kl_divergence(target_distribution, log_predicted_distribution, labels):
        kl_loss = nn.KLDivLoss(reduction="none")
        divergence = kl_loss(log_predicted_distribution, target_distribution)
        # ignore padded tokens from divergence, i.e. where labels are not set to -100
        padding_mask = labels >= 0
        padding_mask = padding_mask.unsqueeze(-1)
        divergence = divergence * padding_mask
        # take the average over the mini-batch
        divergence = divergence.sum() / padding_mask.sum()
        return divergence

    def contrastive_loss(student_hidden_states, teacher_hidden_states, temperature=1.0):
        final_loss = 0
        mapped_teacher_hidden_states = [teacher_hidden_states[i] for i in training_args.decoder_layers_numbers]
        for student_layer_output, teacher_layer_output in zip(student_hidden_states, mapped_teacher_hidden_states):
            # Normalize the outputs
            teacher_norm = F.normalize(teacher_layer_output, dim=-1)
            student_norm = F.normalize(student_layer_output, dim=-1)
            # Compute cosine similarity
            similarity_matrix = torch.matmul(student_norm, teacher_norm.transpose(-1, -2)) / temperature
            
            # Create labels for contrastive loss (assuming positive pairs are on the diagonal)
            batch_size, sequence_length, _ = teacher_layer_output.size()
            labels = torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1).flatten()
            
            # Flatten the similarity matrix and labels
            similarity_matrix = similarity_matrix.flatten(end_dim=1)
            labels = labels.to(similarity_matrix.device)
            
            # Compute the contrastive loss using cross-entropy
            loss = F.cross_entropy(similarity_matrix, labels)

            final_loss += loss
        final_loss /= len(student_hidden_states)
        return final_loss


    # Define gradient update step fn
    def train_step(
        batch,
        temperature=2.0,
    ):
        model.train()
        teacher_model.eval()

        student_outputs = model(batch, output_hidden_states=True)
        student_layer_outputs = student_outputs.decoder_hidden_states[1:]
        with torch.no_grad():
            # do the full forward pass for the teacher model (encoder + decoder)
            teacher_outputs = teacher_model(input_features=batch["input_features"], labels=batch["labels"], output_hidden_states=True)
            teacher_layer_outputs = teacher_outputs.decoder_hidden_states[1:]

        mse_loss = contrastive_loss(student_layer_outputs, teacher_layer_outputs)
        # CE (data) loss
        ce_loss = student_outputs.loss
        # rescale distribution by temperature to ensure gradients scale correctly
        teacher_distribution = nn.functional.softmax(teacher_outputs.logits / temperature, dim=-1)
        # log softmax of student predictions for numerical stability
        student_distribution = nn.functional.log_softmax(student_outputs.logits / temperature, dim=-1)
        # KL-divergence loss (scaled by temperature)
        kl_loss = kl_divergence(teacher_distribution, student_distribution, batch["labels"]) * temperature**2

        # print("==============")
        # print(student_outputs)
        # print(teacher_outputs)
        # print(f"CE Loss: {ce_loss}, KL Loss: {kl_loss}")

        # use Distil-Whisper formulation (fix weight of CE loss and tune KL weight)
        loss = 0.8 * ce_loss + training_args.kl_weight * kl_loss + training_args.layer_kl_weight * mse_loss
        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}
        return loss, metrics

    # Define eval fn
    def eval_step(batch):
        model.eval()
        teacher_model.eval()

        with torch.no_grad():
            student_outputs = model(batch, output_hidden_states=True)
            student_layer_outputs = student_outputs.decoder_hidden_states[1:]
            teacher_outputs = teacher_model(input_features=batch["input_features"], labels=batch["labels"], output_hidden_states=True)
            teacher_layer_outputs = teacher_outputs.decoder_hidden_states[1:]

        mse_loss = contrastive_loss(student_layer_outputs, teacher_layer_outputs)
        # CE (data) loss
        ce_loss = student_outputs.loss

        # log softmax / softmax for numerical stability
        student_distribution = nn.functional.log_softmax(student_outputs.logits, dim=-1)
        teacher_distribution = nn.functional.softmax(teacher_outputs.logits, dim=-1)
        # temperature is always 1 for eval
        kl_loss = kl_divergence(teacher_distribution, student_distribution, batch["labels"])

        # use Distil-Whisper formulation (fix weight of CE loss and tune KL weight)
        loss = 0.8 * ce_loss + training_args.kl_weight * kl_loss + training_args.layer_kl_weight * mse_loss
        metrics = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}
        return metrics


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {total_train_steps * train_batch_size * gradient_accumulation_steps}")
    if not data_args.streaming:
        logger.info(f"  Num epochs = {num_epochs}")
    logger.info("  Instantaneous batch size per device =" f" {training_args.per_device_train_batch_size}")
    logger.info("  Gradient accumulation steps =" f" {gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")
    logger.info(f"  Learning rate = {training_args.learning_rate}")

    # for item in tqdm(train_dataloader):
    #     print(item["match_labels"].sum())
    #     break

    # ======================== Training ================================
    train_time = 0
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
    )
    continue_training = True
    epochs_trained = 0
    cur_step = 0

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        # Find num steps and epoch from saved state string pattern
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {cur_step}")

        steps_trained_progress_bar.update(cur_step)

        if not data_args.streaming and training_args.max_steps < 0:
            # we know exactly the number of steps per epoch, so can skip through the required number of batches
            resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
        else:
            # Currently we don't know how many steps we've taken in the current epoch
            # So we just shuffle the dataset one extra time and start from a fresh epoch
            # This is "good enough" for our purposes but not fully correct
            resume_step = None
    else:
        resume_step = None

    best_eval_los = float("inf")

    for epoch in range(epochs_trained, num_epochs):
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size,
            num_workers=dataloader_num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=training_args.dataloader_pin_memory,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)

        if resume_step is not None:
            # Skip the first N batches in the dataloader when resuming from a checkpoint
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = None

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                loss, train_metric = train_step(batch, training_args.temperature)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                steps_trained_progress_bar.update(1)
                cur_step += 1

                if cur_step % training_args.logging_steps == 0:
                    steps_trained_progress_bar.write(
                        f"Step... ({cur_step} / {total_train_steps} | Loss:"
                        f" {train_metric['loss']}, Learning Rate:"
                        f" {lr_scheduler.get_last_lr()[0]})"
                    )
                    log_metric(
                        accelerator,
                        metrics=train_metric,
                        learning_rate=lr_scheduler.get_last_lr()[0],
                        train_time=train_time + time.time() - train_start,
                        step=cur_step,
                        epoch=epoch,
                        prefix="train",
                    )

                # save checkpoint and weights after each save_steps and at the end of training
                if training_args.freeze_encoder and cur_step == training_args.freeze_encoder_staring_step:
                    set_trainable_parameters(model.whisper.model.encoder, requires_grad=False)
                    model.whisper.model.encoder.gradient_checkpointing = False
                    logger.info("Freezing encoder weights at step %d", cur_step)

                if (cur_step % training_args.save_steps == 0) or cur_step == total_train_steps:
                    intermediate_dir = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                    model.whisper.save_pretrained(intermediate_dir)
                    # accelerator.save_state(output_dir=intermediate_dir)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        rotate_checkpoints(training_args.save_total_limit, output_dir=training_args.output_dir)

                        if cur_step == total_train_steps:
                            # un-wrap student model for save
                            model = accelerator.unwrap_model(model)
                            # re-wrap student model for final eval
                            model = accelerator.prepare(model)

                        if training_args.push_to_hub:
                            upload_folder(
                                folder_path=training_args.output_dir,
                                repo_id=repo_name,
                                repo_type="model",
                                commit_message=f"Saving train state of step {cur_step}",
                            )

                if training_args.do_eval and (cur_step % eval_steps == 0 or cur_step == total_train_steps):
                    train_time += time.time() - train_start
                    model.eval()
                    # ======================== Evaluating ==============================
                    eval_metrics = []
                    eval_start = time.time()

                    validation_dataloader = DataLoader(
                        eval_dataset,
                        collate_fn=data_collator,
                        batch_size=per_device_eval_batch_size,
                        drop_last=False,
                        num_workers=dataloader_num_workers,
                        prefetch_factor=prefetch_factor,
                        pin_memory=training_args.dataloader_pin_memory,
                    )
                    validation_dataloader = accelerator.prepare(validation_dataloader)

                    for batch in tqdm(
                        validation_dataloader,
                        desc=f"Evaluating ...",
                        position=2,
                        disable=not accelerator.is_local_main_process,
                    ):
                        # Model forward
                        eval_metric = eval_step(batch)
                        eval_metric = accelerator.gather_for_metrics(eval_metric)
                        eval_metrics.append(eval_metric)


                    eval_time = time.time() - eval_start
                    # normalize eval metrics
                    eval_metrics = {
                        key: torch.mean(torch.stack([d[key] for d in eval_metrics])) for key in eval_metrics[0]
                    }

                    # Print metrics and update progress bar
                    steps_trained_progress_bar.write(
                        f"Eval results for step ({cur_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']}"
                    )

                    log_metric(
                        accelerator,
                        metrics=eval_metrics,
                        train_time=eval_time,
                        step=cur_step,
                        epoch=epoch,
                        prefix="eval",
                    )

                    # flush the train metrics
                    train_start = time.time()

                    # log predictions
                    if eval_metrics["loss"] < best_eval_los:
                        best_eval_los = eval_metrics["loss"]
                        if accelerator.is_main_process:
                            model.whisper.save_pretrained(training_args.output_dir)
                            logger.info(f"New best model saved at step {cur_step}")


                # break condition
                if cur_step == total_train_steps:
                    continue_training = False
                    break

        if not continue_training:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main()