import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
import logging
import os
from datetime import datetime
from transformers import TrainerCallback
import torch.nn as nn
import bitsandbytes as bnb
from typing import Optional, Dict, Union, Any
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class CausalLanguageModelingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch_factor = 2

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=self.prefetch_factor
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=self.prefetch_factor
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = inputs["input_ids"].clone()

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch

        loss.backward()

        return loss.detach()


class CustomCallback(TrainerCallback):
    def __init__(self, trainer, logger):
        self.trainer = trainer
        self.logger = logger
        self.best_loss = float('inf')
        self.steps_without_improvement = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info(f"Step {state.global_step}: {logs}")


class TranslationTrainer:
    def __init__(self, model_name="numind/NuExtract-1.5-smol"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        torch.cuda.empty_cache()

        self.checkpoint_base = os.path.join("checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.checkpoint_base, exist_ok=True)
        self.setup_model_and_tokenizer()

    def setup_model_and_tokenizer(self):
        logger.info(f"Initializing model {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right",
            model_max_length=48
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Quantization config
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        # LoRA configuration
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=None
        )

        self.model = get_peft_model(self.model, peft_config)

        # Set up trainable parameters
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"trainable params: {trainable_params} || all params: {total_params} || trainable%: {100 * trainable_params / total_params:.4f}")

    def prepare_dataset(self, file_path):
        logger.info("Preparing dataset...")
        df = pd.read_parquet(file_path)

        # Log dataset size
        logger.info(f"Total number of examples in dataset: {len(df)}")

        # Split into train and evaluation sets
        train_df, eval_df = train_test_split(df, test_size=0.03, random_state=42)
        logger.info(f"Training set size: {len(train_df)}")
        logger.info(f"Evaluation set size: {len(eval_df)}")

        self.train_dataset = Dataset.from_pandas(train_df)
        self.eval_dataset = Dataset.from_pandas(eval_df)

        def preprocess_function(examples):
            prompts = [
                f"EN: {src}\nDE: {tgt}</s>"
                for src, tgt in zip(examples["text"], examples["translation"])
            ]

            model_inputs = self.tokenizer(
                prompts,
                max_length=48,
                truncation=True,
                padding="max_length",
                return_tensors=None
            )

            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs

        # Process datasets without multiprocessing
        self.tokenized_train = self.train_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=1000,
            remove_columns=self.train_dataset.column_names,
            desc="Processing training data",
            num_proc=None  # Disabled multiprocessing
        )

        self.tokenized_eval = self.eval_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=1000,
            remove_columns=self.eval_dataset.column_names,
            desc="Processing validation data",
            num_proc=None  # Disabled multiprocessing
        )

        self.tokenized_train.set_format("torch")
        self.tokenized_eval.set_format("torch")

        logger.info(f"Processed {len(self.tokenized_train)} training examples")
        logger.info(f"Processed {len(self.tokenized_eval)} evaluation examples")

    def train(self, resume_from=None):
        training_args = TrainingArguments(
            output_dir=os.path.join(self.checkpoint_base, "checkpoints"),
            eval_strategy="steps",
            eval_steps=500,  # Erhöht von 200
            save_strategy="steps",
            save_steps=500,  # Erhöht von 200
            save_total_limit=3,
            learning_rate=5e-4,  # Erhöht von 2e-4
            per_device_train_batch_size=4,  # Erhöht von 2
            per_device_eval_batch_size=4,  # Erhöht von 2
            gradient_accumulation_steps=8,  # Reduziert von 16
            num_train_epochs=2,
            weight_decay=0.05,  # Erhöht von 0.01
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            bf16=True,
            logging_dir=os.path.join(self.checkpoint_base, "logs"),
            logging_steps=50,  # Erhöht von 10
            report_to="tensorboard",
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            warmup_ratio=0.05,  # Erhöht von 0.03
            remove_unused_columns=False,
            label_smoothing_factor=0.1,
            optim="paged_adamw_8bit",
            ddp_find_unused_parameters=False,
            dataloader_num_workers=4,
            group_by_length=True
        )

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_eval,
            callbacks=[CustomCallback(self, logger)],
            optimizers=(self._create_optimizer(training_args), None)
        )

        try:
            logger.info("Starting training...")
            trainer.train(resume_from_checkpoint=resume_from)
            self.save_model(os.path.join(self.checkpoint_base, "final_model"))
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            self.save_model(os.path.join(self.checkpoint_base, "emergency_save"))
            raise

    def _create_optimizer(self, training_args):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)
                          and p.requires_grad],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)
                          and p.requires_grad],
                "weight_decay": 0.0,
            }
        ]

        optimizer_kwargs = {
            "lr": training_args.learning_rate,
            "betas": (0.9, 0.95),
            "eps": 1e-8
        }
        optimizer = bnb.optim.AdamW8bit(
            optimizer_grouped_parameters,
            **optimizer_kwargs
        )
        return optimizer

    def save_model(self, output_dir):
        logger.info(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def translate(self, text):
        self.model.eval()
        prompt = f"EN: {text}\nDE:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=48,
                temperature=0.5,
                do_sample=True,
                top_p=0.95,
                num_beams=5,
                length_penalty=0.6,
                no_repeat_ngram_size=3
            )

        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation.split("DE:")[-1].strip()


if __name__ == "__main__":
    MODEL_NAME = "numind/NuExtract-1.5-smol"
    DATASET_PATH = "train-00000-of-00001.parquet"

    # Um komplett neu zu starten (ohne Checkpoint):
    trainer = TranslationTrainer(model_name=MODEL_NAME)
    trainer.prepare_dataset(DATASET_PATH)
    trainer.train(resume_from=None)  # None bedeutet Neustart

    # ODER wenn du den Checkpoint laden willst:
    # 1. Prüfe den exakten Pfad:
    import os

    CHECKPOINT_DIR = os.path.join(os.getcwd(), "20241117_172820/checkpoints/checkpoint-400")

    # 2. Überprüfe ob der Pfad existiert:
    if os.path.exists(CHECKPOINT_DIR):
        trainer = TranslationTrainer(model_name=MODEL_NAME)
        trainer.prepare_dataset(DATASET_PATH)
        trainer.train(resume_from=CHECKPOINT_DIR)
    else:
        print(f"Checkpoint nicht gefunden unter: {CHECKPOINT_DIR}")
        print("Starte neues Training...")
        trainer = TranslationTrainer(model_name=MODEL_NAME)
        trainer.prepare_dataset(DATASET_PATH)
        trainer.train(resume_from=None)


