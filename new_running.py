import gc
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, concatenate_datasets
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
import logging
import os
from datetime import datetime
from transformers import TrainerCallback
import torch.nn as nn
import bitsandbytes as bnb
from typing import Optional, Dict, Union, Any, Generator
from torch.utils.data import DataLoader
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
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

class BatchProcessingTrainer(CustomTrainer):
    def train_epoch_with_batches(self, batch_generator: Generator):
        self.model.train()
        for batch_dataset in batch_generator:
            self.train_dataset = batch_dataset
            super().train()
            gc.collect()
            torch.cuda.empty_cache()


class CustomCallback(TrainerCallback):
    def __init__(self, trainer, logger):
        self.trainer = trainer
        self.logger = logger
        self.best_loss = float('inf')
        self.steps_without_improvement = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info(f"Step {state.global_step}: {logs}")


class BidirectionalTranslationTrainer:
    def __init__(self, model_name="numind/NuExtract-1.5-smol", batch_size=50000):
        self.model_name = model_name
        self.batch_size = batch_size
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

        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"trainable params: {trainable_params} || all params: {total_params} || trainable%: {100 * trainable_params / total_params:.4f}")

    def batch_data_generator(self, data_dir: str) -> Generator:
        train_files = glob.glob(os.path.join(data_dir, "de-en", "train-*.parquet"))

        for file in train_files:
            df = pd.read_parquet(file)
            total_rows = len(df)

            for start in range(0, total_rows, self.batch_size):
                chunk = df.iloc[start:min(start + self.batch_size, total_rows)]
                translations = pd.json_normalize(chunk['translation'])

                combined = pd.concat([
                    pd.DataFrame({
                        'text': translations['de'],
                        'translation': translations['en'],
                        'direction': 'de-en'
                    }),
                    pd.DataFrame({
                        'text': translations['en'],
                        'translation': translations['de'],
                        'direction': 'en-de'
                    })
                ])

                dataset = Dataset.from_pandas(combined)
                processed_dataset = self._process_dataset(dataset)

                yield processed_dataset

                del chunk, translations, combined, dataset
                gc.collect()
                torch.cuda.empty_cache()

            del df
            gc.collect()

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        def preprocess_function(examples):
            prompts = [
                f"{dir.split('-')[0].upper()}: {src}\n{dir.split('-')[1].upper()}: {tgt}</s>"
                for src, tgt, dir in zip(examples["text"], examples["translation"], examples["direction"])
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

        processed = dataset.map(
            preprocess_function,
            batched=True,
            batch_size=1000,
            remove_columns=dataset.column_names,
            desc="Processing batch"
        )
        processed.set_format("torch")
        return processed

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
        return bnb.optim.AdamW8bit(
            optimizer_grouped_parameters,
            **optimizer_kwargs
        )

    def train(self, data_dir: str, resume_from: Optional[str] = "auto"):
        training_args = TrainingArguments(
            output_dir=os.path.join(self.checkpoint_base, "checkpoints"),
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            learning_rate=5e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            weight_decay=0.05,
            bf16=True,
            logging_dir=os.path.join(self.checkpoint_base, "logs"),
            logging_steps=50,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            warmup_ratio=0.05,
            remove_unused_columns=False,
            label_smoothing_factor=0.1,
            optim="paged_adamw_8bit",
            ddp_find_unused_parameters=False,
            dataloader_num_workers=4,
            group_by_length=True,
            report_to="none"  # Disable wandb logging
        )

        # Load validation dataset once
        val_df = pd.read_parquet(os.path.join(data_dir, "de-en", "validation-00000-of-00001.parquet"))
        val_translations = pd.json_normalize(val_df['translation'])
        eval_dataset = self._process_dataset(Dataset.from_pandas(pd.concat([
            pd.DataFrame({'text': val_translations['de'], 'translation': val_translations['en'], 'direction': 'de-en'}),
            pd.DataFrame({'text': val_translations['en'], 'translation': val_translations['de'], 'direction': 'en-de'})
        ])))

        trainer = BatchProcessingTrainer(
            model=self.model,
            args=training_args,
            eval_dataset=eval_dataset,
            callbacks=[CustomCallback(self, logger)],
            optimizers=(self._create_optimizer(training_args), None)
        )

        try:
            for epoch in range(2):
                logger.info(f"Starting epoch {epoch + 1}")
                batch_generator = self.batch_data_generator(data_dir)
                trainer.train_epoch_with_batches(batch_generator)
                self.save_model(os.path.join(self.checkpoint_base, f"epoch_{epoch + 1}"))
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            self.save_model(os.path.join(self.checkpoint_base, "emergency_save"))
            raise

    def save_model(self, output_dir):
        logger.info(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def translate(self, text: str, direction: str = "en-de"):
        self.model.eval()
        src_lang = direction.split('-')[0].upper()
        tgt_lang = direction.split('-')[1].upper()
        prompt = f"{src_lang}: {text}\n{tgt_lang}:"

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
        return translation.split(f"{tgt_lang}:")[-1].strip()


if __name__ == "__main__":
    MODEL_NAME = "numind/NuExtract-1.5-smol"
    DATA_DIR = "wmt14"

    trainer = BidirectionalTranslationTrainer(model_name=MODEL_NAME, batch_size=5000)
    trainer.train(data_dir=DATA_DIR)