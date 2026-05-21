import os
import torch
import random
import numpy as np
import warnings
import mlflow
from trl import GRPOConfig, GRPOTrainer

from scr.config import CFG
from scr.model import build_model_and_tokenizer
from scr.dataset import get_and_format_gsm8k_questions
from scr.rewards import (
    soft_correctness_reward_func,
    strict_format_reward_func,
    check_structure_reward_func
)

warnings.filterwarnings("ignore")


def seed_everything(seed: int):
    """Фиксирует все случайности для воспроизводимости результатов."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    os.environ["MLFLOW_EXPERIMENT_NAME"] = CFG.MLFLOW_EXPERIMENT_NAME
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    seed_everything(CFG.SEED)

    train_ds, test_ds = get_and_format_gsm8k_questions()

    model, tokenizer, peft_config = build_model_and_tokenizer()

    training_arguments = GRPOConfig(
        output_dir=CFG.SAVE_FINETUNED_MODEL_PATH,
        run_name=CFG.RUN_NAME,
        bf16=False,
        fp16=True,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        gradient_accumulation_steps=CFG.GRAD_ACC,
        beta=CFG.KL_BETA,
        learning_rate=CFG.LR,
        weight_decay=CFG.WEIGHT_DECAY,
        warmup_ratio=CFG.WARMUP_RATIO,
        lr_scheduler_type=CFG.LR_SCHEDULER,
        optim=CFG.OPTIMIZER,
        max_grad_norm=CFG.MAX_GRAD_NORM,
        max_steps=CFG.MAX_STEPS,
        num_generations=CFG.NUM_GENERATIONS,
        max_completion_length=CFG.MAX_COMPLETION_LENGTH,
        generation_batch_size=CFG.NUM_GENERATIONS,
        temperature=CFG.TEMPERATURE,
        save_strategy='steps',
        save_steps=CFG.SAVE_STEPS,
        save_total_limit=10,
        gradient_checkpointing=True,
        report_to='mlflow',
        logging_steps=CFG.LOG_STEPS,
        seed=CFG.SEED,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        peft_config=peft_config,
        args=training_arguments,
        reward_funcs=[
            soft_correctness_reward_func,
            strict_format_reward_func,
            check_structure_reward_func
        ],
        train_dataset=train_ds,
    )

    print("\n\n--- Starting GRPO Training --- \n\n")
    trainer.train()

    mlflow.end_run()
    print("Training Completed.")


if __name__ == "__main__":
    main()