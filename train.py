import os
import torch
import wandb
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv

from src.config import CFG
from src.model import build_model_and_tokenizer
from src.dataset import get_gsm8k_dataset
from src.rewards import soft_correctness_reward_func, xml_format_reward_func, aha_moment_reward_func
from src.utils import seed_everything


def main():
    load_dotenv()
    seed_everything(CFG.SEED)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model, tokenizer = build_model_and_tokenizer()
    train_ds, _ = get_gsm8k_dataset()

    training_args = GRPOConfig(
        output_dir=CFG.OUTPUT_DIR,
        run_name=CFG.RUN_NAME,
        learning_rate=CFG.LR,
        per_device_train_batch_size=CFG.BATCH_SIZE,
        gradient_accumulation_steps=CFG.GRAD_ACC,
        max_steps=CFG.MAX_STEPS,
        num_generations=CFG.NUM_GENERATIONS,
        max_prompt_length=CFG.MAX_PROMPT_LENGTH,
        max_completion_length=CFG.MAX_COMPLETION_LENGTH,
        beta=CFG.KL_BETA,
        temperature=CFG.TEMPERATURE,
        optim=CFG.OPTIMIZER if hasattr(CFG, 'OPTIMIZER') else "adamw_8bit",
        save_strategy="steps",
        save_steps=CFG.SAVE_STEPS,
        logging_steps=CFG.LOG_STEPS,
        report_to="wandb",
        fp16=True,
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=[
            soft_correctness_reward_func,
            xml_format_reward_func,
            aha_moment_reward_func
        ],
        train_dataset=train_ds,
    )

    print("--- Starting GRPO Training ---")
    trainer.train()
    trainer.save_model(CFG.OUTPUT_DIR)


if __name__ == "__main__":
    main()