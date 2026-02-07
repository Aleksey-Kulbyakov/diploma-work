import os
import wandb
from dotenv import load_dotenv
from trl import GRPOConfig, GRPOTrainer

from src.config import CFG
from src.utils import seed_everything, print_memory_usage, print_trainable_parameters
from src.model import build_model_and_tokenizer
from src.dataset import get_gsm8k_dataset
from src.rewards import (
    xmlcount_reward_func,
    soft_format_reward_func,
    correctness_reward_func,
    aha_moment_reward_func,
    reasoning_steps_reward_func
)


def main():
    load_dotenv()

    seed_everything(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    print_memory_usage()

    wandb.init(
        project=CFG.PROJECT,
        id=CFG.WANDB_ID,
        resume="allow",
        name=CFG.RUN_NAME
    )

    model, tokenizer, peft_config = build_model_and_tokenizer()
    print_trainable_parameters(model)

    train_ds, test_ds = get_gsm8k_dataset(tokenizer)

    training_args = GRPOConfig(
        output_dir=CFG.SAVE_FINETUNED_MODEL_PATH,
        run_name=CFG.RUN_NAME,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        per_device_train_batch_size=CFG.BATCH_SIZE,
        gradient_accumulation_steps=CFG.GRAD_ACC,
        beta=CFG.KL_BETA,
        optim=CFG.OPTIMIZER,
        learning_rate=CFG.LR,
        warmup_ratio=CFG.WARMUP_RATIO,
        lr_scheduler_type=CFG.LR_SCHEDULER,
        max_steps=CFG.MAX_STEPS,
        num_generations=CFG.NUM_GENERATIONS,
        max_prompt_length=CFG.MAX_PROMPT_LENGTH,
        max_completion_length=CFG.MAX_COMPLETION_LENGTH,
        save_strategy='steps',
        save_steps=CFG.SAVE_STEPS,
        save_total_limit=3,
        logging_steps=CFG.LOG_STEPS,
        report_to='wandb',
        seed=CFG.SEED,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            correctness_reward_func,
            aha_moment_reward_func,
            reasoning_steps_reward_func
        ],
        train_dataset=train_ds,
        # peft_config=peft_config, # В новой версии TRL иногда PEFT конфиг лучше применять к модели до передачи в Trainer
    )

    print('\n\nStarting training...\n\n')

    checkpoint = CFG.RESUME_FROM_CHECKPOINT if CFG.CONTINUE_TRAINING else None
    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(CFG.SAVE_FINETUNED_MODEL_PATH)
    print("Training Completed.")


if __name__ == "__main__":
    import torch

    main()