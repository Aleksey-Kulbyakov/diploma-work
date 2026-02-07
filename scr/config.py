import os


class CFG:

    DEBUG = False
    PERCENT_OF_DATASET = 0.50 if DEBUG else 1.0  # Логика: если дебаг, то меньше данных

    CONTINUE_TRAINING = False
    RESUME_FROM_CHECKPOINT = '/path/to/checkpoint'  # Укажите путь, если CONTINUE_TRAINING = True
    MAX_STEPS = 1000

    PROJECT = 'GRPO-Reasoning-Analysis'
    RUN_NAME = "qwen-3b-aha-moment-v1"
    WANDB_ID = "qwen-3b-thesis-v2"  # Фиксированный ID для продолжения логов

    SAVE_STEPS = 100
    LOG_STEPS = 5
    OUTPUT_DIR = f'./{PROJECT}'
    SAVE_FINETUNED_MODEL_PATH = f"{OUTPUT_DIR}/{RUN_NAME}"

    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

    LORA_R = 4
    LORA_ALPHA = 4 * LORA_R
    LORA_DROPOUT = 0.025
    TARGET_MODULES = ["v_proj", "o_proj"]

    BATCH_SIZE = 2
    GRAD_ACC = 4
    NUM_GENERATIONS = 8
    MAX_PROMPT_LENGTH = 256 if DEBUG else 512
    MAX_COMPLETION_LENGTH = 1024
    KL_BETA = 0.03

    LR = 1e-6
    OPTIMIZER = "adamw_8bit"
    WEIGHT_DECAY = 0.1
    MAX_GRAD_NORM = 0.1
    WARMUP_RATIO = 0.03
    LR_SCHEDULER = "cosine"

    SEED = 983
    DATASET_PATH = 'openai/gsm8k'