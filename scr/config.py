import os
class CFG:
    DEBUG = False
    PERCENT_OF_DATASET = 0.50

    MAX_STEPS = 1500

    PROJECT = 'GRPO-v100'
    RUN_NAME = "v100baseg8withoutxml"

    SAVE_STEPS = 50
    LOG_STEPS = 5

    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

    LORA_R = 64
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.05

    BATCH_SIZE = 1
    GRAD_ACC = 12
    NUM_GENERATIONS = 8

    MAX_PROMPT_LENGTH = 512
    MAX_COMPLETION_LENGTH = 1024
    KL_BETA = 0.06

    LR = 2e-6
    OPTIMIZER = "adamw_torch"
    WEIGHT_DECAY = 0.1
    MAX_GRAD_NORM = 0.3
    WARMUP_RATIO = 0.1
    LR_SCHEDULER = "cosine"

    SEED = 983
    DATASET_PATH = 'openai/gsm8k'
    OUTPUT_DIR = f'./{PROJECT}'
    SAVE_FINETUNED_MODEL_PATH = f"{OUTPUT_DIR}/{RUN_NAME}"
