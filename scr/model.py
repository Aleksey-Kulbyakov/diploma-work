import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from .config import CFG


def build_model_and_tokenizer():
    print(f'\n ********** Building {CFG.MODEL_NAME} ********** \n')

    bnb_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=bnb_dtype,
        bnb_4bit_quant_type='nf4',
    )

    model = AutoModelForCausalLM.from_pretrained(
        CFG.MODEL_NAME,
        quantization_config=quantization_config,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    # Подготовка модели для обучения в 4 бита
    model = prepare_model_for_kbit_training(model)

    # Настройка LoRA
    peft_config = LoraConfig(
        r=CFG.LORA_R,
        lora_alpha=CFG.LORA_ALPHA,
        lora_dropout=CFG.LORA_DROPOUT,
        target_modules=CFG.TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Применяем LoRA
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False  # Важно для обучения

    return model, tokenizer, peft_config