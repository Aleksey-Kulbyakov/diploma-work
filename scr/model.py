import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from .config import CFG


def build_model_and_tokenizer():
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f'\n ********** Building {CFG.MODEL_NAME} ********** \n')

    model = AutoModelForCausalLM.from_pretrained(
        CFG.MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
    )

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=CFG.LORA_R,
        lora_alpha=CFG.LORA_ALPHA,
        lora_dropout=CFG.LORA_DROPOUT,
        target_modules=CFG.TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return model, tokenizer, peft_config