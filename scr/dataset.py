from datasets import load_dataset
from .config import CFG

# SYSTEM_PROMPT = """
# A conversation between User and Assistant. The user asks a question, and the Assistant solves it.

# The assistant first thinks about the reasoning process in the mind and then provides the user
# with the answer. The reasoning process and answer are enclosed within <think> </think> and
# <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
# <answer> answer here </answer>.

# """
SYSTEM_PROMPT = """You are a helpful assistant.
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
The answer inside <answer> tags must be a single number."""


def extract_tagged_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_dataset(tokenizer):
    data = load_dataset(CFG.DATASET_PATH, 'main')

    def format_example(x):
        return {
            'question': x['question'],
            'answer': extract_tagged_answer(x['answer']),
            'prompt': f"<|im_start|>system\n{SYSTEM_PROMPT}\nUser: {x['question']}<|im_end|>\n<|im_start|>assistant\n"
        }

    data = data.map(format_example)

    train_ds = data["train"]
    test_ds = data["test"]

    if CFG.DEBUG:
        train_sample_size = int(len(train_ds) * CFG.PERCENT_OF_DATASET)
        test_sample_size = int(len(test_ds) * CFG.PERCENT_OF_DATASET)
        train_ds = train_ds.shuffle(seed=CFG.SEED).select(range(train_sample_size))
        test_ds = test_ds.shuffle(seed=CFG.SEED).select(range(test_sample_size))

    return train_ds, test_ds