import re


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        # Штраф за слишком длинный ответ внутри тегов
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


# --- Reward Functions ---

def aha_moment_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    aha_markers = [
        "wait", "actually", "let me re-check", "no, that's wrong",
        "correction", "re-calculating", "hold on", "let me rethink"
    ]
    rewards = []
    for r in responses:
        think_match = re.search(r"<think>(.*?)</think>", r, re.DOTALL)
        if think_match:
            content = think_match.group(1).lower()
            has_aha = any(marker in content for marker in aha_markers)
            rewards.append(1.0 if has_aha else 0.0)
        else:
            rewards.append(0.0)
    return rewards


def reasoning_steps_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    rewards = []
    for r in responses:
        steps = re.findall(r"(Step \d|First|Second|Finally)", r)
        rewards.append(0.1 if len(steps) >= 3 else 0.0)
    return rewards


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    # Логирование для отладки (можно убрать в продакшене)
    # Здесь можно добавить print, но лучше использовать logging

    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion for completion in completions]
    return [count_xml(c) for c in contents]