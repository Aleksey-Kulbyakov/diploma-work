import re


def extract_xml_answer(text: str) -> str:
    """Извлекает содержимое тега <answer>."""
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except Exception:
        return ""


def check_structure_reward_func(completions, **kwargs) -> list[float]:
    """Проверяет наличие тегов и штрафует за их отсутствие или дублирование."""
    rewards = []
    for completion in completions:
        score = 0.0
        if completion.count("<think>") == 1: score += 0.2
        if completion.count("</think>") == 1: score += 0.2
        if completion.count("<answer>") == 1: score += 0.2
        if completion.count("</answer>") == 1: score += 0.2

        if "<think>" not in completion: score -= 0.5
        if "<answer>" not in completion: score -= 0.5

        rewards.append(score)
    return rewards


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Строгая проверка формата через Regex."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    responses = [completion for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [1.0 if match else 0.0 for match in matches]


def soft_correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Мягкая проверка ответа с пропорциональными наградами."""
    responses = [extract_xml_answer(c) for c in completions]
    rewards = []

    for r, a in zip(responses, answer):
        if r == a:
            rewards.append(4.0)
            continue

        try:
            r_clean = re.sub(r"[^\d\.\-]", "", r)
            a_clean = re.sub(r"[^\d\.\-]", "", a)

            if not r_clean or not a_clean:
                rewards.append(0.0)
                continue

            r_val = float(r_clean)
            a_val = float(a_clean)

            if r_val == a_val:
                rewards.append(4.0)
            else:
                ratio = r_val / a_val if a_val != 0 else 0
                if 0.9 <= ratio <= 1.1:
                    rewards.append(1.0)
                elif 0.8 <= ratio <= 1.2:
                    rewards.append(0.5)
                else:
                    rewards.append(0.0)
        except Exception:
            rewards.append(0.0)

    return rewards


def deepseek_true_original_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Реконструкция логики из статьи DeepSeek-R1."""
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        reward = 0.0

        has_correct_format = bool(re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", completion, re.DOTALL))
        if has_correct_format:
            reward += 1.0

        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if match:
            extracted_answer = match.group(1).strip()
            r_clean = re.sub(r"[^\d\.\-]", "", extracted_answer)
            gt_clean = re.sub(r"[^\d\.\-]", "", str(ground_truth))
            if r_clean and r_clean == gt_clean:
                reward += 1.0
        rewards.append(reward)
    return rewards