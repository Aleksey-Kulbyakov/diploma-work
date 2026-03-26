def extract_xml_answer(text: str) -> str:
    """Извлекает содержимое тега <answer>."""
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except:
        return ""


# def check_structure_reward_func(completions, **kwargs) -> list[float]:
#     """
#     Проверяет наличие тегов и штрафует за их отсутствие или дублирование.
#     Аналог match_format_approximately из второго кода.
#     """
#     rewards = []
#     for completion in completions:
#         score = 0.0
#         # Проверяем наличие открывающих и закрывающих тегов
#         if completion.count("<think>") == 1: score += 0.2
#         if completion.count("</think>") == 1: score += 0.2
#         if completion.count("<answer>") == 1: score += 0.2
#         if completion.count("</answer>") == 1: score += 0.2

#         # Штраф за отсутствие тегов
#         if "<think>" not in completion: score -= 0.5
#         if "<answer>" not in completion: score -= 0.5

#         rewards.append(score)
#     return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Строгая проверка формата через Regex.
    Аналог match_format_exactly.
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    responses = [completion for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [1.0 if match else 0.0 for match in matches]

def soft_correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Главное улучшение: МЯГКАЯ ПРОВЕРКА ОТВЕТА.
    Аналог check_answer и check_numbers из второго кода.
    """
    responses = [extract_xml_answer(c) for c in completions]
    rewards = []

    for r, a in zip(responses, answer):
        # 1. Точное совпадение строки (Самая высокая награда)
        if r == a:
            rewards.append(4.0)
            continue

        # 2. Попытка числового сравнения (Soft Reward)
        try:
            # Очистка от запятых и лишних символов (например 1,200 -> 1200)
            r_clean = re.sub(r"[^\d\.\-]", "", r)
            a_clean = re.sub(r"[^\d\.\-]", "", a)

            if not r_clean or not a_clean:
                rewards.append(0.0)
                continue

            r_val = float(r_clean)
            a_val = float(a_clean)

            # Точное числовое совпадение (например 10.0 и 10)
            if r_val == a_val:
                rewards.append(4.0)
            # Допустимое отклонение (Ratio 0.9 - 1.1)
            else:
                ratio = r_val / a_val if a_val != 0 else 0
                if 0.9 <= ratio <= 1.1:
                    rewards.append(1.0)  # Малая награда за "близость"
                elif 0.8 <= ratio <= 1.2:
                    rewards.append(0.5)  # Микро награда
                else:
                    rewards.append(0.0)
        except:
            rewards.append(0.0)

    return rewards
