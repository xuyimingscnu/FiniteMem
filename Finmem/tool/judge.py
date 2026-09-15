import re

def judge(response, true_answer):
    pred_answer = parse_answer_from_response(response)
    is_correct, compare_info, pred_answer = judge_answer_correct(pred_answer, true_answer)
    return is_correct, compare_info, pred_answer


def parse_answer_from_response(response):
    if "答案：" in response:
    # if "Answer:" in response:
        pattern = r"【答案：(.*?)】"
        # pattern = r"【Answer:(.*?)】"
        match = re.search(pattern, response)
        if match:
            # 如果找到匹配，返回第一个捕获组的内容（即答案）
            answer = match.group(1)
        else:
            # 如果没找到，返回 None 或你指定的提示
            answer =  None
        return answer
    print("⚠️ 未解析到答案")
    return ""

def judge_answer_correct(pred_answer, true_answer):
    def normalize_answer(ans):
        if not ans:
            return ""
        # 多选答案排序（如ABC和BAC视为相同）
        return "".join(sorted([c for c in ans.strip().upper() if c.isalpha()]))
    
    pred_norm = normalize_answer(pred_answer)
    true_norm = normalize_answer(true_answer)
    is_correct = pred_norm == true_norm
    
    compare_info = f"预测：{pred_answer} | 标准答案：{true_answer} | 是否正确：{is_correct}"
    return is_correct, compare_info, pred_answer
