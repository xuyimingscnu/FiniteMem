from tool.loadData import load_data
from tool.judge import judge
from tool.prompt import solver_system_prompt, react_prompt
from tool.API import generate, get_total_tokens
from memory.retrieve import set_current_library  
from memory.knowsight_mem import KnowsightLibrary
from memory.trajectory_mem import TrajectoryLibrary

from tqdm import tqdm
import re
import json
from datetime import datetime
import os

class Config:
    data_name = "medqa"
    
    generation_save_path = ""
    wrong_save_path = ""
    error_report_save_path = ""
    ans_save_path = ""

    Trajectory_save_path = ""
    Insight_save_path = ""
    Knowledge_save_path = ""
    
    Max_infer = 15 
    ANSWER_PATTERN = r"【答案：(.*?)】"
    MODEL_NAEM = ""

    @classmethod
    def update_paths(cls, timestamp_str=None):
        if timestamp_str is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{cls.data_name}-{cls.MODEL_NAEM}-{timestamp_str}"
        base_dir = f"{folder_name}"
        os.makedirs(base_dir, exist_ok=True)

        cls.generation_save_path = f"{base_dir}/gtskr.json"
        cls.wrong_save_path = f"{base_dir}/gtskr-wrong.json"
        cls.error_report_save_path = f"{base_dir}/gtskr-error_report.json"
        cls.ans_save_path = f"{base_dir}/gtskr.txt"

        cls.Trajectory_save_path = f"{base_dir}/Trajectory.json"
        cls.Insight_save_path = f"{base_dir}/Insight.json"
        cls.Knowledge_save_path = f"{base_dir}/Knowledge.json"

def extract_info(sentence):
    pattern = r'(思考|搜索|完成|总结)\[(.*?)\]'
    match = re.search(pattern, sentence)
    
    if match:
        action = match.group(1)      
        bracket_content = match.group(2)  
        return action, bracket_content
    else:
        print("输入格式不匹配，未提取到内容")
        return None, None


def search(text):
    prompt = "你是一个医学专家，请回答[" + text + "]，只回答一段话"
    return generate(prompt)


def react(question, prompt: str, trajectory: TrajectoryLibrary, insight: KnowsightLibrary, knowledge: KnowsightLibrary):
    insight_text = insight.memory_retrieve(question)
    knowledge_text = knowledge.memory_retrieve(question)
    trajectory_text = trajectory.memory_retrieve(question)

    infer_process = {}
    infer_process["question"] = question
    infer_process["prompt"] = prompt
    track = ""
    step = []
    trajectory_text_list = []
    count = 0 

    reminder = "特别注意，强制要求只能生成思考、搜索、总结和完成中的一个推理步骤。"
    
    while count < Config.Max_infer:
        trajectory_text_list += trajectory_text
        input_text = prompt + "\n[问题]:" + question + "\n[推理步骤]:\n" + track
        input_text = input_text.format(
            insight_text=insight_text,
            knowledge_text=knowledge_text,
            trajectory_text=trajectory_text
        )
        response_text = generate(input_text)
        input_text.replace(reminder, "")
        action, content = extract_info(response_text) 
        if action == "思考" or action == "总结":
            track = track + response_text + "\n"
            step.append(response_text)
        elif action == "搜索":
            step.append(response_text)
            text = search(content) 
            observation = "\n观察[" + text + "]\n"
            track = track + response_text + observation
            step.append(observation.replace("\n", ""))
        elif action == "完成":
            step.append(response_text)
            infer_process["step"] = step
            return response_text, infer_process, trajectory_text_list, insight_text, knowledge_text 
        else:
            track = track + reminder 
        trajectory_text = trajectory.memory_retrieve(question, response_text)
        count += 1
    fallback_prompt = "请根据问题直接生成答案，答案的格式为【答案：X】（X为选项字母）。问题：" + question
    fallback_ans = generate(fallback_prompt)
    infer_process["step"] = step  

    return fallback_ans, infer_process, trajectory_text_list, insight_text, knowledge_text  
            

def run():
    Config.update_paths()
    dataset = load_data(Config.data_name)
    total = len(dataset)
    
    trajectory = TrajectoryLibrary()
    insight = KnowsightLibrary(lib_type="insight")
    knowledge = KnowsightLibrary(lib_type="knowledge")

    
    full_result_list = []
    wrong_detail_list = []
    error_report_list = []
    
    
    for idx, data in tqdm(enumerate(dataset), total=total):
        infer_process = {}
        prompt = solver_system_prompt

        try:
            set_current_library(trajectory) 
            predict_ans, infer_process, trajectory_text, insight_text,  knowledge_text = react(data["question"], prompt, trajectory, insight, knowledge)
            is_correct, compare_info, pred_answer = judge(predict_ans, data["true_answer"])
            infer_process["true_answer"] = data["true_answer"]
            infer_process["pred_answer"] = pred_answer
            infer_process["is_correct"] = is_correct
            infer_process["compare_info"] = compare_info
            if is_correct:
                full_result_list.append(infer_process)
            else:
                full_result_list.append(infer_process)
                wrong_detail_list.append(infer_process)

            trajectory.importance_score_modify(trajectory_text_list=trajectory_text, effect=is_correct)
            insight.importance_score_modify(knowsight_text_list=insight_text, effect=is_correct)
            knowledge.importance_score_modify(knowsight_text_list=knowledge_text, effect=is_correct)
            if is_correct:
                trajectory.build_from_input(infer_process)
                insight.add_from_input(infer_process)
                knowledge.add_from_input(infer_process)
        except:
            error_report_list.append(data)
        
        with open(Config.generation_save_path, "w", encoding="utf-8") as f:
            json.dump(full_result_list, f, ensure_ascii=False, indent=2)
        with open(Config.wrong_save_path, "w", encoding="utf-8") as f:
            json.dump(wrong_detail_list, f, ensure_ascii=False, indent=2)
        with open(Config.error_report_save_path, "w", encoding="utf-8") as f:
            json.dump(error_report_list, f, ensure_ascii=False, indent=2)

        insight.save_to_json(Config.Insight_save_path)
        knowledge.save_to_json(Config.Knowledge_save_path)
        trajectory.save_to_json(Config.Trajectory_save_path)

        if idx % 100 == 0 and idx > 0:
            wrong = len(wrong_detail_list)
            correct = idx - len(wrong_detail_list)
            accuracy = correct / idx
            print(f"总题目数：{total} | 正确数：{correct} | 错误数：{wrong} | 正确率：{accuracy} ")


    wrong = len(wrong_detail_list)
    correct = total - len(wrong_detail_list)
    accuracy = correct / total
    total_tokens = get_total_tokens()          
    
    print(f"总题目数：{total} | 正确数：{correct} | 错误数：{wrong} | 正确率：{accuracy} | 消耗Token总数：{total_tokens}")
    with open(Config.ans_save_path, "w", encoding="utf-8") as f:
        f.write(f"总题目数：{total} | 正确数：{correct} | 错误数：{wrong} | 正确率：{accuracy} | 消耗Token总数：{total_tokens}")

        
    
if __name__ == "__main__":
    run()
