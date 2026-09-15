import json
from typing import List, Dict, Optional, Tuple


def load_data(data_name):
    if data_name == "cmb":
        question_file = ""
        answer_file = ""
        return load_cmb_data(question_file, answer_file)
    elif data_name == "medqa":
        dataset_path = ""
        return load_medqa_dataset(dataset_path)



def load_cmb_data(question_file, answer_file):
    print(f"\n📂 加载题目数据：{question_file}")
    print(f"📂 加载答案数据：{answer_file}")
    
    # 读取文件（兼容中文编码）
    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(answer_file, "r", encoding="utf-8") as f:
        answers = json.load(f)
    
    # 构建答案映射
    answer_map = {item["id"]: item["answer"] for item in answers}
    
    # 合并数据并统一格式
    cmb_data = []
    for q in questions:
        q_id = q["id"]
        # 拼接问题和选项（适配CMB数据格式，假设选项在q["options"]中）
        question_str = q["question"] + "\n"
        if "option" in q:
            # 处理选项格式（兼容字典/列表类型）
            if isinstance(q["option"], dict):
                for key, value in q["option"].items():
                    question_str += f"{key}. {value}\n"
            elif isinstance(q["option"], list):
                for idx, opt in enumerate(q["option"]):
                    question_str += f"{chr(65 + idx)}. {opt}\n"
        # 构建统一格式
        cmb_data.append({
            "question": question_str.strip(),  # 去除末尾换行
            "true_answer": answer_map.get(q_id, "")
        })
    
    print(f"✅ 数据合并完成，共加载 {len(cmb_data)} 道题目")
    return cmb_data


def load_medqa_dataset(dataset_path: str, encoding: str = "utf-8") -> List[Dict]:
    dataset: List[Dict] = []
    try:
        with open(dataset_path, "r", encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 校验必要字段，防止数据异常
                    required_keys = ["question", "options", "answer_idx"]
                    if all(k in data for k in required_keys):
                        # 拼接问题和选项
                        question_str = data["question"] + "\n"
                        # 处理选项（兼容字典/列表类型）
                        if isinstance(data["options"], dict):
                            for key, value in data["options"].items():
                                question_str += f"{key}. {value}\n"
                        elif isinstance(data["options"], list):
                            for idx, opt in enumerate(data["options"]):
                                question_str += f"{chr(65 + idx)}. {opt}\n"
                        # 构建统一格式
                        dataset.append({
                            "question": question_str.strip(),
                            "true_answer": data["answer_idx"]  # 直接使用原答案索引（A/B/C/D或数字）
                        })
                    else:
                        print(f"警告：第{line_num}行缺失必要字段，已跳过")
                except json.JSONDecodeError:
                    print(f"警告：第{line_num}行JSON格式错误，已跳过")
        print(f"✅ 成功加载数据集 | 有效题目数：{len(dataset)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"数据集文件不存在：{dataset_path}")
    except Exception as e:
        raise Exception(f"数据集读取失败：{str(e)}")
    return dataset


def load_medx_dataset(dataset_path: str, encoding: str = "utf-8") -> List[Dict]:
    dataset: List[Dict] = []
    try:
        with open(dataset_path, "r", encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 校验核心字段（适配test copy.json的label为标准答案）
                    required_keys = ["question", "options", "label"]
                    if all(k in data for k in required_keys):
                        # 拼接问题和选项，统一question字段格式
                        question_str = data["question"] + "\n"
                        # 构建统一格式的字典
                        unified_data = {
                            "question": question_str.strip(),  # 去除末尾多余换行
                            "true_answer": data["label"]  # 将原label字段映射为true_answer
                        }
                        dataset.append(unified_data)
                    else:
                        print(f"Warning: Line {line_num} missing key fields, skipped")
                except json.JSONDecodeError:
                    print(f"Warning: Line {line_num} JSON format error, skipped")
        print(f"✅ Load dataset successfully | Valid questions: {len(dataset)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    except Exception as e:
        raise Exception(f"Dataset load failed: {str(e)}")
    return dataset

def load_json_dataset(dataset_path: str, encoding: str = "utf-8") -> List[Dict]:
    dataset: List[Dict] = []
    try:
        with open(dataset_path, "r", encoding=encoding) as f:
            raw_data = json.load(f)
        # 校验并格式化数据，输出统一格式
        for idx, item in enumerate(raw_data):
            required_keys = ["question", "choices", "answer"]
            if all(k in item for k in required_keys):
                # 拼接问题和选项
                question_str = item["question"] + "\n"
                # 为选项添加序号（0→A,1→B,2→C,3→D）
                for i, choice in enumerate(item["choices"]):
                    question_str += f"{chr(65 + i)}. {choice}\n"
                # 构建统一格式
                dataset.append({
                    "question": question_str.strip(),
                    "true_answer": chr(65 + item["answer"])  # 标准答案转为A/B/C/D
                })
            else:
                print(f"警告：第{idx+1}条数据缺失必要字段，已跳过")
        print(f"✅ 成功加载数据集 | 有效题目数：{len(dataset)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"数据集文件不存在：{dataset_path}")
    except json.JSONDecodeError:
        raise Exception(f"数据集格式错误，非标准JSON")
    except Exception as e:
        raise Exception(f"数据集读取失败：{str(e)}")
    return dataset