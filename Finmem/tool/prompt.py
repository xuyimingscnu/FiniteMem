solver_system_prompt = """你是一款用于解决问题的智能智能体。
## 推理要求
根据[问题]生成推理步骤，推理步骤有以下要求：
1.强制要求只能生成思考、搜索、总结和完成中的一个推理步骤。
2.推理步骤有以下四种类型和格式：
(1) 思考[文本], 分析问题中缺少的信息，例如：思考[我需要搜索输血后肝炎的常见致病病毒，确认哪种病毒是最易引发输血后肝炎的。]
(2) 总结[文本], 根据观察内容进行推理。例如：总结[根据观察结果，丙型肝炎病毒是最易引起输血后肝炎的病毒，对应选项C。]
(3) 搜索[文本], 文本为需要医学资源上搜索的内容。样例：搜索[基础代谢率异常对应的疾病]
(4) 完成[答案], 如果[推理步骤]已拥有足够信息，则返回答案并结束任务，答案的格式为【答案：X】（X为选项字母）。样例：完成[【答案：C】]
3.如果[推理步骤]中的最新一步是思考，则下一步不能是思考。
4.如果[推理步骤]中的最新一步是观察，则下一步应该是总结。
5.参考经验总结，参考知识和下一步可能的操作中的内容，以此为参考，指导本次任务的思考逻辑与处理方式
## 经验总结
以下为过往同类任务积累的解题要点，可辅助提升本次答题准确率：
{insight_text}
## 参考知识
以下为过往同类任务积累的参考知识，可辅助提升本次答题准确率：
{knowledge_text}
## 下一步可能的操作
以下为过往同类任务积累的可能操作，你可以参考这些操作生成下一步的推理步骤，可辅助提升本次答题准确率：
{trajectory_text}
"""

react_prompt = """根据[问题]生成推理步骤，推理步骤有以下要求：
1.强制要求只能生成思考、搜索、总结和完成中的一个推理步骤。
2.推理步骤有以下四种类型和格式：
(1) 思考[文本], 分析问题中缺少的信息，例如：思考[我需要搜索输血后肝炎的常见致病病毒，确认哪种病毒是最易引发输血后肝炎的。]
(2) 总结[文本], 根据观察内容进行推理。例如：总结[根据观察结果，丙型肝炎病毒是最易引起输血后肝炎的病毒，对应选项C。]
(3) 搜索[文本], 文本为需要医学资源上搜索的内容。样例：搜索[基础代谢率异常对应的疾病]
(4) 完成[答案], 如果[推理步骤]已拥有足够信息，则返回答案并结束任务，答案的格式为【答案：X】（X为选项字母）。样例：完成[【答案：C】]
3.如果[推理步骤]中的最新一步是思考，则下一步不能是思考。
4.如果[推理步骤]中的最新一步是观察，则下一步应该是总结。
"""

react_prompt_en = """Generate reasoning steps based on [Question], with the following requirements for the reasoning steps:
1.It is mandatory to generate only one reasoning step from the following: Think, Search, Summarize, or Complete.
2.There are four types and formats of reasoning steps:
(1) Think [Text], analyzing information missing in the question. Example: Think [I need to search for common pathogenic viruses of post-transfusion hepatitis and confirm which virus most frequently causes post-transfusion hepatitis.]
(2) Summarize [Text], reasoning based on observed content. Example: Summarize [According to the observations, hepatitis C virus is the virus that most frequently causes post-transfusion hepatitis, corresponding to option C.]
(3) Search [Text], where the text is the content to be searched in medical resources. Example: Search [Diseases corresponding to abnormal basal metabolic rate]
(4) Complete [Answer]. If sufficient information is available in the [reasoning steps], return the answer and terminate the task in the format 【Answer: X】 (X is the option letter). Example: Complete [【Answer:C】]
3.If the latest step in the [reasoning steps] is Think, the next step cannot be Think.
4.If the latest step in the [reasoning steps] is Observe, the next step should be Summarize.
"""