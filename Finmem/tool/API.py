from openai import OpenAI

_TOTAL_TOKENS = 0

class Config:
    # 基础信息
    API_BASE_URL = ""
    # API_BASE_URL = "http://127.0.0.1:8000"
    OPENAI_API_KEY= ""
    Max_tokens = 1024
    Temperature = 0.1
    Top_p = 0.9
    Repetition_penalty = 1.5
    MODEL_NAEM = 
    
def generate(prompt):
    global _TOTAL_TOKENS
    client = OpenAI(
            base_url=f"{Config.API_BASE_URL}/v1",
            api_key=Config.OPENAI_API_KEY  # 本地服务无需真实密钥，任意值即可
        )
    completion = client.chat.completions.create(
            model=Config.MODEL_NAEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=Config.Max_tokens,
            temperature=Config.Temperature,
            top_p=Config.Top_p,
            frequency_penalty=Config.Repetition_penalty,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
    response_text = completion.choices[0].message.content.strip()
    if completion.usage and completion.usage.total_tokens:
        _TOTAL_TOKENS += completion.usage.total_tokens
    
    return response_text


def get_total_tokens():
    return _TOTAL_TOKENS