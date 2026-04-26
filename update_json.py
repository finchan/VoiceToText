import json

def process_json_with_role_list(input_json_str, names_list):
    """
    input_json_str: 原始 JSON 字符串
    names_list: 对话人顺序列表
    """
    try:
        # 尝试解析 JSON
        data = json.loads(input_json_str)
    except json.JSONDecodeError as e:
        return f"脚本报错：JSON 解析失败。具体错误原因：{e}"
    
    # 过滤掉标题行（第一个元素，如「会話文 1」），从第二段对话开始匹配人名
    dialogue_segments = data[1:] 
    
    for i, segment in enumerate(dialogue_segments):
        # 如果名字列表长度不足，则停止处理剩下的段落
        if i >= len(names_list):
            break
            
        current_name = names_list[i]
        name_tag = f"{current_name}　" # 名字后面加了一个日语全角空格
        
        # 1. 在 text 头部插入名字（去除原文本开头的多余空格）
        segment["text"] = name_tag + segment["text"].lstrip()
        
        # 2. 在 words 数组开头插入人名片段，时间戳设为该句的开始时间
        start_time = segment.get("start", 0)
        new_word_segment = {
            "word": name_tag,
            "start": start_time,
            "end": start_time
        }
        
        if "words" in segment:
            segment["words"].insert(0, new_word_segment)
    
    # 生成处理后的 JSON 字符串
    final_json = json.dumps(data, ensure_ascii=False, indent=2)
    
    # --- 修改部分：将结果写入 temp.txt ---
    try:
        with open("temp.txt", "w", encoding="utf-8") as f:
            f.write(final_json)
        print("处理完成！结果已成功写入到 temp.txt")
    except Exception as e:
        print(f"写入文件失败：{e}")
        
    return final_json


# 直接从文件读取，不会有粘贴导致的格式问题
with open(r'C:\PROJECTS\JapaneseVoice\resources\日语综合教程第一册\第10課　上海のバンド\10-会話-3.json', 'r', encoding='utf-8') as f:
    original_json_data = f.read()
speaker_order = ["吕：", "陆：", "吕：", "陆：", "吕：", "陆："]

result = process_json_with_role_list(original_json_data, speaker_order)
print(result)