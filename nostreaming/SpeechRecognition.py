import whisper
import zhconv
import time  # 新增时间模块用于计时
from openai import OpenAI, OpenAIError
    # 加载模型并进行语音识别
def whisper_audio(model_level,audio_path):
    model = whisper.load_model(model_level)#medium,base
    result = model.transcribe(audio_path, language='Chinese', fp16=False, word_timestamps=True)
    s = result["text"]
    s1 = zhconv.convert(s, 'zh-cn')
    # 获取每个字符的时间信息
    char_segments = []
    current_char_index = 0

    # 遍历所有的段
    for segment in result["segments"]:
        # 优先使用单词级时间信息
        if "words" in segment:
            for word_info in segment["words"]:
                word = zhconv.convert(word_info["word"], 'zh-cn').strip()
                if not word:
                    continue
                word_start = word_info["start"]
                word_end = word_info["end"]
                word_duration = word_end - word_start
                # 特殊处理单字符单词
                if len(word) == 1:
                    char = word
                    if current_char_index >= len(s1) and char != s1[current_char_index]:
                        continue
                    char_segments.append({
                        "char": char,
                        "start": word_start,
                        "end": word_end,
                        "index": current_char_index
                    })
                    current_char_index += 1
                    continue
                # 多字符单词的处理
                chars = list(word)
                char_count = len(chars)
                # 基于令牌的持续时间分布
                if "tokens" in word_info:
                    # 使用子令牌信息（如果有）
                    token_durations = []
                    total_token_duration = 0

                    # 计算每个子令牌的持续时间
                    for token in word_info["tokens"]:
                        # 这里简化处理，实际应用中可能需要tokenizer解码
                        token_text = tokenizer.decode([token])
                        token_len = len(token_text)
                        token_durations.append((token_text, token_len))
                        total_token_duration += token_len

                    # 分配时间
                    current_time = word_start
                    for token_text, token_len in token_durations:
                        token_weight = token_len / total_token_duration
                        token_duration = word_duration * token_weight

                        # 为token中的每个字符分配时间
                        for i, char in enumerate(token_text):
                            if current_char_index >= len(s1):
                                break

                            if char != s1[current_char_index]:
                                continue

                            # 线性分配token内的时间
                            char_start = current_time + (i * token_duration / token_len)
                            char_end = char_start + (token_duration / token_len)

                            char_segments.append({
                                "char": char,
                                "start": char_start,
                                "end": char_end,
                                "index": current_char_index
                            })

                            current_char_index += 1

                        current_time += token_duration

                else:
                    # 没有子令牌信息时，平均分配时间
                    char_duration = word_duration / char_count

                    for i, char in enumerate(chars):
                        if current_char_index >= len(s1):
                            break

                        if char != s1[current_char_index]:
                            continue

                        char_start = word_start + (i * char_duration)
                        char_end = char_start + char_duration

                        char_segments.append({
                            "char": char,
                            "start": char_start,
                            "end": char_end,
                            "index": current_char_index
                        })

                        current_char_index += 1

        # 回退到段级别处理
        elif "tokens" in segment:
            tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language='Chinese')
            segment_text = zhconv.convert(segment["text"], 'zh-cn').strip()

            if not segment_text:
                continue

            seg_start = segment["start"]
            seg_end = segment["end"]
            seg_duration = seg_end - seg_start

            # 尝试使用段内的token信息
            if "tokens" in segment:
                # 处理段内的每个token
                current_time = seg_start
                tokens = segment["tokens"]

                # 计算所有token的总长度
                total_token_len = 0
                token_texts = []

                for token in tokens:
                    token_text = tokenizer.decode([token])
                    token_texts.append(token_text)
                    total_token_len += len(token_text)

                # 为每个token分配时间
                for token_text in token_texts:
                    token_len = len(token_text)
                    token_weight = token_len / total_token_len
                    token_duration = seg_duration * token_weight

                    # 为token中的每个字符分配时间
                    for i, char in enumerate(token_text):
                        if current_char_index >= len(s1):
                            break

                        if char != s1[current_char_index]:
                            continue

                        char_start = current_time + (i * token_duration / token_len)
                        char_end = char_start + (token_duration / token_len)

                        char_segments.append({
                            "char": char,
                            "start": char_start,
                            "end": char_end,
                            "index": current_char_index
                        })

                        current_char_index += 1

                    current_time += token_duration

            else:
                # 没有token信息时，简单平均分配
                chars = list(segment_text)
                char_count = len(chars)
                char_duration = seg_duration / char_count

                for i, char in enumerate(chars):
                    if current_char_index >= len(s1):
                        break

                    if char != s1[current_char_index]:
                        continue

                    char_start = seg_start + (i * char_duration)
                    char_end = char_start + char_duration

                    char_segments.append({
                        "char": char,
                        "start": char_start,
                        "end": char_end,
                        "index": current_char_index
                    })

                    current_char_index += 1

    # 后处理：确保所有时间戳顺序正确
    if char_segments:
        # 确保时间戳顺序正确且不重叠
        for i in range(1, len(char_segments)):
            prev = char_segments[i - 1]
            curr = char_segments[i]

            if curr["start"] < prev["end"]:
                curr["start"] = prev["end"]

            if curr["end"] <= curr["start"]:
                curr["end"] = curr["start"] + 0.01  # 最小时间间隔

    # 生成总识别文本
    recognized_text = ''.join([char_info['char'] for char_info in char_segments])

    # 输出统计信息
    print(f"总识别字符数: {len(char_segments)}")
    audio_duration = char_segments[-1]['end'] if char_segments else 0
    print(f"音频总长度: {audio_duration:.2f}秒")
    print(f"识别文本: {recognized_text}")

    # 输出结果示例
    for i, char_info in enumerate(char_segments):
        print(
            f"字符: {char_info['char']}, 位置: {char_info['index']}, 开始时间: {char_info['start']:.2f}s, 结束时间: {char_info['end']:.2f}s")
    return recognized_text,char_segments

# ai识别隐私词
def TextMatching(text):
    print(f"输入的文本: {text}")
    client = OpenAI(
        api_key="sk-084504fc8e2f4d5fa1c3f6be79762aed",
        base_url="https://api.deepseek.com"
    )

    # 构建包含隐私检测提示和用户文本的消息
    messages = [
        {"role": "system",
         "content": "你是一个信息提取助手。你的任务是找出用户提供的文本中的隐私词，包括名字、号码、家庭住址、工作单位名字和工作单位地址。请只返回检测到的隐私词，每个隐私词占一行,每一行再空一格写该隐私词的类型，输出类型只包含名字、号码、家庭住址、工作单位名字和工作单位地址。如果没有找到隐私词，请返回'未检测到隐私词'。"},
        {"role": "user", "content": text}
    ]

    try:
        # 调用DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-reasoner",#deepseek-chat
            messages=messages,
            stream=False
        )

        # 提取模型返回的隐私词
        privacy_words = response.choices[0].message.content.strip()
        privacy_list = []
        # 输出结果
        if privacy_words == "未检测到隐私词":
            print("未检测到隐私词")
        else:
            print("检测到的隐私词:")
            for line in privacy_words.split('\n'):
                parts = line.strip().split(' ', 1)  # 只分割一次，避免类型中有空格
                if len(parts) == 2:
                    word, category = parts
                    privacy_list.append([word, category])
                    print(f"{word} - {category}")

        return privacy_list

    except OpenAIError as e:
        print(f"API调用错误: {e}")
        return None

# 替换音频中的隐私词
def AudioReplacement(audio_path,char_time_segments,privacy_list):
    return 0

def main():
    start_time = time.time()  # 记录开始时间
    model_level="medium"
    audio_path = "../audio/test1.wav"

    recognized_text,char_time_segments=whisper_audio(model_level,audio_path) # 进行语音识别（标记文字时间） base,medium

    privacy_list=TextMatching(recognized_text) # 进行ai识别隐私词

    # 测试
    # privacy_list=TextMatching("5月17日下午3点,张三教授在2号楼405会议室进行能工智能与语音交互技术的主题演讲。参会人员需携带笔记本和U盘,途中注意安全。另外,明天的会议改在8楼东侧会议室。如果有疑问,请拨打138-5672-9014咨询。")
    # privacy_list=[['张三', '名字'], ['138-5672-9014', '号码']]

    # AudioReplacement(audio_path,char_time_segments,privacy_list) # 音频替换，保存替换后的文件


    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总运行时间
    # 打印运行时间，保留两位小数
    print(f"程序运行时间: {total_time:.2f} 秒")


if __name__ == '__main__':
    main()