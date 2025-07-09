import whisper
import zhconv
    # 加载模型并进行语音识别
def whisper_audio():
    model = whisper.load_model("medium")
    result = model.transcribe("test1.wav", language='Chinese', fp16=False, word_timestamps=True)
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

    # 后处理：确保所有时间戳顺序正确且不超过音频总长度
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
