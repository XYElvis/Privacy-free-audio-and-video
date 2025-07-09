#!/usr/bin/env python3
import sys
import os
import numpy as np
import librosa
from functools import lru_cache
import time
import logging
import io
import soundfile as sf
import math

logger = logging.getLogger(__name__)


@lru_cache(10 ** 6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a


def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]


class FasterWhisperASR:
    """使用faster-whisper库作为后端。在GPU上运行速度更快。"""

    sep = ""

    def __init__(self, modelsize="large-v2", cache_dir=None, model_dir=None):
        # 直接指定语言为中文
        self.original_language = "zh"
        self.transcribe_kargs = {}

        # 如果没有指定model_dir，使用当前目录下的model文件夹
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
            logger.info(f"使用默认模型目录: {model_dir}")

        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        # 如果model_dir存在，直接从该目录加载模型
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            logger.info(f"从model_dir加载whisper模型: {model_dir}")
            model_size_or_path = model_dir
        elif modelsize is not None:
            logger.info(f"从网络下载whisper模型: {modelsize}")
            model_size_or_path = modelsize
        else:
            raise ValueError("必须设置modelsize或model_dir参数")

        model = WhisperModel(model_size_or_path,
                             device="cpu",  # 使用cpu,更改点，后续使用GPU(cuda)
                             compute_type="int8",  # 使用FP16精度float16
                             download_root=cache_dir)
        return model

    def transcribe(self, audio, init_prompt=""):
        segments, info = self.model.transcribe(audio, language=self.original_language,
                                               initial_prompt=init_prompt, beam_size=5,
                                               word_timestamps=True, condition_on_previous_text=True,
                                               **self.transcribe_kargs)
        return list(segments)

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True


class HypothesisBuffer:
    """管理识别结果的假设缓冲区，处理增量识别结果的提交和合并。"""

    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None
        self.logfile = logfile

    def insert(self, new, offset):
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1])
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            for j in range(i):
                                self.new.pop(0)
                            break

    def flush(self):
        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer


class OnlineASRProcessor:
    """在线ASR处理器，处理音频块并生成实时转录结果。"""

    SAMPLING_RATE = 16000

    def __init__(self, asr, buffer_trimming=("segment", 15), logfile=sys.stderr):
        self.asr = asr
        self.logfile = logfile
        self.init()
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self, offset=None):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1

        p = self.commited[:k]
        p = [t for _, _, t in p]
        prompt = []
        l = 0
        while p and l < 200:
            x = p.pop(-1)
            l += len(x) + 1
            prompt.append(x)
        return self.asr.sep.join(prompt[::-1])

    def process_iter(self):
        prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(
            f"transcribing {len(self.audio_buffer) / self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)

        if o and self.buffer_trimming_way == "segment":
            if len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec:
                self.chunk_completed_segment(res)

        logger.debug(f"len of buffer now: {len(self.audio_buffer) / self.SAMPLING_RATE:2.2f}")
        return self.to_flush(o)

    def chunk_completed_segment(self, res):
        if self.commited == []: return

        ends = [s.end for s in res]

        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            t = self.commited[-1][1]
            if e <= t:
                logger.debug(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)

    def chunk_at(self, time):
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def finish(self):
        o = self.transcript_buffer.complete()
        self.buffer_time_offset += len(self.audio_buffer) / 16000
        return self.to_flush(o)

    def to_flush(self, sents, sep=None, offset=0):
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b, e, t)


def set_logging(args, logger):
    logging.basicConfig(format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str,
                        default='../audio/test.wav',  # 设置默认路径
                        nargs='?',  # 使参数可选
                        help="16kHz单声道wav音频文件路径")
    parser.add_argument('--min-chunk-size', type=float, default=1.0, help='最小音频块大小（秒）')
    parser.add_argument('--model', type=str, default='large-v2',
                        choices=["tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium",
                                 "large-v1", "large-v2", "large-v3", "large"], help="Whisper模型大小")
    parser.add_argument('--vad', action="store_true", default=False, help='使用语音活动检测')
    parser.add_argument('--model_dir', type=str, default=None, help="自定义模型目录")
    parser.add_argument("-l", "--log-level", dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="设置日志级别", default='INFO')
    args = parser.parse_args()

    set_logging(args, logger)

    # 直接指定语言为中文，不使用命令行参数
    asr = FasterWhisperASR(modelsize=args.model, model_dir=args.model_dir)

    if args.vad:
        asr.use_vad()

    online = OnlineASRProcessor(asr, buffer_trimming=("segment", 15))

    SAMPLING_RATE = 16000
    duration = len(load_audio(args.audio_path)) / SAMPLING_RATE
    logger.info(f"音频时长: {duration:.2f}秒")

    # 预热ASR模型
    a = load_audio_chunk(args.audio_path, 0, 1)
    asr.transcribe(a)

    beg = 0
    end = 0
    start = time.time()

    def output_transcript(o, now=None):
        if now is None:
            now = time.time() - start
        if o[0] is not None:
            print(f"{now * 1000:.4f} {o[0] * 1000:.0f} {o[1] * 1000:.0f} {o[2]}", flush=True)

    min_chunk = args.min_chunk_size

    while True:
        now = time.time() - start
        if now < end + min_chunk:
            time.sleep(min_chunk + end - now)
        end = time.time() - start
        a = load_audio_chunk(args.audio_path, beg, end)
        beg = end
        online.insert_audio_chunk(a)

        try:
            o = online.process_iter()
        except Exception as e:
            logger.error(f"处理错误: {e}")
        else:
            output_transcript(o)

        current_time = time.time() - start
        logger.debug(f"## 已处理 {end:.2f} 秒, 当前时间 {current_time:.2f} 秒, 延迟 {current_time - end:.2f} 秒")

        if end >= duration:
            break

    o = online.finish()
    output_transcript(o)


if __name__ == "__main__":
    main()