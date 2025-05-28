from dataclasses import dataclass, field
import asyncio
from asyncio import Queue
import time
import re
from typing import List


@dataclass
class SegmentationConfig:
    max_waiting_time: float  # 等待超时时间
    max_stream_time: float  # 流式超时时间
    first_min_seg_size: int
    min_seg_size: int


@dataclass
class SegmentationStatus:
    config: SegmentationConfig  # 固定配置

    in_queue: Queue = field(default_factory=Queue)  # 接收外部输入的文本流
    out_queue: Queue = field(default_factory=Queue)  # 输出异步生成器到外部
    mid_queue: Queue = field(default_factory=Queue)  # 传递文本到异步生成器
    source: List[str] = field(default_factory=list)  # 存储原始输入的文本流
    segmenteds: List[str] = field(default_factory=list)  # 存储分割结果
    buffer: str = ""  # 缓存文本流

    is_detecting: bool = False  # 是否在检测
    detect_start_time: float | None = None  # 检测开始时间
    min_seg_size: int = 0  # 当前最小分割大小

    def fill(self, text: str | None):
        self.in_queue.put_nowait(text)

    def fire(self, finished=False):
        if len(self.buffer) > 0:
            self.segmenteds.append(self.buffer)
            self.buffer = ""
            self.mid_queue.put_nowait(None)
            self.is_detecting = False
            self.on_first_segment()

        if not finished:
            self.out_queue.put_nowait(self.get_async_generator())
        else:
            self.out_queue.put_nowait(None)

    async def async_output_stream(self, interval=0.001):
        start_time = time.time()
        while True:
            try:
                generator = self.out_queue.get_nowait()
                if generator is None:
                    return
                start_time = time.time()
                yield generator
            except asyncio.QueueEmpty:
                if (time.time() - start_time) > self.config.max_stream_time:
                    return
                await asyncio.sleep(interval)

    async def get_async_generator(self):
        while True:
            item = await self.mid_queue.get()
            if item is None:
                return
            yield item

    async def step(self):
        self.on_start()
        while True:
            text = await self.in_queue.get()
            if text is None:
                return
            self.source.append(text)
            text = re.sub(r"\s+", " ", text)
            if len(text) > 0:
                for char in text:  # 以字符粒度分割
                    self.buffer += char  # 累积缓存
                    self.mid_queue.put_nowait(char)
                    can_detection, is_waiting_timeout = self.check_conditions()
                    yield can_detection, is_waiting_timeout
                    # self.postprocessing()

    def on_start(self):
        self.fire()
        self.min_seg_size = self.config.first_min_seg_size

    def on_first_segment(self):
        self.min_seg_size = self.config.min_seg_size
        setattr(self, "on_first_segment", lambda: None)

    def check_conditions(self):
        if self.is_detecting:
            is_waiting_timeout = (
                time.time() - self.detect_start_time
            ) > self.config.max_waiting_time
            return True, is_waiting_timeout

        can_detection = len(self.buffer) >= self.min_seg_size
        if can_detection:
            self.is_detecting = True
            self.detect_start_time = time.time()
        return can_detection, False

    def postprocessing(self):
        pass


class SentenceSegmentationPipeline:
    def __init__(
        self,
        sentence_segmenter=None,
        phrase_segmenter=None,
        segment_aux_suffix="####",
        **kwargs,
    ):
        self.config = SegmentationConfig(**kwargs)
        self.status = SegmentationStatus(config=self.config)

        assert any([sentence_segmenter is not None, phrase_segmenter is not None])
        self.sentence_segmenter = sentence_segmenter
        self.phrase_segmenter = phrase_segmenter
        self.segment_aux_suffix = segment_aux_suffix

    def reset_status(self):
        self.status = SegmentationStatus(config=self.config)

    def __detect_breakpoint__(self):
        target_text = self.status.buffer + self.segment_aux_suffix
        if self.sentence_segmenter is not None:  # 句子检测
            tail = self.sentence_segmenter(target_text)[-1]
        if self.phrase_segmenter is not None:  # 短语检测
            tail = self.phrase_segmenter(target_text)[-1]
        return tail == self.segment_aux_suffix

    async def segment(self):
        async for can_detection, is_waiting_timeout in self.status.step():
            if can_detection and (is_waiting_timeout or self.__detect_breakpoint__()):
                self.status.fire()
        self.status.fire(True)

    def add_text(self, text: str | None):
        """Input `None` to indicate the end."""
        self.status.fill(text)

    def get_out_stream(self, interval: float = 0.001):
        return self.status.async_output_stream(interval)

    def get_segmenteds(self):
        return self.status.segmenteds
