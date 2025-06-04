from dataclasses import dataclass, field
import asyncio
from asyncio import Queue
import time
import re
from typing import List, Callable, Union, AsyncGenerator


@dataclass
class SegmentationConfig:
    segmentation_suffix: str
    ##############
    max_waiting_time: float  # 等待超时时间
    max_stream_time: float  # 流式超时时间
    first_min_seg_size: int
    min_seg_size: int


class SegmentationPipeline:
    def __init__(
        self, config: SegmentationConfig, segmenters: List[Callable[[str], str]]
    ):
        self.config = config  # 固定配置
        self.segmenters = segmenters
        self.reset_status()

    def reset_status(self):
        self.in_queue: Queue = Queue()  # 接收外部输入的文本流
        self.out_queue: Queue = Queue()  # 输出分割结果到外部
        self.mid_queue: Queue = Queue()  # 传递文本到异步生成器
        self.source: List[str] = []  # 存储原始输入的文本流
        self.segmenteds: List[str] = []  # 存储分割结果
        self.buffer: str = ""  # 缓存输入的文本流
        self.is_detecting: bool = False  # 是否在检测
        self.detect_start_time: Union[float | None] = None  # 检测开始时间
        self.min_seg_size: int = 0  # 当前最小分割大小

    def get_segmenteds(self):
        return self.segmenteds

    def fill(self, text: Union[str | None]):
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

    async def output_stream(self, interval=0.001):
        start_time = time.time()
        while True:
            try:
                generator: Union[AsyncGenerator[str, None] | None] = (
                    self.out_queue.get_nowait()
                )
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
                    self.postprocessing()

    def detect_breakpoint(self):
        target_text = self.buffer + self.config.segmentation_suffix
        for segmenter in self.segmenters:
            tail = segmenter(target_text)[-1]
            if tail == self.config.segmentation_suffix:
                return True
        return False

    async def segment(self):
        async for can_detection, is_waiting_timeout in self.step():
            if can_detection and (is_waiting_timeout or self.detect_breakpoint()):
                self.fire()
        self.fire(True)

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
