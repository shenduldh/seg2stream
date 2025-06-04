from dataclasses import dataclass, field
import asyncio
from asyncio import Queue
import time
import re
from typing import List, Callable


@dataclass
class SegmentationConfig:
    segmentation_suffix: str
    ##############
    first_max_accu_time: float
    max_accu_time: float
    first_max_buffer_size: int
    max_buffer_size: int
    max_waiting_time: float  # 等待超时时间
    max_stream_time: float  # 流式超时时间
    first_min_seg_size: int  # 首次最小分割大小
    min_seg_size: int  # 最小分割大小
    max_seg_size: int  # 最大分割大小
    loose_steps: int  # 松弛步长
    loose_size: int  # 松弛大小
    fade_in_out_time: float  # 淡入淡出时间
    seconds_per_word: float  # 每词说话时长
    # first_chunk_synthesis_time: float
    # first_chunk_transfer_time: float


@dataclass
class SegmentationPipeline:
    """### 动态调整累积时间 \n
    - 假设：生成首块后，后续播放连续 \n
    - 预留时间 = 分割成功时间 (固定为上一次分割成功时间 * 2) + 合成首块时间 (动态计算) + 传输首块时间 (目前固定) + 淡入淡出时间 (固定) \n
    - 累积时间 = 下一次分割开始 - 当前分割完成 \n
    - 总时间 = 累积时间 + 预留时间 = 合成首块时间 (动态计算) + 传输首块时间 (目前固定) + 完整语音时长 (动态计算) \n
    - 分割策略：累积时间 ==> 开始计算超时时间 ==> 循环 (最多句子检测 -> 最多短语检测 -> 最多韵律检测) 直至超时 ==> 返回剩余缓存
    """

    config: SegmentationConfig  # 固定配置
    segmenters: List[Callable[[str], str]]  # 分割方法

    in_queue: Queue = field(default_factory=Queue)  # 接收外部输入的文本流
    out_queue: Queue = field(default_factory=Queue)  # 输出分割结果到外部
    source: List[str] = field(default_factory=list)  # 存储原始输入的文本流
    segmenteds: List[str] = field(default_factory=list)  # 存储分割结果
    last_combined: str = ""  # 临时保存未满足条件的分割结果
    buffer: str = ""  # 缓存输入的文本流
    is_last_segmented: bool = False  # 用于判断最近是否存在分割

    # 用于触发分割条件
    is_accumulating: bool = True  # 是否正在进行累积
    accu_start_time: float | None = None  # 累积开始时间
    max_buffer_size: int = 0  # 当前最大缓存大小
    max_accu_time: float = 0.0  # 当前最大累积时间

    # 用于计算分割时长和超时时间
    seg_start_time: float | None = None  # 分割开始时间
    all_seg_time: List[float] = field(default_factory=list)  # 保存最近的分割完成时间

    # 用于限制分割结果
    min_seg_size: int = 0  # 当前最小分割大小
    num_consec_splits: int = 0  # 连续未分割成功的次数

    def fill(self, text: str | None):
        self.in_queue.put_nowait(text)

    def fire(self, uncombined_segmenteds: list[str], forced=False, end=False):
        while len(uncombined_segmenteds) > 0:
            seg = uncombined_segmenteds.pop(0)
            if len(seg) > 0:
                striped = seg.strip()
                lc_len = len(self.last_combined)
                if lc_len + len(striped) <= self.config.max_seg_size or lc_len == 0:
                    self.last_combined += striped
                    e = re.search(re.escape(seg), self.buffer).end()
                    self.buffer = self.buffer[e:]
                else:
                    uncombined_segmenteds.insert(0, seg)
                    forced = True

            lc_len = len(self.last_combined)
            if lc_len >= self.min_seg_size or (forced and lc_len > 0):
                self.on_first_segment()
                self.is_last_segmented = True
                self.out_queue.put_nowait(self.last_combined)
                self.segmenteds.append(self.last_combined)
                self.last_combined = ""
        if end:
            self.out_queue.put_nowait(None)

    async def output_stream(self, interval=0.001):
        start_time = time.time()
        while True:
            try:
                segmented: str | None = self.out_queue.get_nowait()
                if segmented is None:
                    return
                start_time = time.time()
                yield segmented
            except asyncio.QueueEmpty:
                if (time.time() - start_time) > self.config.max_stream_time:
                    return
                await asyncio.sleep(interval)

    async def step(self):
        self.on_start()
        while True:
            text = await self.in_queue.get()
            if text is None:
                return
            self.source.append(text)
            text = re.sub(r"\s+", " ", text)
            if len(text) > 0:
                for char in text:  # 以字符粒度进行分割
                    self.buffer += char  # 累积缓存
                    can_segment, is_waiting_timeout = self.check_conditions()
                    yield can_segment, is_waiting_timeout
                    self.postprocessing()

    def segment_once(self):
        for segmenter in self.segmenters:
            target_text = self.buffer + self.config.segmentation_suffix
            segmenteds = segmenter(target_text)[:-1]
            self.fire(segmenteds)

    async def segment(self):
        async for can_segment, is_waiting_timeout in self.step():
            if can_segment:
                self.segment_once()
                if is_waiting_timeout:
                    self.fire([self.buffer], forced=True)

        if len(self.buffer) > 0:
            self.segment_once()
        self.fire([self.buffer], forced=True, end=True)

    def on_start(self):
        self.max_accu_time = self.config.first_max_accu_time
        self.max_buffer_size = self.config.first_max_buffer_size
        self.min_seg_size = self.config.first_min_seg_size
        self.accu_start_time = time.time()

    def on_first_segment(self):
        self.max_buffer_size = self.config.max_buffer_size
        self.min_seg_size = self.config.min_seg_size
        setattr(self, "on_first_segment", lambda: None)

    def check_conditions(self):
        if self.is_accumulating:
            # 在累积时，判断是否可以进行分割
            # 是否达到累积时间或累积大小
            can_segment = (
                time.time() - self.accu_start_time
            ) >= self.max_accu_time or len(self.buffer) > self.max_buffer_size
            if can_segment:
                self.is_accumulating = False
                self.seg_start_time = time.time()
                self.num_consec_splits += 1
            return can_segment, False
        else:
            # 调整连续未分割成功的次数
            if self.num_consec_splits > self.config.loose_steps:
                self.min_seg_size = max(0, self.min_seg_size - self.config.loose_size)
                self.num_consec_splits = 0
            self.num_consec_splits += 1
            is_waiting_timeout = (
                time.time() - self.seg_start_time
            ) > self.config.max_waiting_time
            return True, is_waiting_timeout

    def postprocessing(self):
        if self.is_last_segmented:  # 如果最近存在分割
            self.is_last_segmented = False

            # 计算分割时间
            seg_time = time.time() - self.seg_start_time
            self.all_seg_time.append(seg_time)
            if len(self.all_seg_time) > 5:
                self.all_seg_time.pop(0)

            # 调整累积时间
            # https://speakingtimecalculator.com
            num_words = len(self.segmenteds[-1])
            num_words -= num_words // 10  # 字数 = 字符数 - 标点数
            full_duration = num_words * self.config.seconds_per_word
            mean_seg_time = sum(self.all_seg_time) / len(self.all_seg_time)
            self.max_accu_time = max(
                self.max_accu_time,
                (full_duration - mean_seg_time * 2 - self.config.fade_in_out_time),
            )

            self.num_consec_splits = 0
            self.is_accumulating = True
            self.accu_start_time = time.time()
