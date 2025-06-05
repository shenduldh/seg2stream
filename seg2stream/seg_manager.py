import queue
import asyncio
from dataclasses import dataclass
from typing import List, Callable, Dict
from multiprocessing import Manager, Process

from .seg2stream import (
    SegmentationPipeline as SegSent2StreamPipeline,
    SegmentationConfig as SegSent2StreamConfig,
)
from .seg2generator import (
    SegmentationPipeline as SegSent2GeneratorPipeline,
    SegmentationConfig as SegSent2GeneratorConfig,
)
from .segmenters import get_sentence_segmenter


@dataclass
class SegmentationTask:
    def __init__(
        self,
        id: str,
        pipeline: SegSent2StreamPipeline | SegSent2GeneratorPipeline,
        out_queue: queue.Queue,
    ):
        async def process_output():
            async for output in pipeline.output_stream():
                out_queue.put_nowait((id, output))
            out_queue.put_nowait((id, None))

        self.pipeline = pipeline
        self.future = asyncio.gather(process_output(), pipeline.segment())

    def send(self, text: str | None):
        self.pipeline.fill(text)


class SegmentationManager:
    def __init__(
        self,
        seg_config: SegSent2StreamConfig | SegSent2GeneratorConfig,
        segmenters: List[Callable[[str], str]] | None = None,
    ):
        self.seg_config = seg_config

        if isinstance(seg_config, SegSent2StreamConfig):
            self.seg_pipeline_class = SegSent2StreamPipeline
        elif isinstance(seg_config, SegSent2GeneratorConfig):
            self.seg_pipeline_class = SegSent2GeneratorPipeline

        self.segmenters = segmenters if segmenters else [get_sentence_segmenter()]
        self.manager = Manager()
        self.in_queue = self.manager.Queue()
        self.out_queue = self.manager.Queue()

    def segmentation_process(self):
        tasks: Dict[str, SegmentationTask] = {}

        async def main():
            while True:
                try:
                    id, text = self.in_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.001)
                    continue
                if id is None:
                    break
                if id not in tasks:
                    tasks[id] = SegmentationTask(
                        id=id,
                        pipeline=self.seg_pipeline_class(
                            config=self.seg_config, segmenters=self.segmenters
                        ),
                        out_queue=self.out_queue,
                    )
                tasks[id].send(text)

            for s in tasks.values():
                await s.future

        asyncio.run(main())

    def start(self):
        self.seg_process = Process(target=self.segmentation_process, name="segmenting")
        self.seg_process.start()

    def add_text(self, id: str | None, text: str | None):
        self.in_queue.put_nowait((id, text))

    async def get_async_output(self):
        while True:
            try:
                id, output = self.out_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue
            if id is None:
                return
            yield (id, output)

    def get_output(self):
        while True:
            id, output = self.out_queue.get()
            if id is None:
                return
            yield (id, output)

    def close(self):
        self.in_queue.put((None, None))
        self.seg_process.join()
        self.out_queue.put((None, None))
