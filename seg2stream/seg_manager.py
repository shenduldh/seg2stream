from dataclasses import dataclass, field
import asyncio
from typing import List, Callable, Dict, Coroutine, Any, Literal
from multiprocessing import Manager, Process
import queue
import threading

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
class SegmentationStatus:
    id: str
    pipeline: SegSent2StreamPipeline | SegSent2GeneratorPipeline
    out: queue.Queue
    loop: asyncio.AbstractEventLoop
    futures: List[asyncio.Future] = field(default_factory=list)

    def add_text(self, text: str | None):
        self.pipeline.fill(text)

    def run(self):
        async def process_output():
            async for output in self.pipeline.output_stream():
                self.out.put((self.id, output))

        self.add_future(process_output())
        self.add_future(self.pipeline.segment())

    def add_future(self, coro: Coroutine):
        self.futures.append(asyncio.run_coroutine_threadsafe(coro, self.loop))

    async def run_until_finished(self):
        for f in self.futures:
            if not f.done:
                await f


class SegmentationManager:
    def __init__(
        self,
        seg_config: SegSent2StreamConfig | SegSent2GeneratorConfig,
        on_output: Callable[[str, str], Any],
        output_type: Literal["generator", "string"] = "string",
        segmenters: List[Callable[[str], str]] | None = None,
    ):
        self.seg_config = seg_config

        match output_type:
            case "generator":
                self.seg_pipeline_class = SegSent2GeneratorPipeline
            case "string":
                self.seg_pipeline_class = SegSent2StreamPipeline

        if segmenters is None:
            self.segmenters = [get_sentence_segmenter()]
        else:
            self.segmenters = segmenters

        self.manager = Manager()
        self.in_queue = self.manager.Queue()
        self.out_queue = self.manager.Queue()
        self.output_processing_func = on_output

        self.seg_process = Process(target=self.segmentation_task, name="segmenting")
        self.out_process = Process(target=self.output_processing_task)

    def segmentation_task(self):
        seg_statuses: Dict[str, SegmentationStatus] = {}
        loop = asyncio.new_event_loop()

        def run_event_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=run_event_loop, args=(loop,))
        thread.start()

        while True:
            id, text = self.in_queue.get()
            if id is None:
                break
            if id not in seg_statuses:
                status = SegmentationStatus(
                    id=id,
                    pipeline=self.seg_pipeline_class(
                        config=self.seg_config, segmenters=self.segmenters
                    ),
                    out=self.out_queue,
                    loop=loop,
                )
                seg_statuses[id] = status
                status.run()
            seg_statuses[id].add_text(text)

        async def run_until_all_finished():
            for s in seg_statuses.values():
                await s.run_until_finished()

        asyncio.run(run_until_all_finished())
        loop.call_soon_threadsafe(loop.stop)

    def output_processing_task(self):
        while True:
            id, output = self.out_queue.get()
            if id is None:
                break
            self.output_processing_func(id, output)

    def start(self):
        self.seg_process.start()
        self.out_process.start()

    def add_text(self, id: str | None, text: str | None):
        self.in_queue.put_nowait((id, text))

    def close(self):
        self.in_queue.put((None, None))
        self.seg_process.join()
        self.out_queue.put((None, None))
        self.out_process.join()
