import random
import time
import asyncio
from seg2stream import SegSent2StreamPipeline, get_sentence_segmenter


test_text = """凌晨三点，林夏被手机铃声惊醒。屏幕上显示“未知号码”，她犹豫着接起，电话那头只有沙沙的雨声。
“喂？”她试探着问。“记得带伞。”一个熟悉的声音轻轻响起，是已故母亲的口吻。
林夏猛地坐起，窗外暴雨如注。她冲到玄关，发现一把陌生的黑伞静静立着——伞柄上刻着她的小名，字迹早已褪色。
第二天，新闻播报昨夜基站故障，全市通信中断四小时。林夏握紧伞柄，雨滴从檐角坠落，像谁的眼泪。"""

sentence_segmenter = get_sentence_segmenter("jionlp")


async def task(id, input_text):
    pipline = SegSent2StreamPipeline(
        sentence_segmenter=sentence_segmenter,
        phrase_segmenter=None,
        ################
        first_max_accu_time=0.1,
        max_accu_time=1.0,
        first_max_buffer_size=20,
        max_buffer_size=50,
        max_waiting_time=2.0,
        max_stream_time=30.0,
        first_min_seg_size=20,
        min_seg_size=50,
        max_seg_size=70,
        loose_steps=4,
        loose_size=10,
        fade_in_out_time=0.2,
        seconds_per_word=0.3,
    )

    async def text_clip_generator(text, max_len=3):
        while len(text) > 0:
            l = random.randint(1, max_len)
            await asyncio.sleep(random.random() * 0.01)
            yield text[:l]
            text = text[l:]

    async def add_text():
        async for text_clip in text_clip_generator(input_text):
            pipline.add_text(text_clip)
        pipline.add_text(None)

    async def async_get_sents():
        index = 0
        s = time.time()
        async for sent in pipline.get_out_stream():
            print(
                f"{'-' * 20} {id}-{index} {'-' * 20}\n"
                f"spent time: {time.time() - s}\n"
                f"{sent}"
            )
            index += 1
            s = time.time()

    await asyncio.gather(pipline.segment(), async_get_sents(), add_text())


async def main():
    await asyncio.gather(*[task(i, test_text) for i in range(1)])


asyncio.run(main())
