import re
from typing import Literal


class JioNLPSentenceSegmenter(object):
    """Copied from https://github.com/dongrixinyu/JioNLP."""

    def __init__(self):
        self.puncs_fine = None

    def _prepare(self):
        self.puncs_fine = {
            "……",
            "\r\n",
            "，",
            "。",
            ";",
            "；",
            "…",
            "！",
            "!",
            "?",
            "？",
            "\r",
            "\n",
            "“",
            "”",
            "‘",
            "’",
            "：",
        }
        self.puncs_coarse = {"。", "！", "？", "\n", "“", "”", "‘", "’"}
        self.front_quote_list = {"“", "‘"}
        self.back_quote_list = {"”", "’"}
        self.puncs_coarse_ptn = re.compile("([。“”！？\n])")
        self.puncs_fine_ptn = re.compile("([，：。;“”；…！!?？\r\n])")

    def __call__(self, text, criterion="coarse"):
        if self.puncs_fine is None:
            self._prepare()
        if criterion == "coarse":
            tmp_list = self.puncs_coarse_ptn.split(text)
        elif criterion == "fine":
            tmp_list = self.puncs_fine_ptn.split(text)
        else:
            raise ValueError("The parameter `criterion` must be " "`coarse` or `fine`.")

        final_sentences = []
        quote_flag = False

        for sen in tmp_list:
            if sen == "":
                continue

            if criterion == "coarse":
                if sen in self.puncs_coarse:
                    if len(final_sentences) == 0:  # 即文本起始字符是标点
                        if sen in self.front_quote_list:  # 起始字符是前引号
                            quote_flag = True
                        final_sentences.append(sen)
                        continue

                    # 以下确保当前标点前必然有文本且非空字符串
                    # 前引号较为特殊，其后的一句需要与前引号合并，而不与其前一句合并
                    if sen in self.front_quote_list:
                        if final_sentences[-1][-1] in self.puncs_coarse:
                            # 前引号前有标点如句号，引号等：另起一句，与此合并
                            final_sentences.append(sen)
                        else:
                            # 前引号之前无任何终止标点，与前一句合并
                            final_sentences[-1] = final_sentences[-1] + sen
                        quote_flag = True
                    else:  # 普通,非前引号，则与前一句合并
                        final_sentences[-1] = final_sentences[-1] + sen
                    continue

            elif criterion == "fine":
                if sen in self.puncs_fine:
                    if len(final_sentences) == 0:  # 即文本起始字符是标点
                        if sen in self.front_quote_list:  # 起始字符是前引号
                            quote_flag = True
                        final_sentences.append(sen)
                        continue

                    # 以下确保当前标点前必然有文本且非空字符串
                    # 前引号较为特殊，其后的一句需要与前引号合并，而不与其前一句合并
                    if sen in self.front_quote_list:
                        if final_sentences[-1][-1] in self.puncs_fine:
                            # 前引号前有标点如句号，引号等：另起一句，与此合并
                            final_sentences.append(sen)
                        else:
                            # 前引号之前无任何终止标点，与前一句合并
                            final_sentences[-1] = final_sentences[-1] + sen
                        quote_flag = True
                    else:  # 普通,非前引号，则与前一句合并
                        final_sentences[-1] = final_sentences[-1] + sen
                    continue

            if len(final_sentences) == 0:  # 起始句且非标点
                final_sentences.append(sen)
                continue

            if quote_flag:  # 当前句子之前有前引号，须与前引号合并
                final_sentences[-1] = final_sentences[-1] + sen
                quote_flag = False
            else:
                if final_sentences[-1][-1] in self.back_quote_list:
                    # 此句之前是后引号，需要考察有无其他终止符，用来判断是否和前句合并
                    if len(final_sentences[-1]) <= 1:
                        # 前句仅一个字符。后引号，则合并
                        final_sentences[-1] = final_sentences[-1] + sen
                    else:  # 前句有多个字符，
                        if criterion == "fine":
                            if final_sentences[-1][-2] in self.puncs_fine:
                                # 有逗号等，则需要另起一句，该判断不合语文规范，但须考虑此情况
                                final_sentences.append(sen)
                            else:  # 前句无句号，则需要与前句合并
                                final_sentences[-1] = final_sentences[-1] + sen

                        elif criterion == "coarse":
                            if final_sentences[-1][-2] in self.puncs_coarse:
                                # 有句号，则需要另起一句
                                final_sentences.append(sen)
                            else:  # 前句无句号，则需要与前句合并
                                final_sentences[-1] = final_sentences[-1] + sen
                else:
                    final_sentences.append(sen)

        return final_sentences


def get_sentence_segmenter(
    method_name: Literal["jionlp", "pysbd", "stanza"] = "jionlp",
):
    try:
        match method_name:
            case "pysbd":
                import pysbd

                segmenter = pysbd.Segmenter(language="zh", clean=False)
                return lambda text: segmenter.segment(text)
            case "stanza":
                import stanza

                stanza.download("zh")
                segmenter = stanza.Pipeline(
                    "zh", processors="tokenize", download_method=None
                )
                return lambda text: [s.text for s in segmenter(text).sentences]
            case _:
                raise
    except:
        segmenter = JioNLPSentenceSegmenter()
        return lambda text: segmenter(text, criterion="coarse")


def get_phrase_segmenter(
    method_name: Literal["jionlp", "regex"] = "regex",
):
    match method_name:
        case "jionlp":
            segmenter = JioNLPSentenceSegmenter()
            return lambda text: segmenter(text, criterion="fine")
        case _:
            seg_puncts = re.escape("。？！，；：.?!,;:")
            patern = re.compile(rf"^[\S\s]*?[{seg_puncts}]")

            def func(text):
                phrases = []
                while True:
                    match = patern.search(text)
                    if match is None:
                        break
                    phrases.append(match.group())
                    text = text[match.span()[1] :]
                if len(text) > 0:
                    phrases.append(text)
                return phrases

            return func


def get_rhythm_segmenter(model_path):
    try:
        from paddlespeech.t2s.frontend.zh_frontend import RhyPredictor

        detector = RhyPredictor(model_path)

        def func(text):
            rhythms = []
            pred = detector.get_prediction(text)
            bounds = re.sub("[%`~]", "", pred).split("$")[:-1]
            pi, last_s = 0, 0
            for idx, rhy in enumerate(bounds):
                for ci, char in enumerate(rhy):
                    pi = text.find(char, pi)
                    if idx >= 1 and ci == 0:
                        rhythms.append(text[last_s:pi])
                        last_s = pi
                    pi += 1
            rhythms.append(text[last_s:])
            return rhythms

        return func
    except:
        return None
