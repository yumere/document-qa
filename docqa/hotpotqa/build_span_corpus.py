import numpy as np
import ujson as json
from os.path import join
from tqdm import tqdm
from typing import List, Dict

from docqa.config import CORPUS_DIR
from docqa.configurable import Configurable
from docqa.data_processing.multi_paragraph_qa import MultiParagraphQuestion, DocumentParagraph
from docqa.data_processing.preprocessed_corpus import FilteredData
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.hotpotqa.answer_detection import FastNormalizedAnswerDetector
from docqa.utils import ResourceLoader
from docqa.utils import bcolors


class HotpotQaSpanDataset(Configurable):
    def __init__(self, corpus_name):
        self.corpus_name = corpus_name
        self.dir = join(CORPUS_DIR, self.corpus_name)
        self.tokenizer = NltkAndPunctTokenizer()
        self.detector = FastNormalizedAnswerDetector()
        # self.detector = NormalizedAnswerDetector()

        self._train, self._raw_train = list(), None
        self._dev, self._raw_dev = list(), None

        self.missed_answer = 0

        with open(join(self.dir, "hotpot_train_v1.json"), "rb") as f_train:
            self._raw_train = json.load(f_train)

        with open(join(self.dir, "hotpot_dev_distractor_v1.json"), "rb") as f_dev:
            self._raw_dev = json.load(f_dev)

    def get_train(self) -> List[Dict]:
        return self._train

    def get_dev(self) -> List[Dict]:
        return self._dev

    def get_resource_loader(self):
        return ResourceLoader()

    def preprocess(self):
        dataset = {'train': self._raw_train, 'dev': self._raw_dev}

        for d in dataset:
            tqdm.write(bcolors.OKBLUE + "[+] Preprocess for {} set".format(d) + bcolors.ENDC)
            self.missed_answer = 0
            for question in tqdm(dataset[d], desc=d, ncols=70):
                # if question['type'] == 'bridge':
                # if question['answer'] != 'yes' and question['answer'] != 'no':
                question_id = question['_id']
                question_text = self.tokenizer.tokenize_paragraph_flat(question['question'])
                answer_text = [question['answer']]
                supporting_facts = question['supporting_facts']
                paragraphs = self._get_document_paragraph(question['context'], answer_text, answer_para=supporting_facts)

                if paragraphs is not None:
                    if d == 'train':
                        self._train.append(MultiParagraphQuestion(question_id, question_text, answer_text, paragraphs))
                    elif d == 'dev':
                        self._dev.append(MultiParagraphQuestion(question_id, question_text, answer_text, paragraphs))

            print(bcolors.WARNING + "[*] {} missed data count: {:,}".format(d, self.missed_answer) + bcolors.ENDC)

        print(bcolors.OKBLUE + "[+] Train Size: {:,}".format(len(self._train)) + bcolors.ENDC)
        print(bcolors.OKBLUE + "[+] Dev Size: {:,}".format(len(self._dev)) + bcolors.ENDC)

        self._train = FilteredData(self._train, len(self._train))
        self._dev = FilteredData(self._dev, len(self._dev))

    def _get_document_paragraph(self, documents, answers, answer_para=None):
        paragraphs = list()

        tokenized_aliases = [self.tokenizer.tokenize_paragraph_flat(x) for x in answers]
        self.detector.set_question(tokenized_aliases)

        answer_type = 2
        if answers[0].lower() == 'yes':
            answer_type = 0
        elif answers[0].lower() == 'no':
            answer_type = 1

        if answer_para is not None:
            answer_para_title = [p[0] for p in answer_para]
            documents = [d for d in documents if d[0] in answer_para_title]

        if len(documents) < 2:
            print("ERROR")

        get_answer_span = False

        for i, d in enumerate(documents):
            title, paragraph = d[0], d[1]
            text_paragraph = " ".join(paragraph)
            text = self.tokenizer.tokenize_paragraph_flat(text_paragraph)
            # text = text_paragraph.split()

            start, end = 0, len(text) - 1
            rank = -1

            if answer_type == 2:
                spans = []
                offset = 0
                for s, e in self.detector.any_found([text]):
                    spans.append((s+offset, e+offset-1))

                if len(spans) == 0:
                    answer_spans = np.zeros((0, 2), dtype=np.int32)
                else:
                    get_answer_span = True
                    answer_spans = np.array(spans, dtype=np.int32)
            else:
                get_answer_span = True
                if i == 0:
                    if answer_type == 0:
                        answer_spans = np.array([[0, 0]], dtype=np.int32)
                    else:
                        answer_spans = np.array([[0, 0]], dtype=np.int32)
                else:
                    answer_spans = np.zeros((0, 2), dtype=np.int32)

            answer_yes_no = np.array([answer_type], dtype=np.int32)
            paragraphs.append(DocumentParagraph(title, start, end, rank, answer_spans, text,
                                                answer_yes_no=answer_yes_no))

        if not get_answer_span:
            self.missed_answer += 1
            return None

        return paragraphs

    @property
    def name(self):
        return self.corpus_name
