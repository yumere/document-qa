import argparse
from tqdm import tqdm

import numpy as np
import ujson as json
import pickle
import unicodedata
from itertools import islice
from os import mkdir
from os.path import join, exists
from typing import List, Optional, Dict

from docqa.config import CORPUS_DIR, TRIVIA_QA, TRIVIA_QA_UNFILTERED
from docqa.configurable import Configurable
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.hotpotqa.answer_detection import FastNormalizedAnswerDetector
from docqa.triviaqa.evidence_corpus import TriviaQaEvidenceCorpusTxt
from docqa.triviaqa.read_data import iter_trivia_question, TriviaQaQuestion
from docqa.utils import ResourceLoader

from docqa.data_processing.multi_paragraph_qa import MultiParagraphQuestion, DocumentParagraph
from docqa.data_processing.preprocessed_corpus import FilteredData
from docqa.data_processing.text_utils import NltkAndPunctTokenizer


class HotpotQaSpanDataset(Configurable):
    def __init__(self, corpus_name):
        self.corpus_name = corpus_name
        self.dir = join(CORPUS_DIR, self.corpus_name)
        self.tokenizer = NltkAndPunctTokenizer()
        self.detector = FastNormalizedAnswerDetector()

        self._train, self._raw_train = list(), None
        self._dev, self._raw_dev = list(), None

        with open(join(self.dir, "hotpot_train_v1.json"), "rb") as f_train:
            self._raw_train = json.load(f_train)
            # self._raw_train = json.load(f_train)[:2000]

        with open(join(self.dir, "hotpot_dev_distractor_v1.json"), "rb") as f_dev:
            # self._raw_dev = json.load(f_dev)[:200]
            self._raw_dev = json.load(f_dev)


            # with open(join(self.dir, "file_map.json"), "r") as f:
            #     file_map = json.load(f)
            # for k, v in file_map.items():
            #     file_map[k] = unicodedata.normalize("NFD", v)
            # self.evidence = TriviaQaEvidenceCorpusTxt(file_map)

    # def get_train(self) -> List[Dict]:
    #     with open(join(self.dir, "hotpot_train_v1.json"), "rb") as f:
    #         return json.load(f)

    def get_train(self) -> List[Dict]:
        return self._train

    # def get_dev(self) -> List[Dict]:
    #     with open(join(self.dir, "hotpot_dev_distractor_v1.json"), "rb") as f:
    #         return json.load(f)

    def get_dev(self) -> List[Dict]:
        return self._dev

    # def get_test(self) -> List[Dict]:
    #     with open(join(self.dir, "test.pkl"), "rb") as f:
    #         return pickle.load(f)

    def get_resource_loader(self):
        return ResourceLoader()

    def preprocess(self):
        dataset = {'train': self._raw_train, 'dev': self._raw_dev}

        for d in dataset:
            print("preprocess for {}".format(d))
            for question in tqdm(dataset[d]):
                if question['type'] == 'bridge':
                    question_id = question['_id']
                    question_text = self.tokenizer.tokenize_paragraph_flat(question['question'])
                    answer_text = [question['answer']]
                    supporting_facts = question['supporting_facts']
                    paragraphs = self._get_document_paragraph(question['context'], answer_text,
                                                              answer_para=supporting_facts)


                    if d == 'train':
                        self._train.append(MultiParagraphQuestion(question_id, question_text, answer_text, paragraphs))
                    elif d == 'dev':
                        self._dev.append(MultiParagraphQuestion(question_id, question_text, answer_text, paragraphs))

        self._train = FilteredData(self._train, len(self._train))
        self._dev = FilteredData(self._dev, len(self._dev))

    def _get_document_paragraph(self, documents, answers, answer_para=None):
        paragraphs = list()

        tokenized_aliases = [self.tokenizer.tokenize_paragraph_flat(x) for x in answers]
        self.detector.set_question(tokenized_aliases)

        if answer_para is not None:
            answer_para_title = [p[0] for p in answer_para]
            documents = [d for d in documents if d[0] in answer_para_title]

        if len(documents) < 2:
            print("ERROR")

        for d in documents:
            title, paragraph = d[0], d[1]
            text_paragraph = " ".join(paragraph)
            text = self.tokenizer.tokenize_paragraph_flat(text_paragraph)

            start, end = 0, len(text) - 1
            rank = -1

            spans = []
            offset = 0
            for s, e in self.detector.any_found([text]):
                spans.append((s+offset, e+offset-1))

            if len(spans) == 0:
                answer_spans = np.zeros((0, 2), dtype=np.int32)
            else:
                answer_spans = np.array(spans, dtype=np.int32)

            paragraphs.append(DocumentParagraph(title, start, end, rank, answer_spans, text))

        return paragraphs

    @property
    def name(self):
        return self.corpus_name
