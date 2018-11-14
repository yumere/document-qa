import argparse
import pickle
import ujson as json
from multiprocessing import Pool
from os.path import join
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from docqa.config import CORPUS_DIR
from docqa.configurable import Configurable
from docqa.data_processing.multi_paragraph_qa import MultiParagraphQuestion, DocumentParagraph
from docqa.data_processing.preprocessed_corpus import FilteredData
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.hotpotqa.answer_detection import FastNormalizedAnswerDetector
from docqa.utils import ResourceLoader
from docqa.utils import bcolors
from docqa.utils import split


class HotpotQaSpanDataset(Configurable):
    def __init__(self, corpus_name):
        self.corpus_name = corpus_name
        self.dir = join(CORPUS_DIR, self.corpus_name)
        self.tokenizer = NltkAndPunctTokenizer()
        self.detector = FastNormalizedAnswerDetector()

        self._train, self._raw_train = list(), None
        self._dev, self._raw_dev = list(), None

        self.missed_answer = 0

    def get_train(self) -> List[Dict]:
        return self._train

    def get_dev(self) -> List[Dict]:
        return self._dev

    def get_resource_loader(self):
        return ResourceLoader()

    @property
    def name(self):
        return self.corpus_name

    @staticmethod
    def _build_question(questions, tokenizer, detector):
        questions_chunk = []
        for question in tqdm(questions, desc='Chunk.', ncols=70):
            question_id = question['_id']
            question_text = tokenizer.tokenize_paragraph_flat(question['question'])
            answer_text = [question['answer']]
            supporting_facts = question['supporting_facts']

            paragraphs = []
            tokenized_aliases = [tokenizer.tokenize_paragraph_flat(x) for x in answer_text]
            detector.set_question(tokenized_aliases)

            answer_type = 2
            if answer_text[0].lower() == 'yes':
                answer_type = 0
            elif answer_text[0].lower() == 'no':
                answer_type = 1
            
            # golden paragraph 만 고르는 과정
            if supporting_facts is not None:
                answer_para_title = [p[0] for p in supporting_facts]
                documents = [d for d in question['context'] if d[0] in [s[0] for s in supporting_facts]]

            if len(documents) < 2:
                tqdm.write(bcolors.WARNING + "The number of golden paragraph is not two" + bcolors.ENDC)

            get_answer_span = False
            for i, d in enumerate(documents):
                title, paragraph = d[0], d[1]
                text_paragraph = " ".join(paragraph)
                text = tokenizer.tokenize_paragraph_flat(text_paragraph)

                start, end = 0, len(text) - 1
                rank = -1
                
                # answer가 span 일 경우
                if answer_type == 2:
                    spans = []
                    offset = 0
                    for s, e in detector.any_found([text]):
                        spans.append((s+offset, e+offset - 1))

                    if len(spans) == 0:
                        answer_spans = np.zeros((0, 2), dtype=np.int32)
                    else:
                        get_answer_span = True
                        answer_spans = np.array(spans, dtype=np.int32)
                # answer가 yes/no 일 경우
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
                paragraphs.append(DocumentParagraph(title, start, end, rank, answer_spans, text, answer_yes_no))

            if paragraphs is not None:
                questions_chunk.append(MultiParagraphQuestion(question_id, question_text, answer_text, paragraphs))

        return questions_chunk

    @classmethod
    def _build_dataset(cls, corpus_name, n_processes, train_file: str, dev_file: str):
        hotpotqa = cls(corpus_name=corpus_name)

        with open(join(hotpotqa.dir, train_file), "rt") as f_train:
            _raw_train = json.load(f_train)

        with open(join(hotpotqa.dir, dev_file), "rt") as f_dev:
            _raw_dev = json.load(f_dev)

        dataset = {'train': _raw_train, 'dev': _raw_dev}
        for d in dataset:
            with Pool(n_processes) as pool, tqdm(total=len(dataset[d]), desc=d, ncols=70) as pbar:
                tqdm.write(bcolors.OKBLUE + "[+] Preprocess for {} set".format(d) + bcolors.ENDC)
                missed_answer = 0
                chunks = split(dataset[d], n_processes)

                for questions in pool.starmap(hotpotqa._build_question, [[c, hotpotqa.tokenizer, hotpotqa.detector] for c in chunks]):
                    pbar.update(len(questions))
                    if d == 'train':
                        hotpotqa._train += questions
                    elif d == 'dev':
                        hotpotqa._dev += questions
        hotpotqa._train = FilteredData(hotpotqa._train, len(hotpotqa._train))
        hotpotqa._dev = FilteredData(hotpotqa._dev, len(hotpotqa._dev))

        return hotpotqa

    def save(self):
        with open(join(self.dir, "train.pkl"), "wb") as f:
            f.write(pickle.dumps(self._train))
        with open(join(self.dir, "dev.pkl"), "wb") as f:
            f.write(pickle.dumps(self._dev))

        tqdm.write(bcolors.OKGREEN + "[+] saved at {}".format(self.dir) + bcolors.ENDC)

    @classmethod
    def load(cls, corpus_name):
        hotpot = cls(corpus_name=corpus_name)

        hotpot._train = pickle.load(open(join(hotpot.dir, "train.pkl"), "rb"))
        hotpot._dev = pickle.load(open(join(hotpot.dir, "dev.pkl"), "rb"))
        return hotpot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_processes', type=int, default=10, help="The number of processes for multi-processing")
    parser.add_argument('--save', default=False, action='store_true')
    args = parser.parse_args()

    hotpotqa_dataset = HotpotQaSpanDataset._build_dataset('hotpotqa', args.n_processes, 'hotpot_train_v1.json', 'hotpot_dev_distractor_v1.json')
    if args.save:
        hotpotqa_dataset.save()
