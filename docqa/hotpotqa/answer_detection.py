import re
import string

import numpy as np
from tqdm import tqdm
from typing import List

from docqa.triviaqa.read_data import TriviaQaQuestion
from docqa.triviaqa.trivia_qa_eval import normalize_answer, f1_score
from docqa.utils import flatten_iterable, split


class NormalizedAnswerDetector(object):
    """ Try to labels tokens sequences, such that the extracted sequence would be evaluated as 100% correct
    by the official trivia-qa evaluation script """
    def __init__(self):
        self.answer_tokens = None

    def set_question(self, normalized_aliases):
        self.answer_tokens = normalized_aliases

    def any_found(self, para):
        words = [normalize_answer(w) for w in flatten_iterable(para)]
        occurances = []
        for answer_ix, answer in enumerate(self.answer_tokens):
            answer = [normalize_answer(w) for w in answer]

            word_starts = [i for i, w in enumerate(words) if answer[0] == w]
            n_tokens = len(answer)
            for start in word_starts:
                end = start + 1
                ans_token = 1
                while ans_token < n_tokens and end < len(words):
                    next = words[end]
                    if answer[ans_token] == next:
                        ans_token += 1
                        end += 1
                    elif next == "":
                        end += 1
                    else:
                        break
                if n_tokens == ans_token:
                    occurances.append((start, end))
        return list(set(occurances))


class FastNormalizedAnswerDetector(object):
    """ almost twice as fast and very,very close to NormalizedAnswerDetector's output """

    def __init__(self):
        # These come from the TrivaQA official evaluation script
        self.skip = {"a", "an", "the", ""}
        self.strip = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_", "."])

        self.answer_tokens = None
        self.temp_answer = None
        self.temp_word = list()

    def set_question(self, normalized_aliases):
        self.answer_tokens = normalized_aliases

    def any_found(self, para):
        # Normalize the paragraph
        self.temp_word = list()
        words = [w.lower().strip(self.strip) for w in flatten_iterable(para)]
        occurances = []
        for answer_ix, answer in enumerate(self.answer_tokens):
            answer = [w.lower().strip(self.strip) for w in answer]
            self.temp_answer = answer
            self.temp_word.append(words)
            # Locations where the first word occurs
            word_starts = [i for i, w in enumerate(words) if answer[0] == w]
            n_tokens = len(answer)

            # Advance forward until we find all the words, skipping over articles
            for start in word_starts:
                end = start + 1
                ans_token = 1
                while ans_token < n_tokens and end < len(words):
                    next = words[end]
                    if answer[ans_token] == next:
                        ans_token += 1
                        end += 1
                    elif next in self.skip:
                        end += 1
                    else:
                        break
                if n_tokens == ans_token:
                    occurances.append((start, end))
        return list(set(occurances))