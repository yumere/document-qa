import argparse
import ujson as json
from itertools import chain
from os import makedirs
from os.path import join, exists
from typing import Set

from tqdm import tqdm

from docqa import config
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.triviaqa.read_data import normalize_wiki_filename
from docqa.utils import group, split, flatten_iterable


def build_tokenized_files(filenames, input_root, output_root, tokenizer, override=True) -> Set[str]:
    """
    For each file in `filenames` loads the text, tokenizes it with `tokenizer, and
    saves the output to the same relative location in `output_root`.
    @:return a set of all the individual words seen
    """
    voc = set()
    for filename in filenames:
        out_file = normalize_wiki_filename(filename[:filename.rfind(".")]) + ".txt"
        out_file = join(output_root, out_file)
        if not override and exists(out_file):
            continue
        with open(join(input_root, filename), "r") as in_file:
            text = in_file.read().strip()
        paras = [x for x in text.split("\n") if len(x) > 0]
        paragraphs = [tokenizer.tokenize_paragraph(x) for x in paras]

        for para in paragraphs:
            for i, sent in enumerate(para):
                voc.update(sent)

        with open(join(output_root, out_file), "w") as in_file:
            in_file.write("\n\n".join("\n".join(" ".join(sent) for sent in para) for para in paragraphs))
    return voc


def _build_vocab(arg):
    return build_vocab(*arg)


def build_vocab(corpus, tokenizer)->Set[str]:
    voc = set()
    paragraphs = [tokenizer.tokenize_paragraph(x) for x in corpus]

    for para in paragraphs:
        for i, sent in enumerate(para):
            voc.update(sent)
    return voc


def build_tokenized_corpus(input_root, tokenizer, output_dir, n_processes=1):
    if not exists(output_dir):
        makedirs(output_dir)

    input_file = json.load(open(input_root, "rt"))
    # context 안에 있는 것만 해야 할 지, question 과 answer를 다 포함해야 할 지 모르겠다.
    # 일단 다 포함해서 하는걸로 진행해 보자.
    question_corpus, answer_corpus, context_corpus = zip(*[(f['question'], f['answer'], f['context']) for f in input_file])
    context_corpus = tuple(chain(*[chain(*[c[1] for c in cc]) for cc in context_corpus]))
    corpus = list(question_corpus + answer_corpus + context_corpus)

    if n_processes == 1:
        voc = build_vocab(tqdm(corpus, ncols=80), tokenizer)
    else:
        voc = set()
        from multiprocessing import Pool
        with Pool(n_processes) as pool:
            chunks = split(corpus, n_processes)
            chunks = flatten_iterable(group(c, 500) for c in chunks)
            pbar = tqdm(total=len(chunks), ncols=80)
            for v in pool.imap_unordered(_build_vocab, [[c, tokenizer] for c in chunks]):
                voc.update(v)
                pbar.update(1)
            pbar.close()

    voc_file = join(output_dir, "vocab.txt")
    with open(voc_file, "w") as f:
        for word in sorted(voc):
            f.write(word)
            f.write("\n")


def main():
    parse = argparse.ArgumentParser("Pre-tokenize the HotPotQA evidence corpus")
    parse.add_argument("-o", "--output_dir", type=str, default=join(config.CORPUS_DIR, "hotpotqa", "evidence"))
    parse.add_argument("-s", "--source", type=str, default=join(config.HOTPOT_QA, "hotpot_train_v1.json"))
    # This is slow, using more processes is recommended
    parse.add_argument("-n", "--n_processes", type=int, default=1, help="Number of processes to use")
    args = parse.parse_args()
    build_tokenized_corpus(args.source, NltkAndPunctTokenizer(), args.output_dir, n_processes=args.n_processes)


if __name__ == '__main__':
    main()
