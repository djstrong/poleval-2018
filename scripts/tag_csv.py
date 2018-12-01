#!/usr/bin/env python3

"""
Tags input using all the trained models
and stores results in a single file
in our internal standard.
"""

import glob

import fire
from flair.data import Sentence
from flair.models import SequenceTagger
import tqdm


def pop_results(s):
    opened = ''
    openedAt = -1
    res = list()
    for i, t in enumerate(s):
        g = t.get_tag('ner').value
        tp = g.split('-')[-1]
        if (tp != opened or 'B-' in g) and opened != '':
            tmp = list()
            for j in range(openedAt, i):
                tmp.append(j + 1)
            res.append(opened + ':' + ','.join([str(x) for x in tmp]))
            opened = ''
            openedAt = -1
        if g not in ('', 'O') and 'B-' in g:
            opened = tp
            openedAt = i
        t.add_tag('ner', '')
    return res


def tag_batch(taggers, sentences, mini_batch_size):
    # embed once, needs modified branch of flair
    taggers[0].embeddings.embed(sentences)

    for tagger in taggers:
        tagger.predict(sentences, mini_batch_size=mini_batch_size)

        for s in sentences:
            s.ner.extend([span.to_dict() for span in s.get_spans('ner')])


    return sentences

def samples_generator(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            id = row[0]
            print(id)
            text_id = row[1]
            sequence = row[2]
            text = row[3]

            s = Sentence(text, use_tokenizer='toki')
            s.id=id
            s.text_id=text_id
            s.sequence=sequence
            s.ner=[]

            yield s

def tag(generator, mini_batch_size = 4):
    batch = []
    for sentence in generator:

        batch.append(sentence)
        if len(batch) == mini_batch_size:
            yield tag_batch(taggers, batch, mini_batch_size)
            batch = []
    if batch:
        yield tag_batch(taggers, batch, mini_batch_size)

if __name__ == "__main__":
    import csv
    import sys
    import jsonlines

    input_path=sys.argv[1]

    csv.field_size_limit(sys.maxsize)
    models_pattern = 'data/models/*/final-model.pt'
    taggers = list()
    for file in glob.glob(models_pattern):
        taggers.append(SequenceTagger.load_from_file(file))

    mini_batch_size = 32
    with jsonlines.open(input_path+'.jsonl', mode='w', compact=True) as writer:
        for batch in tag(samples_generator(input_path), mini_batch_size):
            for s in batch:
                writer.write({'ner': s.ner,
                              'id':s.id,
                              'text_id':s.text_id,
                              'sequence':s.sequence
                              })