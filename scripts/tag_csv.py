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

def split_long_text(text, MAX=1000):
    index=0
    while index<len(text):
        max_fragment=text[index:index+MAX]
        try:
            division = max_fragment.rindex('.')+1
            try:
                if max_fragment[division]==' ':
                    division+=1
            except IndexError:
                pass
        except ValueError:
            try:
                division = max_fragment.rindex(' ')+1
            except ValueError:
                division = len(max_fragment)
        yield max_fragment[:division]
        index+=division

def samples_generator_sorted(path, max_text_legth=10000):
    data=[]
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)

    MAX=max_text_legth
    datas=sorted(data, key=lambda x: len(x[3]), reverse=True)
    print('Longest text', len(datas[0][3]))
    for row in datas:

        id = row[0]
        print(id)
        text_id = row[1]
        sequence = row[2]
        text = row[3]

        if len(text)>MAX:
            for fragment in split_long_text(text, MAX):
                s = Sentence(fragment, use_tokenizer='toki')
                s.id=id
                s.text_id=text_id
                s.sequence=sequence
                s.ner=[]
                s.length=len(fragment)

                yield s
  
        else:
            s = Sentence(text, use_tokenizer='toki')
            s.id=id
            s.text_id=text_id
            s.sequence=sequence
            s.ner=[]
            s.length = len(text)

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

    if len(sys.argv)!=2:
        print("provide path to CSV")

    input_path=sys.argv[1]

    csv.field_size_limit(sys.maxsize)
    models_pattern = 'data/models/*/final-model.pt'
    taggers = list()
    for file in glob.glob(models_pattern):
        taggers.append(SequenceTagger.load_from_file(file))

    mini_batch_size = 16
    with jsonlines.open(input_path+'.jsonl', mode='w', compact=True) as writer:
        for batch in tag(samples_generator_sorted(input_path), mini_batch_size):
            for s in batch:
                writer.write({'ner': s.ner,
                              'id':s.id,
                              'text_id':s.text_id,
                              'sequence':s.sequence,
                              'length':s.length
                              })
            del batch
