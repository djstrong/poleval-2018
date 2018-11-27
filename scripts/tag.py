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


def tag_file(input_name='data/test.tsv',
             output_name='data/out.tsv',
             models_pattern='data/models/*/best-model.pt'):
    taggers = list()
    for file in glob.glob(models_pattern):
        taggers.append(SequenceTagger.load_from_file(file))

    lines=0
    with open(input_name) as input:
        for l in input:
            lines+=1

    with open(input_name) as input, open(output_name, 'w') as output:

        #create batches
        batch=[]
        mini_batch_size=4
        for line in tqdm.tqdm(input, total=lines):
            s = Sentence(line.rstrip())

            batch.append(s)
            if len(batch)==mini_batch_size:
                tag_batch(taggers, batch, output, mini_batch_size)
                batch=[]
        if batch:
            tag_batch(taggers, batch, output, mini_batch_size)

def tag_batch(taggers, sentences, output, mini_batch_size):
    # embed once, needs modified branch of flair
    taggers[0].embeddings.embed(sentences)

    results = []
    for tagger in taggers:
        tagger_results = []
        tagger.predict(sentences, mini_batch_size=mini_batch_size)
        for s in sentences:
            tagger_results.append(pop_results(s))
        results.append(tagger_results)

    for x in zip(*results):
        z=[]
        for y in x:
            z.extend(y)
        output.write(' '.join(z) + '\n')
        output.flush()



if __name__ == "__main__":
    fire.Fire(tag_file)
