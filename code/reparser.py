#!/usr/bin/env python
import numpy as np
import string
import sys

filename = sys.argv[1]

data = np.load(filename)


reparsed = []


# parse into lower-case, punctuation-free strings
for i in range(len(abstracts)):
    abstract = ' '.join(abstracts[i].lower().split().split('-'))
    table = str.maketrans({key: None for key in string.punctuation})
    abstract = abstract.translate(table)
    parsed.append(abstract)


np.save('../data/arxiv_parsed.npy',np.array(parsed))
print("parsed abstracts saved, exiting.")
