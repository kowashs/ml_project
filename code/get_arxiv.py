#!/usr/bin/env python
import numpy as np
from sickle import Sickle
import string
import datetime

# Use with caution; this will pull down ALL papers in the specified arXiv set

URL = 'http://export.arxiv.org/oai2' # define repository URL

sickle = Sickle('http://export.arxiv.org/oai2',max_retries=20)

set_ = 'physics:hep-th'
metadataPrefix = 'oai_dc'

records = sickle.ListRecords(**{'metadataPrefix':metadataPrefix, 'set': set_})


# extract abstracts from records
abstracts = np.array([record.metadata['description'][0] for record in records])

np.save('../data/arxiv_raw_abstracts.npy',abstracts)
print("Raw abstracts saved.")

parsed = []



# parse into lower-case, punctuation-free strings
for i in range(len(abstracts)):
    abstract = ' '.join(abstracts[i].lower().split())
    table = str.maketrans({key: None for key in string.punctuation})
    abstract = abstract.translate(table)
    parsed.append(abstract)


np.save('../data/arxiv_parsed.npy',np.array(parsed))
print("Parsed abstracts saved, exiting.")





