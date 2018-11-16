#!/usr/bin/env python
import numpy as np
from oaipmh.client import Client
from oaipmh.metadata import MetadataRegistry, oai_dc_reader
import string

# Use with caution; this will pull down ALL papers in the specified arXiv set

URL = 'http://export.arxiv.org/oai2'

registry = MetadataRegistry()
registry.registerReader('oai_dc',oai_dc_reader)
client = Client(URL, registry)
client.updateGranularity()



set_ = 'physics:hep-th'
metadataPrefix = 'oai_dc'

records = client.listRecords(metadataPrefix=metadataPrefix, set=set_)

abstracts = np.array([record[1].getField('description')[0] for record in records])

np.save('../data/arxiv_raw_abstracts.npy',abstracts)

parsed = []


for i in range(len(abstracts)):
    abstract = ' '.join(abstracts[i].lower().split())
    table = str.maketrans({key: None for key in string.punctuation})
    abstract = abstract.translate(table)
    parsed.append(abstract)


np.save('../data/arxiv_parsed.npy',np.array(parsed))






