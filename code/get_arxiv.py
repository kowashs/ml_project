#!/usr/bin/env python
import numpy as np
from oaipmh.client import Client
from oaipmh.metadata import MetadataRegistry, oai_dc_reader
import string
import datetime

# Use with caution; this will pull down ALL papers in the specified arXiv set

URL = 'http://export.arxiv.org/oai2' # define repository URL

registry = MetadataRegistry()
registry.registerReader('oai_dc',oai_dc_reader) # define metadata reader
client = Client(URL, registry) # create harvester
print("Updating granularity if necessary.")
#client.updateGranularity() # if selective harvesting necessary
print("Client created.")


set_ = 'physics:hep-th'
metadataPrefix = 'oai_dc'



print("Making request.")
# get records from repository; probably a long time (~80k papers)
records = client.listRecords(metadataPrefix=metadataPrefix, set=set_)
print("Harvesting complete.")

# extract abstracts from records
abstracts = np.array([record[1].getField('description')[0] for record in records])

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





