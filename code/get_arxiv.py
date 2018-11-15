from oaipmh.client import Client
from oaipmh.metadata import MetadataRegistry, oai_dc_reader

URL = 'http://export.arxiv.org/oai2'

registry = MetadataRegistry()
registry.registerReader('oai_dc',oai_dc_reader)
client = Client(URL, registry)

set_ = 'physics:hep-th'


