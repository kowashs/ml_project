import urllib.request
import feedparser, string
import numpy as np
import subprocess

###################################################################################################################
# Get arxiv abstracts
###################################################################################################################
# Parse abstract
def my_parser(abstract):
  abstract = abstract.lower()    # no capitals
  abstract = abstract.replace('-',' ')
  abstract = abstract.split()    # split on spaces and hyphens
  table = str.maketrans({key: None for key in string.punctuation}) # table to remove all punctuation

  parsed_abstract = ['<s>']
  for w in abstract:
    if w[-1] in '.?': # if word ended in ". " or "? ", assume it ended a sentence
      parsed_abstract.append(w[:-1].translate(table))
      parsed_abstract.append('<e>')
      parsed_abstract.append('<s>')
    else:
      parsed_abstract.append(w.translate(table))
  if parsed_abstract[-1] == '<s>':
    del parsed_abstract[-1]
  if parsed_abstract[-1] != '<e>':
    parsed_abstract.append('<e>')

  return parsed_abstract



def get_arxiv(start,N):
  # Set query parameters
  search_query = 'cat:hep-th'       # only look at hep-th papers

  start = min(start,30000)          # choose which results from the search to keep
  max_results = min(N,30000)

  # Construct url of query result
  base_url = 'http://export.arxiv.org/api/query?'
  query = 'search_query=%s&start=%i&max_results=%i' % (search_query, start, max_results)
  url = base_url+query     #e.g., url = 'http://export.arxiv.org/api/query?search_query=cat:hep-th&start=10&max_results=1'

  # Perform GET request to get raw response (unparsed)
  response = urllib.request.urlopen(url).read()

  # Prep parser (magic I will not question)
  feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
  feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'


  # Parse the response using feedparser
  feed = feedparser.parse(response)

  # Get abstracts
  arxiv_abstracts = []
  for entry in feed.entries:
    #title = entry.title;        #print(title)
    #lead_author = entry.author; #print(lead_author)
    abstract = entry.summary;    #print(abstract)
    parsed_abstract = my_parser(abstract)
    snarxiv_abstracts.append(parsed_abstract)
  return arxiv_abstracts


# Get stored arxiv abstracts
def get_stored_arxiv(N_arxiv):
    # data = np.load('/gscratch/stf/kowash/ml_project/data/arxiv_parsed.npy')
    data = np.load('/gscratch/stf/kowash/ml_project/data/arxiv_raw_abstracts.npy')
    print(data[0])
    parsed_abstracts = [my_parser(abstract) for abstract in data[:N_arxiv]]
    return parsed_abstracts


###################################################################################################################
# Get snarxiv abstracts
###################################################################################################################
def get_snarxiv(N):
  # NOTE: I modified the last line of my snarxiv.gram to only give the abstract
  snarxiv_path = './snarxiv/snarxiv'
  snarxiv_abstracts = []
  for i in range(N):
    abstract = subprocess.Popen(snarxiv_path, stdout=subprocess.PIPE)
    abstract = abstract.stdout.read().decode("utf-8")
    parsed_abstract = my_parser(abstract)
    snarxiv_abstracts.append(parsed_abstract)
  print('Generated',N,'snarXiv abstracts')
  return snarxiv_abstracts


#############################################################################################################################
# Each function above outputs a "matrix" (lists of lists) with a different abstract in each row.
# More precisely: each row is the list of words in an abstract with no capital letters and no punctuation.

# Choose how many abstracts to get
#N_arxiv = 2	# must be <2000
#N_snarxiv = 2

# Choose where to start pulling papers in the arxiv search results
#arxiv_start = 0 # must be <30000-N_arxiv

#print(get_arxiv(arxiv_start,N_arxiv))
#print('\n\n\n')
#print(get_snarxiv(N_snarxiv))
