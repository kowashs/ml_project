import urllib.request
import feedparser, string
import subprocess

###################################################################################################################
# Get arxiv abstracts
###################################################################################################################
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
  #print(response)

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

    # Format abstract
    abstract = abstract.lower()	# no capitals
    table = str.maketrans({key: None for key in string.punctuation}) # removes all punctuation marks (makes some weird words, but overall works pretty well)
    abstract = abstract.translate(table)
    abstract = abstract.split()	# split abstract into list of lowercase words with no punctuation
    #print(abstract)

    arxiv_abstracts.append(abstract)
  return arxiv_abstracts


###################################################################################################################
# Get snarxiv abstracts
###################################################################################################################
def get_snarxiv(N):
  # NOTE: I modified the last line of my snarxiv.gram to only give the abstract
  snarxiv_path = '/gscratch/stf/blanton1/Project/snarxiv/snarxiv'
  snarxiv_abstracts = []
  for i in range(N):
    abstract = subprocess.Popen(snarxiv_path, stdout=subprocess.PIPE)
    abstract = abstract.stdout.read().decode("utf-8")

    # Format abstract
    abstract = abstract.lower() # no capitals
    table = str.maketrans({key: None for key in string.punctuation}) # remove all punctuation
    abstract = abstract.translate(table)

    abstract = abstract.split()
    #print(abstract)

    snarxiv_abstracts.append(abstract)
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
