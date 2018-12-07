import numpy as np, sys
import api_get_abstracts as get_abstracts

###############################################################################
# Create vocabulary of words for arXiv and snarXiv abstracts
###############################################################################
def make_vocab(arxiv_abstracts, snarxiv_abstracts):
  N_arxiv = len(arxiv_abstracts)
  N_snarxiv = len(snarxiv_abstracts)

  # Store word info in dictionary (not ideal, but should work)
  # Structure: dict['hello'] = [#('hello' in arxiv), #('hello' in snarxiv)]
  dict = {}

  arxiv_words = 0       # total # of words in arxiv training set
  for abstract in arxiv_abstracts:
    arxiv_words += len(abstract)
    for word in abstract:
      if word in dict:
        dict[word] += np.array([1,0])
      else:
        dict[word] = np.array([1,0])

  snarxiv_words = 0      # total # of words in snarxiv training set
  for abstract in snarxiv_abstracts:
    snarxiv_words += len(abstract)
    for word in abstract:
      if word in dict:
        dict[word] += np.array([0,1])
      else:
        dict[word] = np.array([0,1])

  # Only put words occurring multiple times into the vocabulary (not sure if smart or not)
  # vocab = []
  # bad_words = np.array([0,0])
  # for word in dict:
  #   if sum(dict[word])==1:
  #     bad_words += dict[word]
  #   else:
  #     vocab.append(word)
  #vocab = [word for word in dict if sum(dict[word]) > 1]
  #print(len(dict))
  #print(len(vocab))
  #print(bad_words)

  # Put all words in vocabulary
  vocab = [word for word in dict]
  return vocab

###############################################################################
# Create vocabulary & save to .npz file

# Choose how many abstracts to get (train+test)
N_arxiv = 120000
N_snarxiv = 120000
# Choose where to start pulling papers in the arxiv search results
# arxiv_start = 3000   # must be <=30000-N_arxiv

# Get abstracts
# arxiv_abstracts = get_abstracts.get_arxiv(arxiv_start, N_arxiv)

# Get abstracts from pre-downloaded data
arxiv_abstracts = get_abstracts.get_stored_arxiv(N_arxiv)
print(f"Loaded {N_arxiv} arXiv abstracts")

snarxiv_abstracts = get_abstracts.get_snarxiv(N_snarxiv)

# Create vocabulary and save to file
vocab = make_vocab(arxiv_abstracts,snarxiv_abstracts)
np.savez('big_vocab',vocab=vocab)

print('Vocabulary created! Size:',len(vocab))
