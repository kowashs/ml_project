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
  for abst in enumerate(arxiv_abstracts):
    i,abstract = abst
    arxiv_words += len(abstract)
    for i,word in enumerate(abstract):
      if word == '':
        print('arxiv',abstract[i-2:i+2])
      if word in dict:
        dict[word] += np.array([1,0])
      else:
        dict[word] = np.array([1,0])

    if (i+1)%1000==0:
      print(f"Processed {i+1} arXiv abstracts")

  snarxiv_words = 0      # total # of words in snarxiv training set
  for abst in enumerate(snarxiv_abstracts):
    i,abstract = abst
    snarxiv_words += len(abstract)
    for i,word in enumerate(abstract):
      if word == '':
        print('snarxiv',abstract[i-2:i+2])
      if word in dict:
        dict[word] += np.array([0,1])
      else:
        dict[word] = np.array([0,1])

    if i%1000==0:
      print(f"Processed {i+1} snarXiv abstracts")

  #Only put words occurring multiple times into the vocabulary (not sure if smart or not)
  vocab = []
  bad_words = np.array([0,0])
  for word in dict:
    if sum(dict[word])<=1:
      bad_words += dict[word]
    else:
      vocab.append(word)
  vocab = [word for word in dict if sum(dict[word]) > 1]
  print(len(dict))
  print(len(vocab))
  print(bad_words)

  # Put all words in vocabulary
  #vocab = [word for word in dict]
  return vocab

###############################################################################
# Create vocabulary & save to .npz file

# Choose how many abstracts to get (train+test)
N_arxiv = 12000
N_snarxiv = 12000
# Choose where to start pulling papers in the arxiv search results
# arxiv_start = 3000   # must be <=30000-N_arxiv

# Get abstracts
# arxiv_abstracts = get_abstracts.get_arxiv(arxiv_start, N_arxiv)

# Get abstracts from pre-downloaded data
arxiv_abstracts = get_abstracts.get_stored_arxiv(N_arxiv)
print(f"Loaded {N_arxiv} arXiv abstracts")

snarxiv_abstracts = get_abstracts.get_snarxiv(N_snarxiv)

# Create vocabulary and save to file (currently contains 15,315 words)
vocab = make_vocab(arxiv_abstracts,snarxiv_abstracts)
np.savez('big_vocab',vocab=vocab)

print('Vocabulary created! Size:',len(vocab))
