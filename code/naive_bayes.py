import numpy as np
import api_get_abstracts as get_abstracts

###############################################################################
# Create naive Bayes classifier using bag-of-words model
###############################################################################
# Compute conditional probabilities for training data (used in classifier)
def train(arxiv_abstracts, snarxiv_abstracts, vocab):
  N_arxiv = len(arxiv_abstracts)
  N_snarxiv = len(snarxiv_abstracts)

  P_arxiv = N_arxiv/(N_arxiv+N_snarxiv)  # P_snarxiv = 1-P_arxiv

  # Store word info in dictionary (not ideal, but should work)
  # Structure: dict['hello'] = [#('hello' in arxiv), #('hello' in snarxiv)]
  dict = {}

  N_arxiv_words = 0       # total # of words in arxiv training set
  for abstract in arxiv_abstracts:
    N_arxiv_words += len(abstract)
    for word in abstract:
      if word not in vocab:
        word = '<UNK>' # special character for words not in the vocabulary
      dict.setdefault(word,np.array([0,0]))
      dict[word][0] += 1

  N_snarxiv_words = 0       # total # of words in arxiv training set
  for abstract in snarxiv_abstracts:
    N_snarxiv_words += len(abstract)
    for word in abstract:
      if word not in vocab:
        word = '<UNK>' # special character for words not in the vocabulary
      dict.setdefault(word,np.array([0,0]))
      dict[word][1] += 1

  # arxiv_words = 0       # total # of words in arxiv training set
  # new_words=0
  # for abstract in arxiv_abstracts:
  #   arxiv_words += len(abstract)
  #   for word in abstract:
  #     if word in dict:
  #       dict[word] += np.array([1,0])
  #     else:
  #       dict[word] = np.array([1,0])
  #       new_words+=1
  # print(new_words)
  #
  # snarxiv_words = 0      # total # of words in snarxiv training set
  # new_words=0
  # for abstract in snarxiv_abstracts:
  #   snarxiv_words += len(abstract)
  #   for word in abstract:
  #     if word in dict:
  #       dict[word] += np.array([0,1])
  #     else:
  #       dict[word] = np.array([0,1])
  #       new_words+=1
  # print(new_words)

  # Create dictionary of conditional probabilities
  # Structure: P_dict['hello'] = [P('hello'|arxiv), P('hello'|snarxiv)]
  P_dict = {}
  V = len(vocab)
  for word in dict:
    P_word_arxiv = (dict[word][0]+1)/(N_arxiv_words+V)
    P_word_snarxiv = (dict[word][1]+1)/(N_snarxiv_words+V)
    P_dict[word] = np.array([P_word_arxiv, P_word_snarxiv])

  return P_dict, N_arxiv_words,N_snarxiv_words, P_arxiv


# Classify a test abstract as arxiv or snarxiv using P_dict from train
def classify(abstract, P_dict,N_arxiv_words,N_snarxiv_words,P_arxiv, vocab):
  # Compute [P(abstract|arxiv), P(abstract|snarxiv)] using bag-of-words model
  V = len(vocab)
  log_P_abstract_source = np.array([0.,0.])
  for word in abstract:
    if word not in vocab:
      word = '<UNK>'
    if word in P_dict:
      log_P_abstract_source += np.log(P_dict[word])
    else:
      log_P_abstract_source += np.log(np.array([1/(N_arxiv_words+V),1/(N_snarxiv_words+V)]))

  P_abstract_source = np.exp(log_P_abstract_source)
  # Compare P(arxiv|abstract) with P(snarxiv|abstract)
  if P_arxiv*P_abstract_source[0] > (1.-P_arxiv)*P_abstract_source[1]:
    return 'arxiv'
  else:
    return 'snarxiv'


###############################################################################
# Larry test: Is this garbage?
###############################################################################
# Choose how many abstracts to get (train+test)
N_arxiv = 1000	    # must be <2000
N_snarxiv = 1000
# Choose where to start pulling papers in the arxiv search results
arxiv_start = 1500   # must be <30000-N_arxiv

# Get abstracts
arxiv_abstracts = get_abstracts.get_stored_arxiv(N_arxiv)
print(f"Loaded {N_arxiv} arXiv abstracts")
snarxiv_abstracts = get_abstracts.get_snarxiv(N_snarxiv)

# Split abstracts into train and test sets
N_arxiv_train = int(round(0.8*N_arxiv))       # N_arxiv_test = N_arxiv - N_arxiv_train
N_snarxiv_train = int(round(0.8*N_snarxiv))   # N_snarxiv_test = N_snarxiv - N_snarxiv_train

arxiv_train = arxiv_abstracts[:N_arxiv_train]
arxiv_test = arxiv_abstracts[N_arxiv_train:]

snarxiv_train = snarxiv_abstracts[:N_snarxiv_train]
snarxiv_test = snarxiv_abstracts[N_snarxiv_train:]

# Load vocabulary (currently contains 11645 words)
vocab = np.load('vocab.npz')['vocab']

# Get word probabilities from training set
P_dict,N_arxiv_words,N_snarxiv_words,P_arxiv = train(arxiv_train, snarxiv_train, vocab)

# Attempt to classify test abstracts & compute classification error
arxiv_wins=0; arxiv_losses=0
for abstract in arxiv_test:
  if classify(abstract, P_dict,N_arxiv_words,N_snarxiv_words,P_arxiv, vocab) == 'arxiv':
    arxiv_wins+=1
  else:
    arxiv_losses+=1
arxiv_accuracy = arxiv_wins/(arxiv_wins+arxiv_losses)

snarxiv_wins=0; snarxiv_losses=0
for abstract in snarxiv_test:
  if classify(abstract, P_dict,N_arxiv_words,N_snarxiv_words,P_arxiv, vocab) == 'snarxiv':
    snarxiv_wins+=1
  else:
    snarxiv_losses+=1
snarxiv_accuracy = snarxiv_wins/(snarxiv_wins+snarxiv_losses)

wins = arxiv_wins+snarxiv_wins
losses = arxiv_losses+snarxiv_losses
accuracy = wins/(wins+losses)   # E_test = losses/(wins+losses)

print('arXiv accuracy: %.3f (%i wins, %i losses)' %(arxiv_accuracy,arxiv_wins,arxiv_losses))
print('snarXiv accuracy: %.3f (%i wins, %i losses)' %(snarxiv_accuracy,snarxiv_wins,snarxiv_losses))
print('Accuracy: %.3f (%i wins, %i losses)' %(accuracy,wins,losses))
