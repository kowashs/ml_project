import numpy as np
import get_abstracts

###############################################################################
# Create naive Bayes classifier using bag-of-words model
###############################################################################
# Compute conditional probabilities for training data (used in classifier)
def train(arxiv_abstracts, snarxiv_abstracts):
  N_arxiv = len(arxiv_abstracts)
  N_snarxiv = len(snarxiv_abstracts)

  P_arxiv = N_arxiv/(N_arxiv+N_snarxiv)  # P_snarxiv = 1-P_arxiv

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

  # Create dictionary of conditional probabilities
  # Structure: P_dict['hello'] = [P('hello'|arxiv), P('hello'|snarxiv)]
  P_dict = {}
  for word in dict:
    P_word_arxiv = dict[word][0]/arxiv_words
    P_word_snarxiv = dict[word][1]/snarxiv_words
    P_dict[word] = np.array([P_word_arxiv, P_word_snarxiv])

  return P_dict, P_arxiv


# Classify a test abstract as arxiv or snarxiv using P_dict from train
def classify(abstract,P_dict,P_arxiv):
  # Compute [P(abstract|arxiv), P(abstract|snarxiv)]
  P_abstract_source = np.array([1.,1.])
  for word in abstract:
    if word in P_dict:
      P_abstract_source *= P_dict[word]

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
arxiv_start = 600   # must be <30000-N_arxiv

# Get abstracts
arxiv_abstracts = get_abstracts.get_arxiv(arxiv_start, N_arxiv)
snarxiv_abstracts = get_abstracts.get_snarxiv(N_snarxiv)

# Split abstracts into train and test sets
N_arxiv_train = int(round(0.8*N_arxiv))       # N_arxiv_test = N_arxiv - N_arxiv_train
N_snarxiv_train = int(round(0.8*N_snarxiv))   # N_snarxiv_test = N_snarxiv - N_snarxiv_train

arxiv_train = arxiv_abstracts[:N_arxiv_train]
arxiv_test = arxiv_abstracts[N_arxiv_train:]

snarxiv_train = snarxiv_abstracts[:N_snarxiv_train]
snarxiv_test = snarxiv_abstracts[N_snarxiv_train:]

# Get word probabilities from training set
P_dict, P_arxiv = train(arxiv_train, snarxiv_train)

# Attempt to classify test abstracts & compute classification error
wins=0; losses=0
for abstract in arxiv_test:
  if classify(abstract,P_dict,P_arxiv) == 'arxiv':
    wins+=1
  else:
    losses+=1

for abstract in snarxiv_test:
  if classify(abstract,P_dict,P_arxiv) == 'snarxiv':
    wins+=1
  else:
    losses+=1

accuracy = wins/(wins+losses)   # E_test = losses/(wins+losses)
#print('Wins:',wins)
#print('Losses:',losses)
print('Accuracy: %.3f (%i wins, %i losses)' %(accuracy,wins,losses))
