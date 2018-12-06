import numpy as np, sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import api_get_abstracts as get_abstracts

###############################################################################
# Create Neyman-Pearson classifier using bag-of-words model
###############################################################################
# Compute conditional probabilities for training data (used in classifier)
def train(arxiv_abstracts, snarxiv_abstracts, vocab):
  N_arxiv = len(arxiv_abstracts)
  N_snarxiv = len(snarxiv_abstracts)

  #P_arxiv = N_arxiv/(N_arxiv+N_snarxiv)  # not needed for Neyman-Pearson

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

  N_snarxiv_words = 0      # total # of words in snarxiv training set
  for abstract in snarxiv_abstracts:
    N_snarxiv_words += len(abstract)
    for word in abstract:
      if word not in vocab:
        word = '<UNK>'
      dict.setdefault(word,np.array([0,0]))
      dict[word][1] += 1

  # Create dictionary of conditional probabilities w/ Laplace smoothing
  # Structure: P_dict['hello'] = [P('hello'|arxiv), P('hello'|snarxiv)]
  P_dict = {}
  V = len(vocab)
  for word in dict:
    P_word_arxiv = (dict[word][0]+1)/(N_arxiv_words+V)
    P_word_snarxiv = (dict[word][1]+1)/(N_snarxiv_words+V)
    P_dict[word] = np.array([P_word_arxiv, P_word_snarxiv])
  #print(N_arxiv_words, N_snarxiv_words)
  return P_dict, N_arxiv_words, N_snarxiv_words #, P_arxiv


# Classify a test abstract as arxiv or snarxiv using P_dict from train
def classify(abstract,P_dict,N_arxiv_words,N_snarxiv_words, eta,gamma, vocab):
  # Compute [P(abstract|arxiv), P(abstract|snarxiv)] using bag-of-words model
  # Use log[P(abstract|arxiv)/P(abstract|snarxiv)] to classify
  V = len(vocab)
  log_arx2snarx_ratio = 0
  for word in abstract:
    if word not in vocab:
      word = '<UNK>'
    if word in P_dict:
      log_arx2snarx_ratio += np.log(P_dict[word][0])-np.log(P_dict[word][1])
    else:
      log_arx2snarx_ratio += np.log((N_snarxiv_words+V)/(N_arxiv_words+V))
  log_arx2snarx_ratio = log_arx2snarx_ratio/len(abstract)


  log_eta = np.log(eta)
  if log_arx2snarx_ratio > log_eta:
    return 'arxiv', log_arx2snarx_ratio
  elif log_arx2snarx_ratio < log_eta:
    return 'snarxiv', log_arx2snarx_ratio
  elif log_arx2snarx_ratio == log_eta:
    if np.random.rand() <= gamma:
      return 'arxiv', log_arx2snarx_ratio
    else:
      return 'snarxiv', log_arx2snarx_ratio


###############################################################################
# Apply to test data
###############################################################################
# Choose how many abstracts to get (train+test)
N_arxiv = 1000	    # must be <2000
N_snarxiv = 1000
# Choose where to start pulling papers in the arxiv search results
arxiv_start = 100   # must be <30000-N_arxiv

# Choose N-P parameters eta & gamma
#eta_list = np.round(np.linspace(0.01,5,100),2)
eta_list = [0.67]
#eta = 0.5
gamma = 0.5

# Get abstracts
arxiv_abstracts = get_abstracts.get_arxiv(arxiv_start, N_arxiv)
snarxiv_abstracts = get_abstracts.get_snarxiv(N_snarxiv)

# Split abstracts into train and test sets
N_arxiv_train = int(round(0.5*N_arxiv))       # N_arxiv_test = N_arxiv - N_arxiv_train
N_snarxiv_train = int(round(0.5*N_snarxiv))   # N_snarxiv_test = N_snarxiv - N_snarxiv_train

arxiv_train = arxiv_abstracts[:N_arxiv_train]
arxiv_test = arxiv_abstracts[N_arxiv_train:]

snarxiv_train = snarxiv_abstracts[:N_snarxiv_train]
snarxiv_test = snarxiv_abstracts[N_snarxiv_train:]


# Load vocabulary (currently contains 11645 words)
vocab = np.load('vocab.npz')['vocab']

# Get word probabilities from training set
P_dict, N_a, N_s = train(arxiv_train, snarxiv_train, vocab)


# Apply to test data
TPR_list = []
FDR_list = []
for i in range(len(eta_list)):
  eta = eta_list[i]
  # Attempt to classify test abstracts & compute classification error
  arxiv_wins=0; arxiv_losses=0
  log_ratio_arxiv_list = []
  for abstract in arxiv_test:
    label, log_arx2snarx = classify(abstract, P_dict,N_a,N_s, eta,gamma, vocab)
    log_ratio_arxiv_list.append(log_arx2snarx)
    if label == 'arxiv':
      arxiv_wins+=1
    else:
      arxiv_losses+=1
  arxiv_accuracy = arxiv_wins/(arxiv_wins+arxiv_losses)

  snarxiv_wins=0; snarxiv_losses=0
  log_ratio_snarxiv_list = []
  for abstract in snarxiv_test:
    label, log_arx2snarx = classify(abstract, P_dict,N_a,N_s, eta,gamma, vocab)
    log_ratio_snarxiv_list.append(log_arx2snarx)
    if label == 'snarxiv':
      snarxiv_wins+=1
    else:
      snarxiv_losses+=1
  snarxiv_accuracy = snarxiv_wins/(snarxiv_wins+snarxiv_losses)

  # Call snarxiv positive, arxiv negative (think spam flag)
  TPR_list.append(snarxiv_accuracy)
  if arxiv_losses == 0:
    FDR_list.append(0)
  else:
    FDR_list.append(arxiv_losses/(arxiv_losses + snarxiv_wins))

  wins = arxiv_wins+snarxiv_wins
  losses = arxiv_losses+snarxiv_losses
  accuracy = wins/(wins+losses)   # E_test = losses/(wins+losses)

  print('eta:',eta)
  print('arXiv accuracy: %.3f (%i wins, %i losses)' %(arxiv_accuracy,arxiv_wins,arxiv_losses))
  print('snarXiv accuracy: %.3f (%i wins, %i losses)' %(snarxiv_accuracy,snarxiv_wins,snarxiv_losses))
  print('Accuracy: %.3f (%i wins, %i losses)' %(accuracy,wins,losses))
  print(TPR_list[-1],FDR_list[-1])
  print()


i_best = np.argmax(np.array(TPR_list)-np.array(FDR_list))
eta_best = eta_list[i_best]

# Make TPR vs. FDR plot
# plt.figure()
# #plt.plot(FDR_list,TPR_list,'b.')
# plt.plot(FDR_list[:i_best],TPR_list[:i_best],'b.')
# plt.plot(FDR_list[i_best],TPR_list[i_best],'rx')
# plt.plot(FDR_list[i_best+1:],TPR_list[i_best+1:],'g.')
# plt.xlim((-.05,1.05)); plt.ylim((-.05,1.05))
# plt.legend(['$\eta>$'+str(eta_best), '$\eta=$'+str(eta_best),
#  '$\eta<$'+str(eta_best)],loc='lower right')
# plt.xlabel('False Discovery Rate (FDR)')
# plt.ylabel('True Positive Rate (TPR)')
# plt.savefig('../figures/FDR_TPR_plot.png')
# plt.close()


# Make histogram of P(X|arx)/P(X|snarx) (X is an arxiv/snarxiv abstract)
my_bins = np.logspace(-1,1, 30)

plt.figure()
plt.hist(np.exp(log_ratio_arxiv_list),bins=my_bins,alpha=0.5)
plt.hist(np.exp(log_ratio_snarxiv_list),bins=my_bins,alpha=0.5)
plt.gca().set_xscale("log")
#plt.gca().set_yscale("log")

plt.legend([r'$X\in$arXiv',r'$X\in$snarXiv'])
plt.xlabel('$P(X|$arXiv$)/P(X|$snarXiv$)$')
plt.ylabel('Number of abstracts')

plt.axvline(0.67, color='k', linestyle='dashed', linewidth=1)
_, max_ = plt.ylim()
plt.text(0.67 + 0.067,
         max_ - max_/10,
         r'$\eta=0.67$')
plt.savefig('../figures/BOW_histogram.png')
plt.close()


# # Make histogram of word probabilities
# # WRONG THING! want P(X|arx)/P(X|snarx) when X is an arxiv/snarxiv abstract
# P_arxiv_words = [P_dict[word][0] for word in P_dict]
# P_snarxiv_words = [P_dict[word][1] for word in P_dict]
#
# my_bins = np.logspace(-6,0, 50)
#
# plt.figure()
# plt.hist(P_arxiv_words,bins=my_bins,alpha=0.5)
# plt.hist(P_snarxiv_words,bins=my_bins,alpha=0.5)
# plt.gca().set_xscale("log")
# #plt.gca().set_yscale("log")
#
# plt.legend(['arXiv','snarXiv'])
# plt.xlabel('$P(w|Y)$')
# plt.ylabel('Number of words')
# plt.savefig('../figures/histogram.png')
# plt.close()
