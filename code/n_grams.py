import numpy as np, sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import api_get_abstracts as get_abstracts

###############################################################################
# Create likelihood-ratio (Neyman-Pearson) classifier using n-grams
# (currently the train/classify functions only works for bigrams)
###############################################################################

# Create (or add to existing) n_grams dictionary for given (parsed) abstract
def n_grams(abstract,n,source, dict_n={}):
  #dict_n = {}
  for i in range(len(abstract)-n+1):
    n_gram = tuple(abstract[i:i+n])
    dict_n.setdefault(n_gram,np.array([0,0]))
    if source == 'arxiv':
      dict_n[n_gram][0] += 1
    elif source == 'snarxiv':
      dict_n[n_gram][1] += 1
  return dict_n


# Compute conditional probabilities for training data (used in classifier)
def train(arxiv_abstracts, snarxiv_abstracts, vocab):
  N_arxiv = len(arxiv_abstracts)
  N_snarxiv = len(snarxiv_abstracts)

  #P_arxiv = N_arxiv/(N_arxiv+N_snarxiv)  # not needed for Neyman-Pearson

  # Store word info in dictionary (not ideal, but should work)
  # Structure: dict['hello'] = [#('hello' in arxiv), #('hello' in snarxiv)]
  dict_1 = {}
  dict_2 = {}

  N_arxiv_words = 0       # total # of words in arxiv training set
  for abstract in arxiv_abstracts:
    N_arxiv_words += len(abstract)
    for i,word in enumerate(abstract):
      if word not in vocab:
        word = '<UNK>'; abstract[i] = '<UNK>'
      dict_1.setdefault(word,np.array([0,0]))
      dict_1[word][0] += 1
    dict_2 = n_grams(abstract, 2, 'arxiv', dict_2)

  N_snarxiv_words = 0      # total # of words in snarxiv training set
  for abstract in snarxiv_abstracts:
    N_snarxiv_words += len(abstract)
    for i,word in enumerate(abstract):
      if word not in vocab:
        word = '<UNK>'; abstract[i] = '<UNK>'
      dict_1.setdefault(word,np.array([0,0]))
      dict_1[word][1] += 1
    dict_2 = n_grams(abstract, 2, 'snarxiv', dict_2)


  # Create dictionary of conditional probabilities with Laplace smoothing
  # Structure: P_dict['hello'] = [P('hello'|arxiv), P('hello'|snarxiv)]
  V = len(vocab)
  # P_dict_1 = {}
  # for word in dict_1:
  #   P_word_arxiv = (dict_1[word][0]+1)/(N_arxiv_words+V)
  #   P_word_snarxiv = (dict_1[word][1]+1)/(N_snarxiv_words+V)
  #   P_dict_1[word] = np.array([P_word_arxiv, P_word_snarxiv])
  P_dict_2 = {}
  for bigram in dict_2:
    w1 = bigram[0]
    P_bigram_arxiv = (dict_2[bigram][0]+1)/(dict_1[w1][0]+V)
    P_bigram_snarxiv = (dict_2[bigram][1]+1)/(dict_1[w1][1]+V)
    P_dict_2[bigram] = np.array([P_bigram_arxiv,P_bigram_snarxiv])
  return P_dict_2, dict_1


# Classify a test abstract as arxiv or snarxiv using P_dict from train
def classify(abstract, P_dict_2,dict_1, eta_PP,gamma, vocab):
  # Compute [P(abstract|arxiv), P(abstract|snarxiv)] using bag-of-words model
  # Use log[PP(abstract|arxiv)/PP(abstract|snarxiv)] to classify
  V = len(vocab)
  for i,word in enumerate(abstract):
    if word not in vocab:
      abstract[i] = '<UNK>'
  #N_bigrams = max(len(abstract)-1,1)
  log_PP_arx2snarx = 0
  for i in range(len(abstract)-1):
    bigram = tuple(abstract[i:i+2])
    if bigram in P_dict_2:
      log_PP_arx2snarx += np.log(P_dict_2[bigram][1])-np.log(P_dict_2[bigram][0])
    elif bigram[0] in dict_1:
      w1 = bigram[0]
      log_PP_arx2snarx += np.log((dict_1[w1][0]+V)/(dict_1[w1][1]+V))
    #else: add log(V/V) = 0
  log_PP_arx2snarx = log_PP_arx2snarx/len(abstract)  # log(perplexities ratio)

  log_eta = np.log(eta_PP)
  if log_PP_arx2snarx > log_eta:
    return 'snarxiv', log_PP_arx2snarx
  elif log_PP_arx2snarx < log_eta:
    return 'arxiv', log_PP_arx2snarx
  elif log_PP_arx2snarx == log_eta:
    if np.random.rand() <= gamma:
      return 'snarxiv', log_PP_arx2snarx
    else:
      return 'arxiv', log_PP_arx2snarx


###############################################################################
# Apply to test data
###############################################################################
# Choose which data sets to save to file and/or plot
make_TPR_FDR_plot = False
save_TPR_FDR_data = False

make_hist_plot = True
save_hist_data = True

# Choose how many abstracts to get
N_arxiv_train = 1000
N_snarxiv_train = 1000

N_arxiv_test = 1000
N_snarxiv_test = 1000

N_arxiv = N_arxiv_train + N_arxiv_test
N_snarxiv = N_snarxiv_train + N_snarxiv_test


# Choose LR parameters eta & gamma
#eta_list = sorted([0.7,np.round(10**np.arange(-0.5,0.5,0.1),2)])
eta_list = [1.4]
gamma = 0.5

# Generate new snarxiv abstracts
snarxiv_abstracts = get_abstracts.get_snarxiv(N_snarxiv)

# Get arxiv abstracts from pre-downloaded data
arxiv_abstracts = get_abstracts.get_stored_arxiv(N_arxiv)
print(f"Loaded {N_arxiv} arXiv abstracts")

arxiv_train = arxiv_abstracts[:N_arxiv_train]
arxiv_test = arxiv_abstracts[N_arxiv_train:]

snarxiv_train = snarxiv_abstracts[:N_snarxiv_train]
snarxiv_test = snarxiv_abstracts[N_snarxiv_train:]

# Load vocabulary
vocab = np.load('big_vocab.npz')['vocab']

# Get word probabilities from training set
P_dict_2, dict_1 = train(arxiv_train, snarxiv_train, vocab)

# Apply to test data
TPR_list = []; FDR_list = []
for i in range(len(eta_list)):
  eta = eta_list[i]
  # Attempt to classify test abstracts & compute classification error
  arxiv_wins=0; arxiv_losses=0
  log_ratio_arxiv_list = []
  for abstract in arxiv_test:
    label, log_arx2snarx = classify(abstract,P_dict_2,dict_1,eta,gamma,vocab)
    log_ratio_arxiv_list.append(log_arx2snarx)
    if label == 'arxiv':
      arxiv_wins+=1
    else:
      arxiv_losses+=1
  arxiv_accuracy = arxiv_wins/(arxiv_wins+arxiv_losses)

  snarxiv_wins=0; snarxiv_losses=0
  log_ratio_snarxiv_list = []
  for abstract in snarxiv_test:
    label, log_arx2snarx = classify(abstract,P_dict_2,dict_1,eta,gamma,vocab)
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


if save_TPR_FDR_data == True:
  # Save TPR & FDR data to file
  np.savez('bi_TPR_FDR_data', eta_list=eta_list, TPR_list=TPR_list, FDR_list=FDR_list)

if make_TPR_FDR_plot == True:
  # Make TPR vs. FDR plot
  if len(eta_list)==1:
    print('Warning: only one eta_PP point in TPR vs. FDR plot')
  # Make TPR vs. FDR plot
  plt.figure()
  #plt.plot(FDR_list,TPR_list,'b.')
  plt.plot(FDR_list[:i_best],TPR_list[:i_best],'b.')
  plt.plot(FDR_list[i_best],TPR_list[i_best],'rx')
  plt.plot(FDR_list[i_best+1:],TPR_list[i_best+1:],'g.')
  plt.xlim((-.05,1.05)); plt.ylim((-.05,1.05))
  plt.legend(['$\eta>$'+str(eta_best), '$\eta=$'+str(eta_best),
   '$\eta<$'+str(eta_best)],loc='lower right')
  plt.xlabel('False Discovery Rate (FDR)')
  plt.ylabel('True Positive Rate (TPR)')
  plt.savefig('../figures/FDR_TPR_plot_bi.png')
  plt.close()


if save_hist_data==True:
  # Save histogram data to file
  if len(eta_list)!=1:
    print('Warning: only data for last value of eta_PP in eta_list will be be used for histogram')
  np.savez('bi_hist_data', snarx_list=log_ratio_snarxiv_list, arx_list=log_ratio_arxiv_list)

if make_hist_plot==True:
  # Make histogram of P(X|arx)/P(X|snarx) (X is an arxiv/snarxiv abstract)
  my_bins = np.logspace(-1,1, 100)

  plt.figure()
  plt.hist(np.exp(log_ratio_arxiv_list),bins=my_bins,alpha=0.5)
  plt.hist(np.exp(log_ratio_snarxiv_list),bins=my_bins,alpha=0.5)
  plt.gca().set_xscale("log")
  #plt.gca().set_yscale("log")

  plt.legend([r'$X\in\,$arXiv',r'$X\in\,$snarXiv'])
  plt.xlabel('PP$(X|$arXiv$)/$PP$(X|$snarXiv$)$')
  plt.ylabel('Number of abstracts')

  plt.axvline(1.4, color='k', linestyle='dashed', linewidth=1)
  _, max_ = plt.ylim()
  # plt.text(0.77,
  #          max_ - max_/10,
  #          r'$\eta_{$PP$}=0.7$')
  plt.savefig('../figures/bi_histogram.png')
  plt.close()
