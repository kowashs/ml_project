import numpy as np, sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)
# plt.rc('text', usetex=True)
# plt.rc('font', family='sans-serif')
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

  # Only keep words appearing in both corpora
  bad_words = []
  for word in dict:
    if dict[word][0]==0 or dict[word][1]==0:
      bad_words.append(word)
  for word in bad_words:
    del dict[word]

  # Create dictionary of conditional probabilities
  # Structure: P_dict['hello'] = [P('hello'|arxiv), P('hello'|snarxiv)]
  P_dict = {}
  for word in dict:
    P_word_arxiv = dict[word][0]/arxiv_words
    P_word_snarxiv = dict[word][1]/snarxiv_words
    P_dict[word] = np.array([P_word_arxiv, P_word_snarxiv])
  #print(arxiv_words,snarxiv_words)
  return P_dict, P_arxiv


# Classify a test abstract as arxiv or snarxiv using P_dict from train
def classify(abstract,P_dict,P_arxiv,eta,gamma):
  # Compute [P(abstract|arxiv), P(abstract|snarxiv)] using bag-of-words model
  # Use log[P(abstract|arxiv)/P(abstract|snarxiv)] to classify
  log_arx2snarx_ratio = 0
  len_abstract = 0
  for word in abstract:
    if word in P_dict:
      log_arx2snarx_ratio += np.log(P_dict[word][0])-np.log(P_dict[word][1])
      len_abstract += 1
  log_arx2snarx_ratio = log_arx2snarx_ratio/len_abstract

  # P_abstract_source = np.array([1.,1.])
  # len_abstract = 0
  # for word in abstract:
  #   if word in P_dict:
  #     P_abstract_source *= P_dict[word]
  #     len_abstract += 1
  # P_abstract_source = P_abstract_source**(1/len_abstract)
  #
  #if P_abstract_source[1] < 1e-14:
  #  print(P_abstract_source[0]/P_abstract_source[1])
  #arx2snarx_ratio = P_arxiv*P_abstract_source[0] / ((1.-P_arxiv)*P_abstract_source[1])

  # if log_arx2snarx_ratio < -14:
  #   print(log_arx2snarx_ratio)

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
arxiv_start = 600   # must be <30000-N_arxiv

# Choose N-P parameters eta & gamma
#eta_list = np.round(np.linspace(0.01,5,100),2)
eta_list = [0.67]
#eta = 0.5
gamma = 0.5

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


# Apply to test data
TPR_list = []
FDR_list = []
for i in range(len(eta_list)):
  eta = eta_list[i]
  # Attempt to classify test abstracts & compute classification error
  arxiv_wins=0; arxiv_losses=0
  log_ratio_arxiv_list = []
  for abstract in arxiv_test:
    label, log_arx2snarx = classify(abstract,P_dict,P_arxiv,eta,gamma)
    log_ratio_arxiv_list.append(log_arx2snarx)
    if label == 'arxiv':
      arxiv_wins+=1
    else:
      arxiv_losses+=1
  arxiv_accuracy = arxiv_wins/(arxiv_wins+arxiv_losses)

  snarxiv_wins=0; snarxiv_losses=0
  log_ratio_snarxiv_list = []
  for abstract in snarxiv_test:
    label, log_arx2snarx = classify(abstract,P_dict,P_arxiv,eta,gamma)
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
# plt.savefig('FDR_TPR_plot.png')
# plt.close()


# Make histogram of P(X|arx)/P(X|snarx) (X is an arxiv/snarxiv abstract)
my_bins = np.logspace(-1,0.5, 30)

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
plt.savefig('histogram.png')
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
# plt.savefig('histogram.png')
# plt.close()
