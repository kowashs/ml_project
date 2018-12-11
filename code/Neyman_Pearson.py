import numpy as np, sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import api_get_abstracts as get_abstracts

###############################################################################
# Create likelihood-ratio (Neyman-Pearson) classifier using bag-of-words model
###############################################################################
# Compute conditional probabilities for training data (used in classifier)
def train(arxiv_abstracts, snarxiv_abstracts, vocab):
  N_arxiv = len(arxiv_abstracts)
  N_snarxiv = len(snarxiv_abstracts)

  #P_arxiv = N_arxiv/(N_arxiv+N_snarxiv)  # not needed for Neyman-Pearson

  # Store word info in dictionary
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
def classify(abstract,P_dict,N_arxiv_words,N_snarxiv_words, eta_PP, gamma, vocab):
  # Compute [P(abstract|arxiv), P(abstract|snarxiv)] using bag-of-words model
  # Use log[PP(abstract|arxiv)/PP(abstract|snarxiv)] to classify
  V = len(vocab)
  log_PP_arx2snarx = 0
  for word in abstract:
    if word not in vocab:
      word = '<UNK>'
    if word in P_dict:
      log_PP_arx2snarx += np.log(P_dict[word][1])-np.log(P_dict[word][0])
    else:
      log_PP_arx2snarx += np.log((N_arxiv_words+V)/(N_snarxiv_words+V))
  log_PP_arx2snarx = log_PP_arx2snarx/len(abstract) # log(perplexities ratio)


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
# Choose whether to save data to file and/or make plots
save_data = True

make_hist_plot = False
make_TPR_FDR_plot = False

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
P_dict, N_a, N_s = train(arxiv_train, snarxiv_train, vocab)


# Apply to test data
TPR_list = []; FDR_list = []; i_perfect = []; eta_perfect = []
for i in range(len(eta_list)):
  eta = eta_list[i]
  # Attempt to classify test abstracts & compute classification error
  arxiv_wins=0; arxiv_losses=0
  log_ratio_arxiv_list = []; abstract_len_list=[]
  for abstract in arxiv_test:
    label, log_arx2snarx = classify(abstract, P_dict,N_a,N_s, eta,gamma, vocab)
    log_ratio_arxiv_list.append(log_arx2snarx)
    abstract_len_list.append(len(abstract))
    if label == 'arxiv':
      arxiv_wins+=1
    else:
      arxiv_losses+=1
  arxiv_accuracy = arxiv_wins/(arxiv_wins+arxiv_losses)
  PP_eta_min = np.exp(max(log_ratio_arxiv_list))
  eta_min = np.exp(max(np.array(log_ratio_arxiv_list)*np.array(abstract_len_list))) # should be eta_min
  print(np.mean(abstract_len_list))

  snarxiv_wins=0; snarxiv_losses=0
  log_ratio_snarxiv_list = []; abstract_len_list=[]
  for abstract in snarxiv_test:
    label, log_arx2snarx = classify(abstract, P_dict,N_a,N_s, eta,gamma, vocab)
    log_ratio_snarxiv_list.append(log_arx2snarx)
    abstract_len_list.append(len(abstract))
    if label == 'snarxiv':
      snarxiv_wins+=1
    else:
      snarxiv_losses+=1
  snarxiv_accuracy = snarxiv_wins/(snarxiv_wins+snarxiv_losses)
  PP_eta_max = np.exp(min(log_ratio_snarxiv_list))
  eta_max = np.exp(min(np.array(log_ratio_snarxiv_list)*np.array(abstract_len_list))) # should be eta_max
  print(np.mean(abstract_len_list))

  # Call snarxiv positive, arxiv negative (think spam flag)
  TPR_list.append(snarxiv_accuracy)
  if arxiv_losses == 0:
    FDR_list.append(0)
  else:
    FDR_list.append(arxiv_losses/(arxiv_losses + snarxiv_wins))

  wins = arxiv_wins+snarxiv_wins
  losses = arxiv_losses+snarxiv_losses
  accuracy = wins/(wins+losses)   # E_test = losses/(wins+losses)
  if losses==0:
    eta_perfect.append(eta)
    i_perfect.append(i)

  print('eta:',eta)
  print('arXiv accuracy: %.3f (%i wins, %i losses)' %(arxiv_accuracy,arxiv_wins,arxiv_losses))
  print('snarXiv accuracy: %.3f (%i wins, %i losses)' %(snarxiv_accuracy,snarxiv_wins,snarxiv_losses))
  print('Accuracy: %.3f (%i wins, %i losses)' %(accuracy,wins,losses))
  print(TPR_list[-1],FDR_list[-1])
  print()


if save_data==True:
  # Save data to file
  np.savez('BoW_data', snarx_list=log_ratio_snarxiv_list, arx_list=log_ratio_arxiv_list)


# if make_hist_plot==True:
#   # Make histogram of P(X|arx)/P(X|snarx) (X is an arxiv/snarxiv abstract)
#   if len(eta_list)>1:
#     print('Warning: only data for last value of eta_PP in eta_list will be be used for histogram')
#
#   my_bins = np.logspace(-1,1, 100)
#
#   plt.figure()
#
#   plt.hist(np.exp(log_ratio_arxiv_list),bins=my_bins,alpha=0.5)
#   plt.hist(np.exp(log_ratio_snarxiv_list),bins=my_bins,alpha=0.5)
#   plt.gca().set_xscale("log")
#
#   plt.legend([r'$X\in\,$arXiv',r'$X\in\,$snarXiv'])
#   plt.xlabel('PP$(X|$arXiv$)/$PP$(X|$snarXiv$)$')
#   plt.ylabel('Number of abstracts')
#
#   eta_PP = 1.4
#   plt.axvline(eta_PP, color='k', linestyle='dashed', linewidth=1)
#   _, max_ = plt.ylim()
#   plt.text(1.05*eta_PP, 0.9*max_, r'$\eta_{PP}=$'+str(eta_PP))
#   plt.savefig('../figures/BoW_histogram.png')
#   plt.close()


#################################################################
# i_best = np.argmax(np.array(TPR_list)-np.array(FDR_list))
# eta_best = eta_list[i_best]
#
# if save_TPR_FDR_data == True:
#   # Save TPR & FDR data to file
#   np.savez('BoW_TPR_FDR_data', eta_list=eta_list, TPR_list=TPR_list, FDR_list=FDR_list)
#
# if make_TPR_FDR_plot == True:
#   # Make TPR vs. FDR plot
#   if len(eta_list)==1:
#     print('Warning: only one eta_PP point in TPR vs. FDR plot')
#
#   plt.figure()
#   if len(eta_perfect)==0:
#     i_best = np.argmax(np.array(TPR_list)-np.array(FDR_list))
#     eta_best = eta_list[i_best]
#
#     plt.plot(FDR_list[:i_best],TPR_list[:i_best],'b.')
#     plt.plot(FDR_list[i_best],TPR_list[i_best],'rx')
#     plt.plot(FDR_list[i_best+1:],TPR_list[i_best+1:],'g.')
#     plt.legend(['$\eta>$'+str(eta_best), '$\eta=$'+str(eta_best),
#      '$\eta<$'+str(eta_best)], loc='lower right')
#   else:
#     i_low = i_perfect[0]; i_high = i_perfect[-1]
#     eta_low = eta_perfect[0]; eta_high = eta_perfect[-1]
#     plt.plot(FDR_list[:i_low],TPR_list[:i_low],'b.')
#     plt.plot(FDR_list[i_low:i_high+1],TPR_list[i_low:i_high+1],'rx')
#     plt.plot(FDR_list[i_high+1:],TPR_list[i_high+1:],'g.')
#     plt.legend(['$\eta>$'+str(eta_high), str(eta_low)+'$\leq\eta\leq$'+str(eta_high),
#      '$\eta<$'+str(eta_low)], loc='lower right')
#
#   plt.xlim((-.05,1.05)); plt.ylim((-.05,1.05))
#   plt.xlabel('False Discovery Rate (FDR)')
#   plt.ylabel('True Positive Rate (TPR)')
#   plt.savefig('../figures/FDR_TPR_plot_BoW.png')
#   plt.close()


print(eta_min, eta_max)
print(round(PP_eta_min,2),round(PP_eta_max,2))
