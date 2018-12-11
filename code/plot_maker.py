import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
#plt.switch_backend('agg')

# Use LaTeX for rendering
matplotlib.rcParams["text.usetex"] = True
# load the xfrac package
matplotlib.rcParams["text.latex.preamble"].append(r'\usepackage{xfrac}')

# Choose which plots to make
make_hist_BoW = True
make_TPR_FDR_BoW = True

make_hist_bi = True
make_TPR_FDR_bi = True

# Load necessary data
if make_hist_BoW or make_TPR_FDR_BoW:
  # Load data for BoW-LR model
  BoW_data = np.load('BoW_data.npz')
  BoW_log_ratio_snarx_list = BoW_data['snarx_list']
  BoW_log_ratio_arx_list = BoW_data['arx_list']

if make_hist_bi or make_TPR_FDR_bi:
  # Load data for bi-LR model
  bi_data = np.load('bi_data.npz')
  bi_log_ratio_snarx_list = bi_data['snarx_list']
  bi_log_ratio_arx_list = bi_data['arx_list']

################################################################################
# Bag-of-words plots

# Make histogram of PP(X|arx)/PP(X|snarx) for BoW-LR model (X is an arxiv/snarxiv abstract)
if make_hist_BoW == True:
  eta_PP = 1.4
  my_bins = 10.**np.arange(-1,1.3,0.02)
  #my_bins = np.logspace(-1,1, 100)

  fig, ax = plt.subplots()

  ax.hist(np.exp(BoW_log_ratio_arx_list),bins=my_bins,alpha=0.5)
  ax.hist(np.exp(BoW_log_ratio_snarx_list),bins=my_bins,alpha=0.5)
  ax.set_xscale("log")

  ax.set_title('Bag of words', fontsize=22)
  ax.set_xlabel('PP$(X|$arXiv$)/$PP$(X|$snarXiv$)$',fontsize=20,labelpad=6)
  ax.set_ylabel('Number of abstracts',fontsize=20,labelpad=10)

  ax.tick_params(axis='x', which='major', labelsize=14)
  ax.tick_params(axis='y', which='major', labelsize=14)

  ax.set_xticks([0.1,0.2,0.5,1,2,5,10,20])
  ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
  ax.set_xlim(0.1,20)
  ax.set_ylim(0,160)

  ax.legend([r'$X\in\,$arXiv',r'$X\in\,$snarXiv'], fontsize=14, loc='upper left')

  plt.axvline(eta_PP, color='k', linestyle='dashed', linewidth=1)
  _, max_ = plt.ylim()
  plt.text(0.96*eta_PP, 0.7*max_, '$\eta_{\\text{PP}}=$'+str(eta_PP), horizontalalignment='right', fontsize=14)

  fig.subplots_adjust(left=0.15)
  fig.subplots_adjust(bottom=0.16)

  plt.savefig('../figures/BoW_histogram.png')
  plt.savefig('../figures/BoW_histogram.pdf')
  #plt.show()
  plt.close()


# Make TPR vs. FDR plot for BoW-LR model
if make_TPR_FDR_BoW == True:
  # Create list of different eta_PP values to plot
  log_eta_PP_min = min(BoW_log_ratio_arx_list) - np.log(2)
  log_eta_PP_max = max(BoW_log_ratio_snarx_list) + np.log(2)
  log_eta_PP_list = np.linspace(log_eta_PP_min, log_eta_PP_max, 100)
  N_eta = len(log_eta_PP_list)

  # Find range of eta_PP values that give 100% classification accuracy (range = bump separation)
  log_perfect_eta_PP_min = max(BoW_log_ratio_arx_list)
  log_perfect_eta_PP_max = min(BoW_log_ratio_snarx_list)

  perfect_eta_min = round(np.exp(log_perfect_eta_PP_min),2)
  perfect_eta_max = round(np.exp(log_perfect_eta_PP_max),2)

  #log_perfect_eta_PP_list = [log_eta for log_eta in log_eta_PP_list if log_perfect_eta_PP_min < log_eta < log_perfect_eta_PP_max]
  i_low = np.where(log_eta_PP_list > log_perfect_eta_PP_min)[0][0]
  i_high = np.where(log_eta_PP_list < log_perfect_eta_PP_max)[0][-1]
  #perfect_eta_PP_list = np.exp(log_eta_PP_list[i_low:i_high+1])

  # Calculate TPR and FDR for each eta_PP
  TPR_list = np.zeros(N_eta); FDR_list = np.zeros(N_eta)
  for i in range(N_eta):
    log_eta_PP = log_eta_PP_list[i]

    arx_wins = sum(BoW_log_ratio_arx_list < log_eta_PP)
    arx_losses = sum(BoW_log_ratio_arx_list > log_eta_PP)

    snarx_wins = sum(BoW_log_ratio_snarx_list > log_eta_PP)
    snarx_losses = sum(BoW_log_ratio_snarx_list < log_eta_PP)

    TPR_list[i] = snarx_wins/(snarx_wins + snarx_losses)
    if arx_losses == 0:
      FDR_list[i] = 0
    else:
      FDR_list[i] = arx_losses/(arx_losses + snarx_wins)


  # Make TPR vs. FDR plot
  fig, ax = plt.subplots()

  plt.plot(FDR_list[:i_low],TPR_list[:i_low],'c.', markersize=10)
  plt.plot(FDR_list[i_low:i_high+1],TPR_list[i_low:i_high+1],'rx', markersize=10)
  plt.plot(FDR_list[i_high+1:],TPR_list[i_high+1:],'b.', markersize=10)

  plt.legend(['$\eta_{\\text{PP}}<$'+str(perfect_eta_min), str(perfect_eta_min)+'$\leq\eta_{\\text{PP}}\leq$'+str(perfect_eta_max),
   '$\eta_{\\text{PP}}>$'+str(perfect_eta_max)], fontsize=14, loc='lower right')

  plt.plot(FDR_list[i_low:i_high+1],TPR_list[i_low:i_high+1],'rx', markersize=10)

  ax.set_xlim((-.025,0.525)); ax.set_ylim((-.05,1.05))

  ax.set_title('Bag of words', fontsize=22)
  ax.set_xlabel('False Discovery Rate (FDR)', fontsize=20, labelpad=6)
  ax.set_ylabel('True Positive Rate (TPR)', fontsize=20, labelpad=10)

  ax.tick_params(axis='x', which='major', labelsize=14)
  ax.tick_params(axis='y', which='major', labelsize=14)


  fig.subplots_adjust(left=0.15)
  fig.subplots_adjust(bottom=0.16)

  plt.savefig('../figures/BoW_TPR_FDR.png')
  plt.savefig('../figures/BoW_TPR_FDR.pdf')
  #plt.show()
  plt.close()


################################################################################

# Bigram plots
if make_hist_bi == True:
  # Load histogram data for bi-LR model


  # Make histogram of P(X|arx)/P(X|snarx) (X is an arxiv/snarxiv abstract)
  eta_PP = 1.4
  my_bins = 10.**np.arange(-1,1.3,0.02)
  #my_bins = np.logspace(-1,1.3, 100)

  fig, ax = plt.subplots()

  ax.hist(np.exp(bi_log_ratio_arx_list),bins=my_bins,alpha=0.5)
  ax.hist(np.exp(bi_log_ratio_snarx_list),bins=my_bins,alpha=0.5)
  ax.set_xscale("log")

  ax.set_title('Bigram', fontsize=22)
  ax.set_xlabel('PP$(X|$arXiv$)/$PP$(X|$snarXiv$)$',fontsize=20,labelpad=6)
  ax.set_ylabel('Number of abstracts',fontsize=20,labelpad=10)

  ax.tick_params(axis='x', which='major', labelsize=14)
  ax.tick_params(axis='y', which='major', labelsize=14)

  ax.set_xticks([0.1,0.2,0.5,1,2,5,10,20])
  ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
  #ax.get_xaxis().get_major_formatter().labelOnlyBase = False
  ax.set_xlim(0.1,20)
  ax.set_ylim(0,160)

  ax.legend([r'$X\in\,$arXiv',r'$X\in\,$snarXiv'], fontsize=14, loc='upper left')

  plt.axvline(eta_PP, color='k', linestyle='dashed', linewidth=1)
  _, max_ = plt.ylim()
  plt.text(1.05*eta_PP, 0.9*max_, '$\eta_{\\text{PP}}=$'+str(eta_PP), horizontalalignment='left', fontsize=14)

  fig.subplots_adjust(left=0.15)
  fig.subplots_adjust(bottom=0.16)

  plt.savefig('../figures/bi_histogram.png')
  plt.savefig('../figures/bi_histogram.pdf')
  #plt.show()
  plt.close()



if make_TPR_FDR_bi == True:
  # Make TPR vs. FDR plot for bi-LR model

  # Create list of different eta_PP values to plot
  log_eta_PP_min = min(bi_log_ratio_arx_list) - np.log(2)
  log_eta_PP_max = max(bi_log_ratio_snarx_list) + np.log(2)
  log_eta_PP_list = np.linspace(log_eta_PP_min, log_eta_PP_max, 100)
  N_eta = len(log_eta_PP_list)

  # Find range of eta_PP values that give 100% classification accuracy (range = bump separation)
  log_perfect_eta_PP_min = max(bi_log_ratio_arx_list)
  log_perfect_eta_PP_max = min(bi_log_ratio_snarx_list)

  perfect_eta_min = round(np.exp(log_perfect_eta_PP_min),2)
  perfect_eta_max = round(np.exp(log_perfect_eta_PP_max),2)

  #log_perfect_eta_PP_list = [log_eta for log_eta in log_eta_PP_list if log_perfect_eta_PP_min < log_eta < log_perfect_eta_PP_max]
  i_low = np.where(log_eta_PP_list > log_perfect_eta_PP_min)[0][0]
  i_high = np.where(log_eta_PP_list < log_perfect_eta_PP_max)[0][-1]
  #perfect_eta_PP_list = np.exp(log_eta_PP_list[i_low:i_high+1])

  # Calculate TPR and FDR for each eta_PP
  TPR_list = np.zeros(N_eta); FDR_list = np.zeros(N_eta)
  for i in range(N_eta):
    log_eta_PP = log_eta_PP_list[i]

    arx_wins = sum(bi_log_ratio_arx_list < log_eta_PP)
    arx_losses = sum(bi_log_ratio_arx_list > log_eta_PP)

    snarx_wins = sum(bi_log_ratio_snarx_list > log_eta_PP)
    snarx_losses = sum(bi_log_ratio_snarx_list < log_eta_PP)

    TPR_list[i] = snarx_wins/(snarx_wins + snarx_losses)
    if arx_losses == 0:
      FDR_list[i] = 0
    else:
      FDR_list[i] = arx_losses/(arx_losses + snarx_wins)


  # Make TPR vs. FDR plot
  fig, ax = plt.subplots()

  plt.plot(FDR_list[:i_low],TPR_list[:i_low],'c.', markersize=10)
  plt.plot(FDR_list[i_low:i_high+1],TPR_list[i_low:i_high+1],'rx', markersize=10)
  plt.plot(FDR_list[i_high+1:],TPR_list[i_high+1:],'b.', markersize=10)

  plt.legend(['$\eta_{\\text{PP}}<$'+str(perfect_eta_min), str(perfect_eta_min)+'$\leq\eta_{\\text{PP}}\leq$'+str(perfect_eta_max),
   '$\eta_{\\text{PP}}>$'+str(perfect_eta_max)], fontsize=14, loc='lower right')

  plt.plot(FDR_list[i_low:i_high+1],TPR_list[i_low:i_high+1],'rx', markersize=10)

  ax.set_xlim((-.025,0.525))
  #ax.set_xlim((-.05,0.55))
  ax.set_ylim((-.05,1.05))

  ax.set_title('Bigram', fontsize=22)
  ax.set_xlabel('False Discovery Rate (FDR)', fontsize=20, labelpad=6)
  ax.set_ylabel('True Positive Rate (TPR)', fontsize=20, labelpad=10)

  ax.tick_params(axis='x', which='major', labelsize=14)
  ax.tick_params(axis='y', which='major', labelsize=14)


  fig.subplots_adjust(left=0.15)
  fig.subplots_adjust(bottom=0.16)

  plt.savefig('../figures/bi_TPR_FDR.png')
  plt.savefig('../figures/bi_TPR_FDR.pdf')
  #plt.show()
  plt.close()
