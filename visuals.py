###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score



def evaluate(results, auprc_benchmark, recall_benchmark):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (11,7))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000','#A0A0A0']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'auprc_train','recall_train','pred_time','auprc_cv','recall_cv']):
            for i in np.arange(4):
                
                # Creative plot code
                ax[j/3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j/3, j%3].set_xticks([0.45, 1.45, 2.45, 3.45])
                ax[j/3, j%3].set_xticklabels(["1%", "10%","50%","100%"])
                ax[j/3, j%3].set_xlabel("Training Set Size")
                # ax[j/3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Precision")
    ax[0, 2].set_ylabel("Recall")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Precision")
    ax[1, 2].set_ylabel("Recall")

    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("AUPRC on Training Subset")
    ax[0, 2].set_title("Recall Score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Cross Validation Score on Training Set (AUPRC)")
    ax[1, 2].set_title("Cross Validation Score on Training Set (Recall)")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = auprc_benchmark, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = auprc_benchmark, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = recall_benchmark, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = recall_benchmark, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))


    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 4, fontsize = 'x-large')
    
    # Aesthetics
    pl.suptitle("Performance Metrics for The Four Classification Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()
    