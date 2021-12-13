def printProgressBar (iteration, total, prefix = '', suffix = 'Complete', decimals = 1, length = 20, fill = ['🌑','🌒','🌓','🌔','🌕','🌖','🌗','🌘'], printEnd = ""):
    """
    progress bar for waiting for the code to run
    """
    x = ' '
    prefix = 25 * x + '| Progress:'
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    fillcount = iteration
    while fillcount >= len(fill):
        fillcount -= len(fill)

    # Print New Line on Complete
    if iteration == total:
        bar = '🌕' * filledLength
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    else:
        bar = fill[fillcount] * filledLength + '🌑' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)


def printProgress(message):
    x = ' '
    spaces = 25 - len(message)
    print('\r' + message + spaces * x, end = '')


import os
import re
def save_data(data, filename):
    filename = re.sub('/+','/', filename)
    filetype = filename.split('.')[-1]
    output_dir = '/'.join(filename.split('/')[:-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)
    if filetype == 'csv':
        data.to_csv(filename)
    elif filetype == 'parquet':
        data.to_parquet(filename)
    else:
        data.savefig(filename,bbox_inches='tight')



# def results_to_plot(tprs, aucs, auprcs, precisions):
#     # calculate values using metrics for plotting
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(self.mean_vals, mean_tpr)
#     std_auc = np.std(aucs)
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     mean_precision = np.mean(precisions, axis=0)
#     mean_precision[0] = 1.0
#     mean_auprc = auc(self.mean_vals, mean_precision)
#     std_auprc = np.std(auprcs)
#     std_precision = np.std(precisions, axis=0)
#     precisions_upper = np.minimum(mean_precision + std_precision, 1)
#     precisions_lower = np.maximum(mean_precision - std_precision, 0)
#     return mean_tpr,tprs_lower, tprs_upper, mean_precision, precisions_lower, precisions_upper, mean_auc, std_auc, mean_auprc, std_auprc
