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
def save_data(data, filename):
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
