import matplotlib.pyplot as plt


def show_results_plot(history, fold_n=None):
    for metric_name, values  in history.items():
        plt.figure()
        title = f'Fold {fold_n}' if fold_n else metric_name
        plt.suptitle(title)
        plt.plot(range(1,len(values) + 1),values)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.show()