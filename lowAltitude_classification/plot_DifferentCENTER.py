import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df_val = pd.read_csv('lowAltitude_classification/Result_Val_CENTER/Result_Val_CENTER.csv')
df_test = pd.read_csv('lowAltitude_classification/Result_Test_CENTER/Result_Test_CENTER.csv')

central_sizes = df_val['Central Size'].unique()

def plot_and_save(df, metric, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(10, 7))
    for central_size in central_sizes:
        subset = df[df['Central Size'] == central_size]
        ax.plot(subset['Overlap'], subset[metric], marker='o', label=f'Central Size {central_size}')
    ax.set_xlabel('Overlap')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.savefig(filename)
    plt.close(fig)

plot_and_save(df_val, 'F1', 'F1 Score', 'F1 Score vs Overlap for Different Central Sizes (Validation)', 'lowAltitude_classification/Result_Val_CENTER/val_f1_score.png')
plot_and_save(df_val, 'pAcc', 'Pixel Accuracy', 'Pixel Accuracy vs Overlap for Different Central Sizes (Validation)', 'lowAltitude_classification/Result_Val_CENTER/val_pixel_accuracy.png')
plot_and_save(df_val, 'mIoU', 'mIoU', 'mIoU vs Overlap for Different Central Sizes (Validation)', 'lowAltitude_classification/Result_Val_CENTER/val_miou.png')

plot_and_save(df_test, 'F1', 'F1 Score', 'F1 Score vs Overlap for Different Central Sizes (Test)', 'lowAltitude_classification/Result_Test_CENTER/test_f1_score.png')
plot_and_save(df_test, 'pAcc', 'Pixel Accuracy', 'Pixel Accuracy vs Overlap for Different Central Sizes (Test)', 'lowAltitude_classification/Result_Test_CENTER/test_pixel_accuracy.png')
plot_and_save(df_test, 'mIoU', 'mIoU', 'mIoU vs Overlap for Different Central Sizes (Test)', 'lowAltitude_classification/Result_Test_CENTER/test_miou.png')
