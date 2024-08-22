import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_val = pd.read_csv('lowAltitude_classification/Result_Val_DifferentPatcheSize/Result_Val_DifferentPatcheSize.csv')
df_test = pd.read_csv('lowAltitude_classification/Result_Test_DifferentPatcheSize/Result_Test_DifferentPatcheSize.csv')

# Columns to exclude0
exclude_columns = ['mIoU']
combined_columns = [f'{row[df_val.columns[0]]}_{row[df_val.columns[1]]}' for _, row in df_val.iterrows()]

fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2

x = np.arange(len(df_val.index))
filtered_columns_val = [col for col in df_val.columns[2:] if col not in exclude_columns]
for i, column in enumerate(filtered_columns_val):
    ax.bar(x + i * width, df_val[column], width, label=f'Val - {column}')

ax.set_xticks(x + width * (len(filtered_columns_val) / 2 - 0.5))
ax.set_xticklabels(combined_columns, rotation=45, ha='right')
ax.set_xlabel('patchSize_overlap')
ax.set_ylabel('Performance')
ax.legend()
ax.grid(True)
# plt.tight_layout()
plt.title('(A) Validation Data')
plt.show()
# plt.savefig(fname='lowAltitude_classification/Result_Val_DifferentPatcheSize/Validation_DifferentPatchSizes.png')

fig, ax = plt.subplots(figsize=(10, 6))
filtered_columns_test = [col for col in df_test.columns[2:] if col not in exclude_columns]
for i, column in enumerate(filtered_columns_test):
    ax.bar(x + i * width, df_test[column], width, label=f'Test - {column}')

ax.set_xticks(x + width * (len(filtered_columns_test) / 2 - 0.5))
ax.set_xticklabels(combined_columns, rotation=45, ha='right')
ax.set_xlabel('patchSize_overlap')
ax.set_ylabel('Performance')
ax.legend()
ax.grid(True)
# plt.tight_layout()
plt.title('(B) Test Data')
plt.show()
# plt.savefig(fname='lowAltitude_classification/Result_Test_DifferentPatcheSize/Test_DifferentPatchSizes.png')
