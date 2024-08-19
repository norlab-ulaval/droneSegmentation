import pandas as pd
import matplotlib.pyplot as plt

df_val = pd.read_csv('lowAltitude_classification/Result_Val_DifferentPatcheSize/Result_Val_DifferentPatcheSize.csv')
df_test = pd.read_csv('lowAltitude_classification/Result_Test_DifferentPatcheSize/Result_Test_DifferentPatcheSize.csv')

combined_columns = [f'{row[df_val.columns[1]]}_{row[df_val.columns[2]]}' for _, row in df_val.iterrows()]

fig, ax = plt.subplots()
for column in df_val.columns[3:]:
    ax.plot(df_val[df_val.columns[0]], df_val[column], label=f'Val - {column}')

ax.set_xticks(range(len(df_val.index)))
ax.set_xticklabels(combined_columns, rotation=45, ha='right')
ax.set_xlabel('patchSize_overlap')
ax.set_ylabel('Performance')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.title('Validation Data')
plt.savefig(fname='lowAltitude_classification/Result_Val_DifferentPatcheSize/Validation_DifferentPatchSizes.png')


fig, ax = plt.subplots()
for column in df_test.columns[3:]:
    ax.plot(df_test[df_test.columns[0]], df_test[column], label=f'Test - {column}')

ax.set_xticks(range(len(df_test.index)))
ax.set_xticklabels(combined_columns, rotation=45, ha='right')
ax.set_xlabel('patchSize_overlap')
ax.set_ylabel('Performance')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.title('Test Data')
plt.savefig(fname='lowAltitude_classification/Result_Test_DifferentPatcheSize/Test_DifferentPatchSizes.png')
