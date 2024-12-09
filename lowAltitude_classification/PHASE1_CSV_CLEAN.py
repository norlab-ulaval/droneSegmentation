import pandas as pd


df = pd.read_csv('results_Journal/phase1_evaluation_drone/phase1_evaluation_drone_merged.csv')
df['Weight File'] = df['Weight File'].apply(lambda x: x.split('_')[0])
df_val = df[df['Dataset'] == 'Validation']
df_test = df[df['Dataset'] == 'Test']

df_val['Weight File'] = df_val['Weight File'].apply(lambda x: x[:-1])


index_to_name_mapping = {
    '0': '0-Base',
    '1': '1-Filtered',
    '20': '20-Aug0',
    '21': '21-Aug1',
    '22': '22-Aug2',
    '23': '23-Aug3',
    '24': '24-Aug4',
    '25': '25-Aug5',
    '30': '30-Balanced0',
    '4': '4-Background',
    '5': '5-Final',
}
df_val['Weight File'] = df_val['Weight File'].map(index_to_name_mapping)

df_val_acc = df_val.groupby('Weight File')['Accuracy'].mean().reset_index()
df_val_f1_MACRO = df_val.groupby('Weight File')['F1 Score - Macro'].mean().reset_index()
# df_val_f1_MACRO.rename(columns={'F1 Score - Macro': 'F1'}, inplace=True)
df_val_metrics = pd.merge(df_val_f1_MACRO, df_val_acc, on='Weight File')
#print(df_val_metrics)


df_val_metrics.to_csv('results/phase1_final/VAL_ViT.csv', index=False)

###############################################################################
#
df_test['Weight File'] = df_test['Weight File'].apply(lambda x: x[:-1])

df_test['Weight File'] = df_test['Weight File'].map(index_to_name_mapping)


df_test_acc = df_test.groupby('Weight File')['Accuracy'].mean().reset_index()
df_test_f1_MACRO = df_test.groupby('Weight File')['F1 Score - Macro'].mean().reset_index()
df_test_metrics = pd.merge(df_test_acc, df_test_f1_MACRO, on='Weight File')

# print(df_test_metrics)
df_test_metrics.to_csv('results/phase1_final/TEST_ViT.csv', index=False)
