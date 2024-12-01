import pandas as pd


df = pd.read_csv('lowAltitude_classification/results/phase1_ViT/ViT.csv')
df_filtered = df[df['Overlap'] != 0.2]
df_filtered['Weight File'] = df_filtered['Weight File'].apply(lambda x: x.split('_')[0])
df_val = df_filtered[df_filtered['Dataset'] == 'Validation']
df_test = df_filtered[df_filtered['Dataset'] == 'Test']

df_val['Weight File'] = df_val['Weight File'].apply(lambda x: x[:-1])


# index_to_name_mapping = {
#     '0': '0-Base',
#     '1': '1-Filtered',
#     '20': '20-Aug0',
#     '21': '21-Aug1',
#     '22': '22-Aug2',
#     '23': '23-Aug3',
#     '24': '24-Aug4',
#     '25': '25-Aug5',
#     '30': '30-Balanced0',
#     '31': '31-Balanced1',
#     '4': '4-Background Cleaning',
#     '5': '5-Final (1+25+30+4)',
# }
# df_val['Weight File'] = df_val['Weight File'].map(index_to_name_mapping)


df_val_acc = df_val.groupby('Patch Size')['Accuracy'].mean().reset_index()
df_val_f1_MACRO = df_val.groupby('Patch Size')['F1 Score - Macro'].mean().reset_index()
df_val_f1_weighted = df_val.groupby('Patch Size')['F1 Score - Weighted'].mean().reset_index()
df_val_f1_MACRO.rename(columns={'F1 Score - Macro': 'F1'}, inplace=True)
df_val_metrics = pd.merge(df_val_f1_MACRO, df_val_acc, on='Patch Size')
# print(df_val_metrics)
# exit()

df_val_metrics.to_csv('lowAltitude_classification/results/phase1_ViT/VAL_ViT.csv', index=False)

###############################################################################

df_test['Weight File'] = df_test['Weight File'].apply(lambda x: x[:-1])

# index_to_name_mapping = {
#     # '0': '0-Base',
#     # '1': '1-Filtered',
#     # '20': '20-Aug0',
#     # '21': '21-Aug1',
#     # '22': '22-Aug2',
#     # '23': '23-Aug3',
#     # '24': '24-Aug4',
#     # '25': '25-Aug5',
#     # '30': '30-Balanced0',
#     # '31': '31-Balanced1',
#     # '4': '4-Background Cleaning',
#     '5': '5-Final (1+25+30+4)',
# }
# df_test['Weight File'] = df_test['Weight File'].map(index_to_name_mapping)

df_test_acc = df_test.groupby('Patch Size')['Accuracy'].mean().reset_index()
df_test_f1_MACRO = df_test.groupby('Patch Size')['F1 Score - Macro'].mean().reset_index()
df_test_f1_weighted = df_test.groupby('Patch Size')['F1 Score - Weighted'].mean().reset_index()
df_test_f1_MACRO.rename(columns={'F1 Score - Macro': 'F1'}, inplace=True)
df_test_metrics = pd.merge(df_test_f1_MACRO, df_test_acc, on='Patch Size')
# df_test_metrics = pd.merge(df_test_acc, df_test_f1_MACRO, on='Weight File')
# df_test_metrics = pd.merge(df_test_metrics, df_test_f1_weighted, on='Weight File')

# print(df_test_metrics)
df_test_metrics.to_csv('lowAltitude_classification/results/phase1_ViT/TEST_ViT.csv', index=False)