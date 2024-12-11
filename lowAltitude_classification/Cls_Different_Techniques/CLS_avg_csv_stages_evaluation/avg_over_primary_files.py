import pandas as pd


df = pd.read_csv(
    'lowAltitude_classification/Cls_Different_Techniques/CLS_primary_csv_stages_evaluation/phase1_evaluation_drone_merged.csv')
df['Experiment'] = df['Experiment'].apply(lambda x: x.split('_')[0])
df_val = df[df['Dataset'] == 'Validation']
df_test = df[df['Dataset'] == 'Test']

df_val['Experiment'] = df_val['Experiment'].apply(lambda x: x[:-1])
############## we decided to discard 21 aug1
df_val = df_val[df_val['Experiment'] != 21]

############## we decided to discard 21 aug1
index_to_name_mapping = {
    '0': '0-Base',
    '1': '1-Filtered',
    '20': '20-Aug0',
    '22': '21-Aug1',
    '23': '22-Aug2',
    '24': '23-Aug3',
    '25': '24-Aug4',
    '30': '30-Balanced0',
    '4': '4-Background',
    '5': '5-Final',
}
df_val['Experiment'] = df_val['Experiment'].map(index_to_name_mapping)

df_val_acc = df_val.groupby('Experiment')['pAcc'].mean().reset_index()
df_val_f1_MACRO = df_val.groupby('Experiment')['F1'].mean().reset_index()
df_val_metrics = pd.merge(df_val_f1_MACRO, df_val_acc, on='Experiment')

df_val_metrics.to_csv('lowAltitude_classification/Cls_Different_Techniques/CLS_avg_csv_stages_evaluation/VAL_CLS_evaluation_drone_AVG.csv', index=False)

###############################################################################

df_test['Experiment'] = df_test['Experiment'].apply(lambda x: x[:-1])
############## we decided to discard 21 aug1
df_test = df_test[df_test['Experiment'] != 21]

df_test['Experiment'] = df_test['Experiment'].map(index_to_name_mapping)

df_test_acc = df_test.groupby('Experiment')['pAcc'].mean().reset_index()
df_test_f1_MACRO = df_test.groupby('Experiment')['F1'].mean().reset_index()
df_test_metrics = pd.merge(df_test_acc, df_test_f1_MACRO, on='Experiment')

df_test_metrics.to_csv('lowAltitude_classification/Cls_Different_Techniques/CLS_avg_csv_stages_evaluation/TEST_CLS_evaluation_drone_AVG.csv', index=False)
