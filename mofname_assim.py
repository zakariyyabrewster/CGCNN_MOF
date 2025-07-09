import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import os

labels = pd.read_csv('test_datasets/full_CoRE2019_alldata.csv')
labels_names = labels[['MOFname']]
structure_names = pd.read_csv('test_datasets/filenames_CoRE2019.txt', header=None, names=['MOFname'])
structure_names['MOFname'] = structure_names['MOFname'].str.replace(".cif", "", regex=False)
data = []

with open("test_datasets/core_mofid.smi") as fin:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        if line[0] == '*':
            continue
        smiles, rest = line.split(" ", 1)
        mofid, mofname = rest.split(";")
        top_cat = ".".join(mofid.split(".")[1:])
        mofid = f"{smiles}&&{top_cat}"
        data.append((mofid, mofname))

mof_ids = pd.DataFrame(data, columns=['MOFID', 'MOFname'])

id1 = set(labels_names['MOFname'])
id2 = set(structure_names['MOFname'])
id3 = set(mof_ids['MOFname'])

common_names = sorted(list(id1 & id2 & id3))

common_names_df = pd.DataFrame(common_names, columns=['MOFname'])
common_names_df.to_csv('test_datasets/common_mofnames.csv', index=False)

train_names, test_names = train_test_split(common_names, test_size=0.2, random_state=42, shuffle=True)


os.makedirs("test_datasets/folds/train_val", exist_ok=True)

pd.DataFrame(train_names, columns=["MOFname"]).to_csv("test_datasets/folds/train_full.csv", index=False)
pd.DataFrame(test_names, columns=["MOFname"]).to_csv("test_datasets/folds/test_holdout.csv", index=False)

train_names = sorted(train_names) 

kf = KFold(n_splits=5, shuffle=True, random_state=42)



for fold, (train_idx, val_index) in enumerate(kf.split(train_names)):
    fold_train = [train_names[i] for i in train_idx]
    fold_val = [train_names[i] for i in val_index]

    pd.DataFrame(fold_train, columns=["MOFname"]).to_csv(f"test_datasets/folds/train_val/fold_{fold}_train.csv", index=False)
    pd.DataFrame(fold_val, columns=["MOFname"]).to_csv(f"test_datasets/folds/train_val/fold_{fold}_val.csv", index=False)

properties = {
    'Di': 'Di',
    'Dif': 'Dif',
    'Df': 'Df',
    'CO2_LP': 'pure_uptake_CO2_298.00_15000',
    'CH4_HP': 'pure_uptake_methane_298.00_6500000',
    'logKH_CO2': 'logKH_CO2'
}

# CGCNN tables
reduced_labels = labels[labels['MOFname'].isin(common_names)]

for prop, col_name in properties.items():
    id_prop = reduced_labels[['MOFname', col_name]]
    id_prop.columns = ['MOFname', prop]
    id_prop.to_csv(f'test_datasets/id_{prop}.csv', index=False)

    mofid_prop = mof_ids.merge(reduced_labels[['MOFname', col_name]], on='MOFname', how='inner')
    mofid_prop = mofid_prop[['MOFID', col_name, 'MOFname']]
    mofid_prop.columns = ['MOFID', prop, 'MOFname']
    mofid_prop.to_csv(f'test_datasets/mofid_{prop}.csv', index=False)

