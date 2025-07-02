import pandas as pd
data = []

import pandas as pd
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

df = pd.DataFrame(data, columns=['MOFID', 'MOFname'])
full_df = pd.read_csv("test_datasets/full_CoRE2019_alldata.csv")
df = df.merge(full_df[['MOFname', 'Di', 'Dif', 'Df', 'pure_uptake_CO2_298.00_15000', 'pure_uptake_methane_298.00_6500000', 'logKH_CO2']], on='MOFname', how='inner')
df = df.rename(columns={
    'MOFname': 'MOFname',
    'Di': 'Di',
    'Dif': 'Dif',
    'Df': 'Df',
    'pure_uptake_CO2_298.00_15000': 'CO2_LP',
    'pure_uptake_methane_298.00_6500000': 'CH4_HP',
    'logKH_CO2': 'logKH_CO2'})

mofid_Df = df[['MOFID', 'Df', 'MOFname']]
mofid_Df.to_csv('test_datasets/mofid_Df.csv', index=False)
mofid_Dif = df[['MOFID', 'Dif', 'MOFname']]
mofid_Dif.to_csv('test_datasets/mofid_Dif.csv', index=False)
mofid_Di = df[['MOFID', 'Di', 'MOFname']]
mofid_Di.to_csv('test_datasets/mofid_Di.csv', index=False)

mofid_CO2_LP = df[['MOFID', 'CO2_LP', 'MOFname']]
mofid_CO2_LP.to_csv('test_datasets/mofid_CO2_LP.csv', index=False)
mofid_CH4_HP = df[['MOFID', 'CH4_HP', 'MOFname']]
mofid_CH4_HP.to_csv('test_datasets/mofid_CH4_HP.csv', index=False)
mofid_logKH_CO2 = df[['MOFID', 'logKH_CO2', 'MOFname']]
mofid_logKH_CO2.to_csv('test_datasets/mofid_logKH_CO2.csv', index=False)
