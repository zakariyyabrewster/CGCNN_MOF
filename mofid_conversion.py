import pandas as pd
data = []

smi_path = "test_datasets/core_mofid.smi"
label_dir = "test_datasets/full_CoRE2019_alldata.csv"

target_properties = {
    'Di': 'Di',
    'Df': 'Df',
    'Dif': 'Dif',
    'pure_uptake_CO2_298.00_15000': 'CO2_LP',
    'pure_uptake_methane_298.00_6500000': 'CH4_HP',
    'logKH_CO2': 'logKH_CO2'
}

with open(smi_path) as fin:
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
full_df = pd.read_csv(label_dir)

for col_name, prop_name in target_properties.items():
    if col_name not in full_df.columns:
        print(f"Warning: {col_name} not found in the dataset.")
        continue
    
    merged = df.merge(full_df[['MOFname', col_name]], on='MOFname', how='inner')
    merged = merged.rename(columns={col_name: prop_name})
    merged = merged[['MOFID', prop_name, 'MOFname']]
    merged.to_csv(f'test_datasets/mofid_{prop_name}.csv', index=False)