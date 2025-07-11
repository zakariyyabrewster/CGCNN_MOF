import pandas as pd


label_dir = 'test_datasets/full_CoRE2019_alldata.csv'
# This script generates CSV files containing the MOF names and their corresponding properties from the full CoRE 2019 dataset.
df = pd.read_csv(label_dir)
# Choose the properties to extract
target_properties = {
    'column_name': 'property_name',
    'Di': 'Di',
    'pure_uptake_methane_298.00_6500000': 'CH4_HP',
    'pure_uptake_CO2_298.00_15000': 'CO2_LP',
    'logKH_CO2': 'logKH_CO2'
}
for col_name, prop_name in target_properties.items():
    id_prop = df[['MOFname', col_name]]
    id_prop.columns = ['MOFname', prop_name]
    id_prop.to_csv(f'test_datasets/id_{prop_name}.csv', index=False)


