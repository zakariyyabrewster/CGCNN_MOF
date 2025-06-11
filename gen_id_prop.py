import pandas as pd

df = pd.read_csv('test_datasets/full_CoRE2019_alldata.csv')
id_Di = df[['MOFname', 'Di']]
id_Di.to_csv('test_datasets/id_Di.csv', index=False)
id_CH4_HP = df[['MOFname', 'pure_uptake_methane_298.00_6500000']]
id_CH4_HP.to_csv('test_datasets/id_CH4_HP.csv', index=False)
id_CO2_LP = df[['MOFname', 'pure_uptake_CO2_298.00_15000']]
id_CO2_LP.to_csv('test_datasets/id_CO2_LP.csv', index=False)
id_logKH_CO2 = df[['MOFname', 'logKH_CO2']]
id_logKH_CO2.to_csv('test_datasets/id_logKH_CO2.csv', index=False)