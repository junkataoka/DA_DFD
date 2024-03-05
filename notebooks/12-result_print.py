
import pandas as pd
import numpy as np
# Read src_domain column as string

table = pd.read_csv("/data/home/jkataok1/DA_DFD/reports/PU_WDCNN.csv", dtype={'src_domain':str, 'tar_domain':str})

#CWRU
table.loc[table["src_domain"].str.contains("0"), "src_domain"] = "A"
table.loc[table["src_domain"].str.contains("1"), "src_domain"] = "B"
table.loc[table["src_domain"].str.contains("2"), "src_domain"] = "C"
table.loc[table["src_domain"].str.contains("3"), "src_domain"] = "D"

table.loc[table["tar_domain"].str.contains("0"), "tar_domain"] = "A"
table.loc[table["tar_domain"].str.contains("1"), "tar_domain"] = "B"
table.loc[table["tar_domain"].str.contains("2"), "tar_domain"] = "C"
table.loc[table["tar_domain"].str.contains("3"), "tar_domain"] = "D"
table.sort_values(by=["src_domain", "tar_domain"], inplace=True)



for row in table.iterrows():
	print(row[1]["src_domain"], row[1]["tar_domain"], np.round(row[1]["tar_acc.max"]*100, 1))
	#print(row[1]["src_domain"], row[1]["tar_domain"], row[1]["accuracy"], row[1]["precision"], row[1]["recall"], row[1]["f1"])
