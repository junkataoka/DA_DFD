
import pandas as pd
import numpy as np
wdcnn_table = pd.read_csv("/data/home/jkataok1/DA_DFD/reports/CWRU_small_STARBEAR.csv")

wdcnn_table.loc[wdcnn_table["src_domain"].str.contains("0"), "src_domain"] = "A"
wdcnn_table.loc[wdcnn_table["src_domain"].str.contains("1"), "src_domain"] = "B"
wdcnn_table.loc[wdcnn_table["src_domain"].str.contains("2"), "src_domain"] = "C"
wdcnn_table.loc[wdcnn_table["src_domain"].str.contains("3"), "src_domain"] = "D"

wdcnn_table.loc[wdcnn_table["tar_domain"].str.contains("0"), "tar_domain"] = "A"
wdcnn_table.loc[wdcnn_table["tar_domain"].str.contains("1"), "tar_domain"] = "B"
wdcnn_table.loc[wdcnn_table["tar_domain"].str.contains("2"), "tar_domain"] = "C"
wdcnn_table.loc[wdcnn_table["tar_domain"].str.contains("3"), "tar_domain"] = "D"

wdcnn_table.sort_values(by=["src_domain", "tar_domain"], inplace=True)



for row in wdcnn_table.iterrows():
	print(row[1]["src_domain"], row[1]["tar_domain"], np.round(row[1]["tar_acc.max"]*100, 1))
	#print(row[1]["src_domain"], row[1]["tar_domain"], row[1]["accuracy"], row[1]["precision"], row[1]["recall"], row[1]["f1"])