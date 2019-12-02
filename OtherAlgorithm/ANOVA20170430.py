import csv 
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import numpy as np 
import scipy

def two_ways_anova(df, n):
	col = []
	row = []
	for i in data:
		col.append((data[i]))
	for i in range(len(data)):
		row.append(df.loc[i])

	if n == 1:
		total_mean, tss, colss, rowss, crss = 0, 0, 0, 0, 0

		for i in col:
			total_mean += sum(i)/(len(row) * len(col))

		tss, colss, rowss, crss = 0, 0, 0, 0
		for i in range(len(col)):
			for j in range(len(row)):
				crss += (col[i][j] - sum(col[i])/len(col[i]) - sum(row[j])/len(row[j]) + total_mean)**2
				tss += (col[i][j] - total_mean)**2
				colss += (sum(col[i])/len(col[i]) - total_mean)**2
				rowss += (sum(row[j])/len(row[j]) - total_mean)**2

		wss = tss - colss - rowss - crss
		colmss = colss / (len(col) - 1)
		rowmss = rowss / (len(row) - 1)
		crmss = crss / ((len(row) - 1) * (len(col) - 1))

		rowname = ["Row", "Column", "Error", "Total"]
		ss = [rowss, colss, crss, tss]
		df = [len(row) - 1, len(col) - 1, ((len(row) - 1) * (len(col) - 1)), (len(row) * len(col)) - 1]
		ms = [rowmss, colmss, crmss, ""]
		f = [rowmss / crmss, colmss / crmss, "", ""]
		p_value = [1 - scipy.stats.f.cdf(f[0], df[0], df[2]), 1 - scipy.stats.f.cdf(f[1], df[1], df[2]), "", ""]
		
		dataframe = pd.DataFrame({"SS": pd.Series(ss, index = rowname), "DF": pd.Series(df, index = rowname), "MS": pd.Series(ms, index = rowname), 
			"F": pd.Series(f, index = rowname), "p-value": pd.Series(p_value, index = rowname)})
		dataframe = dataframe[["SS", "DF", "MS", "F", "p-value"]]
		return dataframe 

	elif n < 0:
		return 'vaild repeat size'
	else:
		total_mean, tss, colss, rowss, crss = 0, 0, 0, 0, 0

		for i in col:
			for j in i:
				total_mean += sum(j) / (len(col) * len(row) * n) 

		tss, colss, rowss, crss, wss = 0, 0, 0, 0, 0

		col_mean = []
		row_mean = []
		for i in range(len(col)):
			temp = 0
			for j in range(len(col[0])):
				temp += sum(col[i][j])
			col_mean.append(temp)

		for i in range(len(row)):
			temp = 0
			for j in range(len(row[0])):
				temp += sum(row[i][j])
			row_mean.append(temp)


		for i in range(len(col)):
			for j in range(len(row)):
				for k in range(n):
					tss += (col[i][j][k] -total_mean) ** 2
					colss += (col_mean[i] - total_mean) ** 2
					rowss += (row_mean[j] - total_mean) ** 2
					crss += ((sum(col[i][j]) / n) - col_mean[i] - row_mean[j] + total_mean) ** 2
					wss += (col[i][j][k] - (sum(col[i][j]) / n)) ** 2


		colmss = colss / (len(col) - 1)
		rowmss = rowss / (len(row) - 1)
		crmss = crss / ((len(row) - 1) * (len(col) - 1))
		wmss = wss / (len(col)*len(row)*(n-1))

		rowname = ["Row", "Column", "interaction", "Error", "Total"]
		ss = [rowss, colss, crss, wss, tss]
		df = [len(row) - 1, len(col) - 1, ((len(row) - 1) * (len(col) - 1)), (len(col) * len(row) * (n - 1)), (len(row) * len(col) * n) - 1]
		ms = [rowmss, colmss, crmss, wmss,""]
		f = [rowmss / wmss, colmss / wmss, crmss / wmss, "", ""]
		p_value = [1 - scipy.stats.f.cdf(f[0], df[0], df[3]), 1 - scipy.stats.f.cdf(f[1], df[1], df[3]), 1 - scipy.stats.f.cdf(f[2], df[2], df[3]), "", ""]
		
		dataframe = pd.DataFrame({"SS": pd.Series(ss, index = rowname), "DF": pd.Series(df, index = rowname), "MS": pd.Series(ms, index = rowname), 
			"F": pd.Series(f, index = rowname), "p-value": pd.Series(p_value, index = rowname)})
		dataframe = dataframe[["SS", "DF", "MS", "F", "p-value"]]
		return dataframe 
