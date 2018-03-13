# -*- coding: utf-8 -*-
"""
Spyder Editor

Source code for New Imputation Method
"""
import statistics as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
from pandas import Categorical
from nmilib import *


    
#S is input set of patient records    
datatable = pd.read_csv("data2.txt",na_values=["",'?'])
print('rawdata =\n',datatable)
data = prepare_numeric_data(datatable)

cats = Categorical([True,False,'+','-','yes','YES','Yes','no','NO','No'])
# M is the subset of S having records without missing values in the nth column
M = filter_mv_decisions(data)
print("\n\ndata without mv in decision column")
print(M)
#print('M Frame values')
#print(M.iloc[0:,0:])

#compute idices
#print("\nid(M) before =",id(M))
IC = compute_indices(pd.DataFrame(M))
print("\n IC =\n",IC)
#compute the indices


dmatrix = compute_distance(list(IC))

mv_record_index_list = [0,]
#dmatrix = [[0,0.99,1.11,1.46],[0.99,0,2.11,2.27],[1.11,2.11,0,2.32],[1.46,2.27,2.32,0]]
dmatrix = [[0,0.86,0.99],[0.86,0,1.32],[0.99,1.32,0]]

#dmatrix = []
zdmatrix = compute_zd(dmatrix,mv_record_index_list,len(dmatrix))
print("zd = ",zdmatrix)

#zdmatrix = compute_zd(dmatrix,1)
#counter = 0
