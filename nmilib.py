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

# read data from the file



def filter_mv_decisions(data):
    #nmv = data.iloc[:,-1]!='?'    
    data1 = pd.DataFrame(data)
    nmv = (data1.iloc[:,-1]).notnull()    
    return( data1[nmv])


def strip_MV_Records(Samples):
    temp = pd.DataFrame(Samples)
    temp_index = temp.index
    #print("\ntemp.index = ",temp_index) 
    for i in temp.index:    
       
        #print("\ntemp.loc[i] =",temp.loc[i])
        nmv = (temp.loc[i]).isnull()
        #print("\nnmv = \n",nmv)
        #print("\nnmv.notnull() =",nmv.notnull())
        #print("\ntemp[nmv] =",temp.loc[nmv])
        if nmv.any():
            temp.drop(i,inplace=True)
#        if nmv.any():
#            temp.drop(temp.loc[i],axis=0,inplace=True)
    
    #print("\n\n\n\ndata after stripping MV\n",temp)
    return (temp)
    

#Recover numeric data
def prepare_numeric_data(datatable):
    for i in datatable.columns:    
        datatable[i] = pd.to_numeric(datatable[i],errors='ignore')
        #print("\n\n",datatable)
    return datatable
    
#replace MV with np.nan exists other than decision cloumn

#compute indices 
#input M:data without MV in decision column
def compute_indices(M):
    ic_panel = []    
    IC=[]
    #M=Samples = SNMV    
    
    R =  M.index
    C = M.columns
    col_index = C.get_loc
    
    
    #mark categorical element types       
    
    mv_row_index = []
    mv_col_index = []  
    non_mv_row_index = []
      
    
    for col in C: 
        # we do not need decision column here, we only need columns with missing attributes
        if col == C[-1]:
            break           
        for row in R:
            #find the index of missing row and col and store in mv_row and mv_col
            if (pd.isnull((M[col][row]))) :
                mv_col_index.append(C.get_loc(col)) 
                mv_row_index.append(R.get_loc(row))
    
    for row in R:
        #print("row =\n",row)
        #print("M[row] =\n",M.loc[row])
        #nn = pd.notnull(M[row])
        
        nn = (M.loc[row]).notnull()
        #print("\nnn =",nn)
        if nn.all():
            non_mv_row_index.append(R.get_loc(row))
            
    print("\nnon_mv_row_index\n",non_mv_row_index)               
    print("\n\nmv_row_index=",mv_row_index)
    print("\n\nmv_col_index =",mv_col_index)
    
    dec_class_yes  = M[M[C[-1]].isin(['+', 'yes', 'YES' , 1,True])]
    
    print("\n\ndec_class_yes = ",dec_class_yes)    
    
    dec_class_no  = M[M[C[-1]].isin(['-' , 'no' , 'NO' , 'No',0,False])]
    print("\n\ndec_class_no = ",dec_class_no)
    
    #calculate IC index based on column having the missing value
    atrib_type  = 0 # will be decided in the loop    
    ICmaster = [] #IC is a 3d panda structure containing values
    
    for l in  mv_col_index:        
        IC_i_list = []
        vals = M[C[l][:]]
        val_exists = vals.isin([True,False,'+','-','yes','YES','Yes','no','NO','No'])
        #for i in mv_row_index:
        #As per NMI we need to strip all tuples with MV while calculating IC
        for i in non_mv_row_index:
            IC_k_list = []
            
            for k in non_mv_row_index: 
                
                last_col_index = C.get_loc(C[-1])
                print("\nlast_col_index=",last_col_index)
                I = M.iloc[i,last_col_index]
                K = M.iloc[k,last_col_index]
                print("\nI =",I," K=",K)
                
                                    
                #if i and k belongs to same decision class                
                if I == K: #same decision  class
                   print("\n\ncase-1:same decision class")  
                   #IC for same record is 0                
                   if i==k:
                        print("RECORDS ARE SAME, IC =0")
                        IC_k_list.insert(k,0)
                   #group the elements into related class                  
                   
                   #print("\n\nvals =\n",val_exists.any())       
                   elif val_exists.any():                       
                       print('\n\ncase-1:categorical')
                       ic_val = compute_case1_categorical(pd.DataFrame(M),C,l,i,k,non_mv_row_index)
                       IC_k_list.insert(k,ic_val)
                  
                   elif (M[C[l]].dtypes == "float64"):
                       print('\n\ncase-1:type is float')
                       ic_val = compute_case1_fractions(pd.DataFrame(M),C,l,i,k,non_mv_row_index)
                       IC_k_list.insert(k,ic_val)
                   else:
                       print('\n\ncase-1:tInput Error, column type could not be determined\n')
                else: #i and k belongs to different decision class
                    print("\n\ncase-2:Different decision class ")
                    
                    if val_exists.any():
                        print('\n\ncase-2:categorical')
                        ic_val = compute_case2_categorical(pd.DataFrame(M),C,l,i,k,non_mv_row_index)
                        IC_k_list.insert(k,ic_val)
                    
                    elif (M[C[l]].dtypes == "float64"):
                        print('\n\ncase-2:type is float')
                        ic_val = compute_case2_fractions(pd.DataFrame(M),C,l,i,k,non_mv_row_index)
                        IC_k_list.insert(k,ic_val)
                    else:
                        print('\n\ncase-2:Input Error, column type could not be determined\n')
                        
            IC_i_list.insert(i,pd.Series(IC_k_list))  
        ICmaster.insert(l,pd.DataFrame(IC_i_list))

    return ICmaster
    

def compute_distance(IC):
    
    
    d= [[0,0.99,1.11,1.46],[0.99,0,2.11,2.27],[1.11,2.11,0,2.32],[1.46,2.27,2.32,0]]
#    distance = []
#    print('\n\nlenght of d =',len(d))
#    d_ik = 0
#    for i in IC:
#        for j in i:
#            for k in j:
#                d_ik+=(k**2)
#                        
#            distance.append(d_ik**0.5)    
#            print('\n\ndistance =',distance)
    
    return d
    #+return distance
    
    
#computing Z(d) as per New Imputation Method
#d is the set of distances calculated as per IEEE paper
#mv_index is the index of record Ri with missing value
#no_records varying index from 1...m where m is total records of set R
def compute_zd(d,mv_index_list,no_records):
    
    zd=[]
    m = 1/(no_records-1)
    for i in mv_index_list:
        #column wise mean w
        #when considered total mean getting error
        mean_val = st.mean(d[i][:])
        z_val = []
        for j in range(0,no_records):
                        
            z_val.append((d[i][j] -mean_val )/((m*get_sigma(d,i,j,1,len(d),mean_val))**0.5))
        
        zd.append([i, min(z_val)])
        
            
    return zd
    
def get_sigma(d,mv_index,cur_pos,start,end,mean_d):
    sig_val=0
    
    for i in range(start,end):
        sig_val+=(d[i][cur_pos]-mean_d)
        
    return sig_val  

    
#program for computing coordinates for case-1 categorical type data

def compute_case1_categorical(A,column_lables,mv_col,i,k,non_mv_row_index):
    #A is the data table pandas DataFrame
    #ci sereis of column indices
    #mv_col column with missed value
    #i is Record-i
    #k is Record-k
    C = A.columns
    class_of_i = A.iloc[i,C.get_loc(C[-1])]
    print("\nclass_of_i =",class_of_i)
    print("\nbefore A =",A)
    A_NO_MV = strip_MV_Records(A)
    
    print("\nafter A =",A_NO_MV)
    req_col = (A_NO_MV[A_NO_MV.columns[-1]]==class_of_i)
    print("\nreq_col =",req_col)
    A_NO_MV = A_NO_MV[req_col]
    print("A =",A_NO_MV)
    B = [] #holds the B gamma ps
    B = A.index
    print("B =",B)
    IC_cat = []
    #construct subsets   
    
    return IC_cat

#program for computing coordinates for case-1 fractions type data    
def compute_case1_fractions(A,ci,col,i,k,non_mv_row_index):
    IC_fraction = []
    return IC_fraction
#program for computing IC for case-2 fractions
def compute_case2_fractions(A,ci,col,i,k,non_mv_row_index):
    IC_fraction = []
    return IC_fraction

#program for computing IC for case-2 categorical
def compute_case2_categorical(A,ci,col,i,k,non_mv_row_index):
    IC_cat = []
    return IC_cat
    
