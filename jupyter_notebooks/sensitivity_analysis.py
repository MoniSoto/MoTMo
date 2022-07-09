import numpy as np
import itertools
import pandas as pd
import warnings
import random
import datetime as dt
import time
import os
from os import listdir
from os.path import isfile,join
import matplotlib.pyplot as plt
from scipy.stats import t as tstats

num_options_per_category = {
    'investment' : 3,
    'policy' : 3,
    'event' : 4
}

def valid_scenarios_for_category(num_options):
    total_scenarios = list(itertools.product(range(2), repeat=num_options))
    valid_scenarios = [i for i in total_scenarios if sum(i)<3]
    return valid_scenarios

def generate_input_space_bool(category_options_dict):
    valid_scenarios_per_category_list = []
    for category, num_options in category_options_dict.items():
        valid_scenarios = valid_scenarios_for_category(num_options)
        valid_scenarios_per_category_list.append(valid_scenarios)
    
    # 539 input scenarios in dataset
    input_scenarios = list(itertools.product(*valid_scenarios_per_category_list))
    return input_scenarios

def input_space_bool_to_decimal_dicts(category_options_dict):
    input_space_decimal_dict = {}
    
    for category, num_options in category_options_dict.items():
        valid_scenarios = valid_scenarios_for_category(num_options)
        num_scenarios = len(valid_scenarios)
        scenerios_decimal = list(range(0,num_scenarios,1))
        scenerios_decimal = [i/(num_scenarios - 1) for i in scenerios_decimal]
        bool_to_decimal_dict = dict(zip(valid_scenarios,scenerios_decimal))
        input_space_decimal_dict[category] = bool_to_decimal_dict
    
    return input_space_decimal_dict

def input_space_decimal_to_bool_dicts(category_options_dict):
    input_space_decimal_dict = {}
    
    for category, num_options in category_options_dict.items():
        valid_scenarios = valid_scenarios_for_category(num_options)
        num_scenarios = len(valid_scenarios)
        scenerios_decimal = list(range(0,num_scenarios,1))
        scenerios_decimal = [i/(num_scenarios - 1) for i in scenerios_decimal]
        bool_to_decimal_dict = dict(zip(scenerios_decimal,valid_scenarios))
        input_space_decimal_dict[category] = bool_to_decimal_dict
    
    return input_space_decimal_dict

def from_dec_to_bool(dec_to_bool_dicts, row):
    d_I = dec_to_bool_dicts['investment']
    d_P = dec_to_bool_dicts['policy']
    d_E = dec_to_bool_dicts['event']
    rbool = [d_I[row[0]], d_P[row[1]], d_E[row[2]]]
    return rbool

# input is a list of tuples [(1,0,0),(0,1,0),(1,0,0,0)]
# output 'CH1SP0SE0WE0BP1RE0CO1DI0WO0CS0'
def get_scenario_string(boolean_scenario): 
    bool_sc = list(sum(boolean_scenario,())) # flatten list
    list_events = ['CH','SP','SE','WE','BP','RE','CO','DI','WO','CS']
    f_n = [l+str(s) for l,s in zip(list_events,bool_sc)]
    f_n = ''.join(f_n)
    file_name = f_n
    return file_name

def sampling(in_space,N):
    AB = random.sample(in_space,2*N)
    A = AB[0:N]
    B = AB[N:2*N]
    return A,B

def mat_AB_i(A,B): 
    d = 3
    N = len(A)
    An,Bn = np.array(A), np.array(B)
    AB_mat = np.zeros((d,N,d))
    for i in range(0,d):
        AB_mat[i] = An
        AB_mat[i][:,i] = Bn[:,i]
    return AB_mat

def generate_inputs_d(inputs, input_space_bool_to_decimal_dicts):
    inputs_d = []
    for i in inputs:
        dict_I = input_space_bool_to_decimal_dicts['investment']
        dict_P = input_space_bool_to_decimal_dicts['policy']
        dict_E = input_space_bool_to_decimal_dicts['event']
        t = [dict_I[i[0]], dict_P[i[1]], dict_E[i[2]]]
        inputs_d.append(t)
    return inputs_d

# df: dataframe (e.g. total emissions df)
# M: matrix, make sure it is a list
def get_output(df, M, dec_to_bool_dicts):
    out_vector = []
    n = len(M)
    for i in range(0,n):
        dec_row = M[i]
        row_bool = from_dec_to_bool(dec_to_bool_dicts, dec_row)
        string = get_scenario_string(row_bool)
        cond = (df['scenario'] == string)
        outp = df[cond].total_emissions.values[0]
        out_vector.append(outp)
    return out_vector

# this function takes as input the matrices A and B
# and the original matrix ABi. 
# the index i at the end is the variable name, it can be 0, 1 or 2
def get_var_xi(df,A,B,ABi,i,dec_to_bool_dicts):
    ABi = ABi[i].tolist() # add '.astype(int)' in the middle if there are any probs with type
    out_A = get_output(df, A, dec_to_bool_dicts)
    out_B = get_output(df, B, dec_to_bool_dicts)
    out_ABi = get_output(df, ABi, dec_to_bool_dicts)    
    N = len(A)
    var = 0
    for j in range(0,N):
        var = out_B[j]*(out_ABi[j]-out_A[j])+var
    var = var/N
    return var

def calculate_sensitivity_indices(df, col_name, num_samples):
    total_samples = 2*num_samples
    input_space = generate_input_space_bool(num_options_per_category)
    input_bool_dec_dicts = input_space_bool_to_decimal_dicts(num_options_per_category)
    inputs_d = generate_inputs_d(input_space, input_bool_dec_dicts)
    A, B = sampling(inputs_d, total_samples)
    ABi = mat_AB_i(A,B)
    
    input_dec_bool_dicts = input_space_decimal_to_bool_dicts(num_options_per_category)

    varI = get_var_xi(df, A,B,ABi,0, input_dec_bool_dicts)
    varP = get_var_xi(df, A,B,ABi,1, input_dec_bool_dicts)
    varE = get_var_xi(df, A,B,ABi,2, input_dec_bool_dicts)
    
    varY = df[col_name].var(axis=0)
    
    S_I = varI / varY
    S_P = varP / varY
    S_E = varE / varY
    
    sensitivity_indices_dict = {
        'investment' : S_I,
        'policy' : S_P,
        'event' : S_E
    }
    return sensitivity_indices_dict

# dataset input needs to be np array
def calculate_confidence_intervals(dataset, confidence):
    mean = dataset.mean() 
    s_dev = dataset.std() 
    num_datapoints = len(dataset)
    dof = num_datapoints-1 
    t_crit = np.abs(tstats.ppf((1-confidence)/2,dof))
    confidence_interval = (mean-s_dev*t_crit/np.sqrt(num_datapoints), mean+s_dev*t_crit/np.sqrt(num_datapoints)) 
    return confidence_interval

def sensitivity_analysis_experiment(num_samples, num_runs, df, col_name):
    S1_I_arr = []
    S1_P_arr = []
    S1_E_arr = []
    for i in range (0, num_runs):
        sensitivity_indices = calculate_sensitivity_indices(df, 'total_emissions', 8)
        S1_I_arr.append(sensitivity_indices['investment'])
        S1_P_arr.append(sensitivity_indices['policy'])
        S1_E_arr.append(sensitivity_indices['event'])
        
    S1_I_np_arr = np.array(S1_I_arr)
    S1_I = S1_I_np_arr.mean()
    S1_I_conf = calculate_confidence_intervals(S1_I_np_arr, 0.95)
    S1_P_np_arr = np.array(S1_P_arr)
    S1_P = S1_P_np_arr.mean()
    S1_P_conf = calculate_confidence_intervals(S1_P_np_arr, 0.95)
    S1_E_np_arr = np.array(S1_E_arr)
    S1_E = S1_E_np_arr.mean()
    S1_E_conf = calculate_confidence_intervals(S1_E_np_arr, 0.95)
    
    sensitivity_indices_dict = {
        'investment' : {
            'S1' : S1_I,
            'S1_conf' : S1_I_conf
        },
        'policy' : {
            'S1' : S1_P,
            'S1_conf' : S1_P_conf
        },
        'event' : {
            'S1' : S1_E,
            'S1_conf' : S1_E_conf    
        }
    }
    return sensitivity_indices_dict