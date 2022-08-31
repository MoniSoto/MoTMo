import numpy as np
import pandas as pd
import itertools
import datetime as dt
import time

import os
from os import listdir
from os.path import isfile,join
import matplotlib.pyplot as plt
from itertools import product

PATH = '/home/moni/Documents/motmo/timeSeries_files/' # original data
PATH2 = '/home/moni/Documents/motmo/data_without_hhID/' # folde docs hhID


num_options_per_category = {
    'investment' : 3,
    'policy' : 3,
    'event' : 4
}
# ------------[INPUT SPACE SECTION]--------------------
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

def input_bool_to_decimal_dicts(category_options_dict):
    input_space_decimal_dict = {}
    
    for category, num_options in category_options_dict.items():
        valid_scenarios = valid_scenarios_for_category(num_options)
        num_scenarios = len(valid_scenarios)
        scenerios_decimal = list(range(0,num_scenarios,1))
        scenerios_decimal = [i/(num_scenarios - 1) for i in scenerios_decimal]
        bool_to_decimal_dict = dict(zip(valid_scenarios,scenerios_decimal))
        input_space_decimal_dict[category] = bool_to_decimal_dict
    
    return input_space_decimal_dict

def input_space_decimal(category_options_dict):
    input_bool = generate_input_space_bool(category_options_dict)
    inputs_d = []
    k=0
    for i in input_bool:
        t = [category_options_dict['investment'][i[0]], category_options_dict['policy'][i[1]], category_options_dict['event'][i[2]]]
        inputs_d.append(t)
    return inputs_d

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

# ------------[USEFUL DICTIONARIES/LISTS SECTION]--------------------
def get_scenario_string(boolean_scenario): 
    bool_sc = list(sum(boolean_scenario,())) # flatten list
    list_events = ['CH','SP','SE','WE','BP','RE','CO','DI','WO','CS']
    s_n = [l+str(s) for l,s in zip(list_events,bool_sc)]
    s_n = ''.join(s_n)
    scenario_name = s_n
    return scenario_name

def get_region_ids():
    return [942, 1515, 1516, 1517, 1518, 1519, 1520, 2331, 2332, 2333, 2334, 2335, 2336, 3312, 3562, 6321]

def get_best_scenarios_dict():
    best_sc_dict = [{'CH1SP1SE0WE1BP0RE1CO0DI1WO1CS0': ['Schleswig-Holstein', 'Bremen',
    'Hamburg']}, 
    {'CH1SP1SE0WE1BP0RE1CO1DI1WO0CS0': ['Nordrhein-Westfalen', 'Saarland',
    'Bayern', 'Mecklenburg-Vorpommern', 'Brandenburg', 'Niedersachsen']},
    {'CH0SP1SE0WE1BP0RE1CO1DI1WO0CS0': ['Baden-Wurttemberg', 'Hessen',
    'Thuringen', 'Sachsen-Anhalt', 'Sachsen']}, 
    {'CH0SP1SE0WE1BP0RE1CO0DI1WO0CS1': ['Rheinland-Pfalz']},
    {'CH0SP1SE0WE0BP1RE1CO1DI0WO0CS0': ['Berlin']}]
    return best_sc_dict

def get_stock_vars():
    return ['stock_C','stock_E','stock_N','stock_P','stock_S']

def get_emission_vars():
    return ['emissions_C','emissions_E','emissions_N','emissions_P','emissions_S']

def get_dict_steps_dates():
    dates_list = pd.date_range(start = "2005-01-01",periods = 181,freq="2M").strftime("%b-%Y").tolist()
    dict_dates = dict(zip(list(range(0,181)),dates_list))
    return dict_dates

def list_file_names(): # gets all file names (scenarios)
    file_names = [f for f in listdir(PATH2) if isfile(join(PATH2, f))]
    if '.~lock.timeSeries_CH0SP0SE0WE0BP0RE0CO0DI0WO0CS0.csv#' in file_names:
        file_names.remove('.~lock.timeSeries_CH0SP0SE0WE0BP0RE0CO0DI0WO0CS0.csv#')
    return file_names
def list_str_scenarios(file_names):# gets the full string and ignores the csv extension and prefix 'timeSeries_'
    sc_str = [name.replace(".csv", "") for name in file_names]
    sc_str = [name.replace("timeSeries_", "") for name in sc_str]
    if '.~lock.CH0SP0SE0WE0BP0RE0CO0DI0WO0CS0#' in sc_str: # there is a weird filename
        sc_str.remove('.~lock.CH0SP0SE0WE0BP0RE0CO0DI0WO0CS0#')
    return sc_str

def get_dict_regions():
    region_names = ["Schleswig-Holstein","Nordrhein-Westfalen","Baden-Wurttemberg","Hessen","Bremen","Thuringen",
                     "Hamburg","Rheinland-Pfalz","Saarland","Bayern","Berlin","Sachsen-Anhalt","Sachsen",
                     "Mecklenburg-Vorpommern","Brandenburg","Niedersachsen"]
    region_keys = get_region_ids()
    
    dict_reg = {region_keys[i]: region_names[i] for i in range(len(region_keys))}
    return dict_reg

# ------------[SOME METRICS (PER REGIONS) SECTION]--------------------
def get_tot_emi_per_region(file_name):
    # this 'file_name' has to be a full string with prefix and .csv extension of a scenario!
    region_dict = get_dict_regions()
    reg_df = pd.DataFrame.from_dict(region_dict, orient='index', columns=['region_name'])
    reg_keys = region_dict.keys()
    values = []
    for regID in reg_keys:
        df = pd.read_csv(PATH2 + file_name)
        emi_reg = df[df['reID']==regID]['total_emissions'].sum()
        values.append(emi_reg)
    reg_df['total_emissions']=values
    reg_df['proportion'] = reg_df.total_emissions / reg_df.total_emissions.sum()
    return reg_df

def get_contri_plot_emis_region(df):
    ax = df.plot.bar(x='region_name', y='proportion', title = "Contribution to emissions per region")

# this function takes as input a dataframe with all metrics
# of all regions and returns the metrics of an specific
# region in a dataframe
def get_metrics_region(df, region_id):
    #
    mask = df['reID']==region_id
    return df[mask]

def read_metrics_scenario(scenario_file_name):
    met_df = pd.read_csv(PATH2 + scenario_file_name)
    met_df = met_df.drop(['Unnamed: 0'],axis=1)
    dict_reg = get_dict_regions()
    # reg_names = list(dict_reg.values())
    reg_ids = list(dict_reg.keys())
    for RID in reg_ids:
        mask = (met_df['reID'] == RID)
        met_df.loc[mask,'region_name'] = dict_reg[RID]
    met_df = met_df[[ 'step','reID','region_name','emissions_C','emissions_E','emissions_N','emissions_P','emissions_S','stock_C','stock_E','stock_N','stock_P','stock_S','total_emissions']]
    # the previous line puts the column of the region name right after the region id column.
    return met_df

def timeStep_stock_emissions_returns(df):
    # the input is a dataframe of a scenario located in PATH2 that has
    # the metrics of all regions and their respective outputs for emissions
    # and stock. The output is the rate of change of each variable (in dataframe)
    stock_vars = get_stock_vars() # variable names of the original df (stock_X)
    stock_vars.append("total_emissions") # fix later, I want to add this col
    prefix = ["change_"]
    new_colnames = list(" ".join(a + b for a, b in product(prefix, stock_vars)).split(" "))
    stock_df = df[['step', 'reID','region_name']] #  (almost) empty dataset
    idList = list(get_dict_regions().keys()) # list of regions ids
    for RID in idList:
        mask = (df['reID'] == RID)
        for name, stock_x in zip(new_colnames, stock_vars):
            stock_df.loc[mask,name] = (df.loc[mask,stock_x] / df.loc[mask,stock_x].shift(1)) -1
    return stock_df.fillna(0)

def get_rate_change_region(df, reg_id):
    df2 = timeStep_stock_emissions_returns(df)
    # make sure that the input dataframe is the result of
    # the function "timeStep_stock_emissions_returns()".
    # This dataframe should contain the rate of change
    # of all regions, and the function returns the rate of
    # change of an specific region.
    return df2[df2['reID']==reg_id]

def get_df_plot_regions(df,start_step,var_name):
    df2 = timeStep_stock_emissions_returns(df)
    dates_dict = get_dict_steps_dates()
    dates_dict = {k:v for (k,v) in dates_dict.items() if k >= start_step}
    dates_list = list(dates_dict.values())

    df_plot = df2[['step','region_name',var_name]].pivot(index='step', columns='region_name', values=var_name).reset_index()
    df_plot['step'] = dates_list
    return df_plot

#-----------------[GLOBAL METRICS SECTION]-----------------
def get_df_emi_change_global(df):
    # this function is wrong. do not use!
    df2 = timeStep_stock_emissions_returns(df)[['step','change_total_emissions']].groupby(['step']).sum().reset_index()
    return df2

def rateChange_all(df):
    # this funciton returns the rate of change per timestep
    # without including info about the regions. the input is a 
    # dataframe corresponding to a scenario (directly read from PATH2)
    new_df = df[['step','total_emissions','stock_C','stock_E','stock_N','stock_P','stock_S']]
    emis_df = new_df.groupby(['step']).sum().reset_index()
    emis_df['change'] = ((emis_df['total_emissions'] / emis_df['total_emissions'].shift(1)) -1).fillna(0)
    return emis_df

def get_sum_vars_scenario(boolean_tuple):
    # INPUT: boolean string representing a category. For example, (1,0,0). 
    # OUTPUT: dataframe with compounded output of each variable.
    # WARNING: 'PATH2' is a local directory and stores all files for each scenario!

    scenario_name = get_scenario_string(boolean_tuple)
    file_name = "timeSeries_" + scenario_name + ".csv"
    scenario_df = pd.read_csv(PATH2 + file_name, index_col=0)
    columns = ['step','stock_C','stock_E','stock_N','stock_P','stock_S','total_emissions']
    scenario_df = scenario_df[columns].groupby(['step']).sum()
    scenario_df = scenario_df.iloc[77:]
    return scenario_df

def get_sum_all_vars(output_vars):
    input_spc_bool = generate_input_space_bool(num_options_per_category)
    # scenarios_string_list = [mo.get_scenario_string(x) for x in input_spc_bool]
    sums_dict = {}
    
    for scenario_bool in input_spc_bool:
        df = get_sum_vars_scenario(scenario_bool)
        sum_dict = df.sum().to_dict()
        scenario_name = get_scenario_string(scenario_bool)
        sum_list = list(sum_dict.values())
        sums_dict[scenario_name] = sum_list
    df = pd.DataFrame.from_dict(sums_dict, orient='index',columns=output_vars)
    return df