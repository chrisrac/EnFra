# -*- coding: utf-8 -*-
"""
Created February 2025

@author: Krzysztof Raczynski
"""

import pandas as pd

# Processing, merging and saving

# EDIT BLOCK

# path to anaylis outputs from step 1 scripts
path = "path_to_data"


# EXECUTION BLOCK

#ranges for save step
datarange = range(200,3400,100)
rqarange = range(20,3300,10)


# read initial data
dfa = pd.read_csv(path + path + "/dfa_100.csv")
genH = pd.read_csv(path + "/genH_100.csv")
lyapunov_dimM = pd.read_csv(path + "/lyapunov_dimM_100.csv")
lyapunov_dimN = pd.read_csv(path + "/lyapunov_dimN_100.csv")
lyapunov_exp = pd.read_csv(path + "/lyapunov_exp_100.csv")
mfdfa = pd.read_csv(path + "/mfdfa_100.csv")
rs = pd.read_csv(path + "/rs_100.csv")
sampEnt = pd.read_csv(path + "/sampEnt_100.csv")
wtmm_coeffs_max = pd.read_csv(path + "/wtmm_coeffs_max_100.csv")
wtmm_coeffs_mean = pd.read_csv(path + "/wtmm_coeffs_mean_100.csv")
wtmm_coeffs_norm = pd.read_csv(path + "/wtmm_coeffs_norm_100.csv")
wtmm_modmax_max = pd.read_csv(path + "/wtmm_modmax_max_100.csv")
wtmm_modmax_mean = pd.read_csv(path + "/wtmm_modmax_mean_100.csv")
wtmm_modmax_norm = pd.read_csv(path + "/wtmm_modmax_norm_100.csv")
wtmm_scales_max = pd.read_csv(path + "/wtmm_scales_max_100.csv")
wtmm_scales_mean = pd.read_csv(path + "/wtmm_scales_mean_100.csv")
wtmm_scales_norm = pd.read_csv(path + "/wtmm_scales_norm_100.csv")
wtmm_slopes_max = pd.read_csv(path + "/wtmm_slopes_max_100.csv")
wtmm_slopes_mean = pd.read_csv(path + "/wtmm_slopes_mean_100.csv")
wtmm_slopes_norm = pd.read_csv(path + "/wtmm_slopes_norm_100.csv")
lyapunov_dimM = pd.read_csv(path + "/lyapunov_dimM_100.csv")
lyapunov_dimN = pd.read_csv(path + "/lyapunov_dimN_100.csv")
lyapunov_exp = pd.read_csv(path + "/lyapunov_exp_100.csv")
RQA_up_W = pd.read_csv(path + "/RQA_up_W_10.csv")
RQA_up_M = pd.read_csv(path + "/RQA_up_M_10.csv")
RQA_up_Q = pd.read_csv(path + "/RQA_up_Q_10.csv")
RQA_up_A = pd.read_csv(path + "/RQA_up_A_10.csv")
RQA_mean_D = pd.read_csv(path + "/RQA_mean_D_10.csv")
RQA_mean_W = pd.read_csv(path + "/RQA_mean_W_10.csv")
RQA_mean_M = pd.read_csv(path + "/RQA_mean_M_10.csv")
RQA_mean_Q = pd.read_csv(path + "/RQA_mean_Q_10.csv")
RQA_mean_A = pd.read_csv(path + "/RQA_mean_A_10.csv")
RQA_low_W = pd.read_csv(path + "/RQA_low_W_10.csv")
RQA_low_M = pd.read_csv(path + "/RQA_low_M_10.csv")
RQA_low_Q = pd.read_csv(path + "/RQA_low_Q_10.csv")
RQA_low_A = pd.read_csv(path + "/RQA_low_A_10.csv")


# merge outputs
for i in datarange:
    try:
        temp_dfa = pd.read_csv(path + "/dfa_"+str(i)+".csv")
        dfa = pd.concat([dfa, temp_dfa])
    except:
        print("DFA for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_genH = pd.read_csv(path + "/genH_"+str(i)+".csv")
        genH = pd.concat([genH, temp_genH])
    except:
        print("genH for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_lyapunov_dimM = pd.read_csv(path + "/lyapunov_dimM_"+str(i)+".csv")
        lyapunov_dimM = pd.concat([lyapunov_dimM, temp_lyapunov_dimM])
    except:
        print("lyapunov_dimM for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_lyapunov_dimN = pd.read_csv(path + "/lyapunov_dimN_"+str(i)+".csv")
        lyapunov_dimN = pd.concat([lyapunov_dimN, temp_lyapunov_dimN])
    except:
        print("lyapunov_dimN for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_lyapunov_exp = pd.read_csv(path + "/lyapunov_exp_"+str(i)+".csv")
        lyapunov_exp = pd.concat([lyapunov_exp, temp_lyapunov_exp])
    except:
        print("lyapunov_exp for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_rs = pd.read_csv(path + "/rs_"+str(i)+".csv")
        rs = pd.concat([rs, temp_rs])
    except:
        print("rs for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_mf = pd.read_csv(path + "/mfdfa_"+str(i)+".csv")
        mfdfa = pd.concat([mfdfa, temp_mf])
    except:
        print("mfdfa for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_sampEnt = pd.read_csv(path + "/sampEnt_"+str(i)+".csv")
        sampEnt = pd.concat([sampEnt, temp_sampEnt])
    except:
        print("sampEnt for " + str(i) + " timestep not available, proceeding to next")
        # ---------------------------
    try:
        temp_wtmm_coeffs_max = pd.read_csv(path + "/wtmm_coeffs_max_"+str(i)+".csv")
        wtmm_coeffs_max = pd.concat([wtmm_coeffs_max, temp_wtmm_coeffs_max])
    except:
        print("wtmm_coeffs_max for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_wtmm_coeffs_mean = pd.read_csv(path + "/wtmm_coeffs_mean_"+str(i)+".csv")
        wtmm_coeffs_mean = pd.concat([wtmm_coeffs_mean, temp_wtmm_coeffs_mean])
    except:
        print("wtmm_coeffs_mean for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_wtmm_coeffs_norm = pd.read_csv(path + "/wtmm_coeffs_norm_"+str(i)+".csv")
        wtmm_coeffs_norm = pd.concat([wtmm_coeffs_norm, temp_wtmm_coeffs_norm])
    except:
        print("wtmm_coeffs_norm for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_wtmm_modmax_max = pd.read_csv(path + "/wtmm_modmax_max_"+str(i)+".csv")
        wtmm_modmax_max = pd.concat([wtmm_modmax_max, temp_wtmm_modmax_max])
    except:
        print("wtmm_modmax_max for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_wtmm_modmax_mean = pd.read_csv(path + "/wtmm_modmax_mean_"+str(i)+".csv")
        wtmm_modmax_mean = pd.concat([wtmm_modmax_mean, temp_wtmm_modmax_mean])
    except:
        print("wtmm_modmax_mean for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_wtmm_modmax_norm = pd.read_csv(path + "/wtmm_modmax_norm_"+str(i)+".csv")
        wtmm_modmax_norm = pd.concat([wtmm_modmax_norm, temp_wtmm_modmax_norm])
    except:
        print("wtmm_modmax_norm for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_wtmm_scales_max = pd.read_csv(path + "/wtmm_scales_max_"+str(i)+".csv")
        wtmm_scales_max = pd.concat([wtmm_scales_max, temp_wtmm_scales_max])
    except:
        print("wtmm_scales_max for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_wtmm_scales_mean = pd.read_csv(path + "/wtmm_scales_mean_"+str(i)+".csv")
        wtmm_scales_mean = pd.concat([wtmm_scales_mean, temp_wtmm_scales_mean])
    except:
        print("wtmm_scales_mean for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_wtmm_scales_norm = pd.read_csv(path + "/wtmm_scales_norm_"+str(i)+".csv")
        wtmm_scales_norm = pd.concat([wtmm_scales_norm, temp_wtmm_scales_norm])
    except:
        print("wtmm_scales_norm for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_wtmm_slopes_max = pd.read_csv(path + "/wtmm_slopes_max_"+str(i)+".csv")
        wtmm_slopes_max = pd.concat([wtmm_slopes_max, temp_wtmm_slopes_max])
    except:
        print("wtmm_slopes_max for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_wtmm_slopes_mean = pd.read_csv(path + "/wtmm_slopes_mean_"+str(i)+".csv")
        wtmm_slopes_mean = pd.concat([wtmm_slopes_mean, temp_wtmm_slopes_mean])
    except:
        print("wtmm_slopes_mean for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_wtmm_slopes_norm = pd.read_csv(path + "/wtmm_slopes_norm_"+str(i)+".csv")
        wtmm_slopes_norm = pd.concat([wtmm_slopes_norm, temp_wtmm_slopes_norm])
    except:
        print("wtmm_slopes_norm for " + str(i) + " timestep not available, proceeding to next")
  
    try:
        temp_lyapunov_dimM = pd.read_csv(path + "/lyapunov_dimM_"+str(i)+".csv")
        lyapunov_dimM = pd.concat([lyapunov_dimM, temp_lyapunov_dimM])
    except:
        print("lyapunov_dimM for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_lyapunov_dimN = pd.read_csv(path + "/lyapunov_dimN_"+str(i)+".csv")
        lyapunov_dimN = pd.concat([lyapunov_dimN, temp_lyapunov_dimN])
    except:
        print("lyapunov_dimN for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_lyapunov_exp = pd.read_csv(path + "/lyapunov_exp_"+str(i)+".csv")
        lyapunov_exp = pd.concat([lyapunov_exp, temp_lyapunov_exp])
    except:
        print("lyapunov_exp for " + str(i) + " timestep not available, proceeding to next")


for i in rqarange:
    try:
        temp_RQA_up_W = pd.read_csv(path + "/RQA_up_W_"+str(i)+".csv")
        RQA_up_W = pd.concat([RQA_up_W, temp_RQA_up_W])
    except:
        print("RQA_up_W for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_RQA_up_M = pd.read_csv(path + "/RQA_up_M_"+str(i)+".csv")
        RQA_up_M = pd.concat([RQA_up_M, temp_RQA_up_M])
    except:
        print("RQA_up_M for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_RQA_up_Q = pd.read_csv(path + "/RQA_up_Q_"+str(i)+".csv")
        RQA_up_Q = pd.concat([RQA_up_Q, temp_RQA_up_Q])
    except:
        print("RQA_up_Q for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_RQA_up_A = pd.read_csv(path + "/RQA_up_A_"+str(i)+".csv")
        RQA_up_A = pd.concat([RQA_up_A, temp_RQA_up_A])
    except:
        print("RQA_up_A for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_RQA_mean_D = pd.read_csv(path + "/RQA_mean_D_"+str(i)+".csv")
        RQA_mean_D = pd.concat([RQA_mean_D, temp_RQA_mean_D])
    except:
        print("RQA_mean_D for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_RQA_mean_W = pd.read_csv(path + "/RQA_mean_W_"+str(i)+".csv")
        RQA_mean_W = pd.concat([RQA_mean_W, temp_RQA_mean_W])
    except:
        print("RQA_mean_W for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_RQA_mean_M = pd.read_csv(path + "/RQA_mean_M_"+str(i)+".csv")
        RQA_mean_M = pd.concat([RQA_mean_M, temp_RQA_mean_M])
    except:
        print("RQA_mean_M for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_RQA_mean_Q = pd.read_csv(path + "/RQA_mean_Q_"+str(i)+".csv")
        RQA_mean_Q = pd.concat([RQA_mean_Q, temp_RQA_mean_Q])
    except:
        print("RQA_mean_Q for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_RQA_mean_A = pd.read_csv(path + "/RQA_mean_A_"+str(i)+".csv")
        RQA_mean_A = pd.concat([RQA_mean_A, temp_RQA_mean_A])
    except:
        print("RQA_mean_A for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_RQA_low_W = pd.read_csv(path + "/RQA_low_W_"+str(i)+".csv")
        RQA_low_W = pd.concat([RQA_low_W, temp_RQA_low_W])
    except:
        print("RQA_low_W for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_RQA_low_M = pd.read_csv(path + "/RQA_low_M_"+str(i)+".csv")
        RQA_low_M = pd.concat([RQA_low_M, temp_RQA_low_M])
    except:
        print("RQA_low_M for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_RQA_low_Q = pd.read_csv(path + "/RQA_low_Q_"+str(i)+".csv")
        RQA_low_Q = pd.concat([RQA_low_Q, temp_RQA_low_Q])
    except:
        print("RQA_low_Q for " + str(i) + " timestep not available, proceeding to next")
        
    try:
        temp_RQA_low_A = pd.read_csv(path + "/RQA_low_A_"+str(i)+".csv")
        RQA_low_A = pd.concat([RQA_low_A, temp_RQA_low_A])
    except:
        print("RQA_low_A for " + str(i) + " timestep not available, proceeding to next")


# fix index
dfa.set_index("Unnamed: 0", drop=True, inplace=True)
mfdfa.set_index("Unnamed: 0", drop=True, inplace=True)
genH.set_index("Unnamed: 0", drop=True, inplace=True)
rs.set_index("Unnamed: 0", drop=True, inplace=True)
sampEnt.set_index("Unnamed: 0", drop=True, inplace=True)
wtmm_coeffs_max.set_index("Unnamed: 0", drop=True, inplace=True)
wtmm_coeffs_mean.set_index("Unnamed: 0", drop=True, inplace=True)
wtmm_coeffs_norm.set_index("Unnamed: 0", drop=True, inplace=True)
wtmm_modmax_max.set_index("Unnamed: 0", drop=True, inplace=True)
wtmm_modmax_mean.set_index("Unnamed: 0", drop=True, inplace=True)
wtmm_modmax_norm.set_index("Unnamed: 0", drop=True, inplace=True)
wtmm_scales_max.set_index("Unnamed: 0", drop=True, inplace=True)
wtmm_scales_mean.set_index("Unnamed: 0", drop=True, inplace=True)
wtmm_scales_norm.set_index("Unnamed: 0", drop=True, inplace=True)
wtmm_slopes_max.set_index("Unnamed: 0", drop=True, inplace=True)
wtmm_slopes_mean.set_index("Unnamed: 0", drop=True, inplace=True)
wtmm_slopes_norm.set_index("Unnamed: 0", drop=True, inplace=True)
lyapunov_dimM.set_index("Unnamed: 0", drop=True, inplace=True)
lyapunov_dimN.set_index("Unnamed: 0", drop=True, inplace=True)
lyapunov_exp.set_index("Unnamed: 0", drop=True, inplace=True)
RQA_up_W.set_index("Unnamed: 0", drop=True, inplace=True)
RQA_up_M.set_index("Unnamed: 0", drop=True, inplace=True)
RQA_up_Q.set_index("Unnamed: 0", drop=True, inplace=True)
RQA_up_A.set_index("Unnamed: 0", drop=True, inplace=True)
RQA_mean_D.set_index("Unnamed: 0", drop=True, inplace=True)
RQA_mean_W.set_index("Unnamed: 0", drop=True, inplace=True)
RQA_mean_M.set_index("Unnamed: 0", drop=True, inplace=True)
RQA_mean_Q.set_index("Unnamed: 0", drop=True, inplace=True)
RQA_mean_A .set_index("Unnamed: 0", drop=True, inplace=True)
RQA_low_W.set_index("Unnamed: 0", drop=True, inplace=True)
RQA_low_M.set_index("Unnamed: 0", drop=True, inplace=True)
RQA_low_Q.set_index("Unnamed: 0", drop=True, inplace=True)
RQA_low_A.set_index("Unnamed: 0", drop=True, inplace=True)


# save merged outputs
dfa.to_csv(path + "/dfa.csv")
genH.to_csv(path + "/genH.csv")
lyapunov_dimM.to_csv(path + "/lyapunov_dimM.csv")
lyapunov_dimN.to_csv(path + "/lyapunov_dimN.csv")
lyapunov_exp.to_csv(path + "/lyapunov_exp.csv")
mfdfa.to_csv(path + "/mfdfa.csv")
rs.to_csv(path + "/rs.csv")
sampEnt.to_csv(path + "/sampEnt.csv")
wtmm_coeffs_max.to_csv(path + "/wtmm_coeffs_max.csv")
wtmm_coeffs_mean.to_csv(path + "/wtmm_coeffs_mean.csv")
wtmm_coeffs_norm.to_csv(path + "/wtmm_coeffs_norm.csv")
wtmm_modmax_max.to_csv(path + "/wtmm_modmax_max.csv")
wtmm_modmax_mean.to_csv(path + "/wtmm_modmax_mean.csv")
wtmm_modmax_norm.to_csv(path + "/wtmm_modmax_norm.csv")
wtmm_scales_max.to_csv(path + "/wtmm_scales_max.csv")
wtmm_scales_mean.to_csv(path + "/wtmm_scales_mean.csv")
wtmm_scales_norm.to_csv(path + "/wtmm_scales_norm.csv")
wtmm_slopes_max.to_csv(path + "/wtmm_slopes_max.csv")
wtmm_slopes_mean.to_csv(path + "/wtmm_slopes_mean.csv")
wtmm_slopes_norm.to_csv(path + "/wtmm_slopes_norm.csv")
lyapunov_dimM.to_csv(path + "/lyapunov_dimM.csv")
lyapunov_dimN.to_csv(path + "/lyapunov_dimN.csv")
lyapunov_exp.to_csv(path + "/lyapunov_exp.csv")
RQA_up_W.to_csv(path + "/RQA_up_W.csv")
RQA_up_M.to_csv(path + "/RQA_up_M.csv")
RQA_up_Q.to_csv(path + "/RQA_up_Q.csv")
RQA_up_A.to_csv(path + "/RQA_up_A.csv")
RQA_mean_D.to_csv(path + "/RQA_mean_D.csv")
RQA_mean_W.to_csv(path + "/RQA_mean_W.csv")
RQA_mean_M.to_csv(path + "/RQA_mean_M.csv")
RQA_mean_Q.to_csv(path + "/RQA_mean_Q.csv")
RQA_mean_A.to_csv(path + "/RQA_mean_A.csv")
RQA_low_W.to_csv(path + "/RQA_low_W.csv")
RQA_low_M.to_csv(path + "/RQA_low_M.csv")
RQA_low_Q.to_csv(path + "/RQA_low_Q.csv")
RQA_low_A.to_csv(path + "/RQA_low_A.csv")