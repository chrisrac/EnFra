# -*- coding: utf-8 -*-
"""
Created February 2025

@author: Krzysztof Raczynski
"""

import pandas as pd
import numpy as np
from fcmeans import FCM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# EDIT BLOCK
path = 'path_to_mergedData'
output_path = 'output_path'

# Postprocessing
# read data
lyapunov_exp = pd.read_csv(path + 'Lyapunov_exp.csv')
lowA = pd.read_csv(path + 'lowA.csv')
lowQ = pd.read_csv(path + 'lowQ.csv')
lowM = pd.read_csv(path + 'lowM.csv')
lowW = pd.read_csv(path + 'lowW.csv')
meanA = pd.read_csv(path + 'meanA.csv')
meanQ = pd.read_csv(path + 'meanQ.csv')
meanM = pd.read_csv(path + 'meanM.csv')
meanW = pd.read_csv(path + 'meanW.csv')
meanD = pd.read_csv(path + 'meanD.csv')
highA = pd.read_csv(path + 'highA.csv')
highQ = pd.read_csv(path + 'highQ.csv')
highM = pd.read_csv(path + 'highM.csv')
highW = pd.read_csv(path + 'highW.csv')


# fix index
lyapunov_exp.set_index('Unnamed: 0',drop=True,inplace=True)
lowA.set_index('Unnamed: 0', inplace=True,drop=True)
lowQ.set_index('Unnamed: 0', inplace=True,drop=True)
lowM.set_index('Unnamed: 0', inplace=True,drop=True)
lowW.set_index('Unnamed: 0', inplace=True,drop=True)
meanA.set_index('Unnamed: 0', inplace=True,drop=True)
meanQ.set_index('Unnamed: 0', inplace=True,drop=True)
meanM.set_index('Unnamed: 0', inplace=True,drop=True)
meanW.set_index('Unnamed: 0', inplace=True,drop=True)
meanD.set_index('Unnamed: 0', inplace=True,drop=True)
highA.set_index('Unnamed: 0', inplace=True,drop=True)
highQ.set_index('Unnamed: 0', inplace=True,drop=True)
highM.set_index('Unnamed: 0', inplace=True,drop=True)
highW.set_index('Unnamed: 0', inplace=True,drop=True)


# Descriptive statistics
lowA_desc = lowA.describe().T
lowQ_desc = lowQ.describe().T
lowM_desc = lowM.describe().T
lowW_desc = lowW.describe().T
meanA_desc = meanA.describe().T
meanQ_desc = meanQ.describe().T
meanM_desc = meanM.describe().T
meanW_desc = meanW.describe().T
meanD_desc = meanD.describe().T
highA_desc = highA.describe().T
highQ_desc = highQ.describe().T
highM_desc = highM.describe().T
highW_desc = highW.describe().T

# Lyapunov
lyap_vals = pd.DataFrame(columns=['var',1,2,3,4,5,6,7,8,9,10])
for tab in [lyapunov_exp['l_W'],lyapunov_exp['l_M'],lyapunov_exp['l_Q'],lyapunov_exp['l_A'],lyapunov_exp['m_D'],
            lyapunov_exp['m_W'],lyapunov_exp['m_M'],lyapunov_exp['m_Q'],lyapunov_exp['m_A'],lyapunov_exp['u_W'],
            lyapunov_exp['u_M'],lyapunov_exp['u_Q'],lyapunov_exp['u_A']]:
    tab_data = tab.to_frame()
    for row in range(0,len(tab_data)):
        row_name = tab.name
        gauge_name = tab_data.index[row]
        values = tab_data.iloc[row]
        arr = np.fromstring(str(values.values).replace("\\n",'').replace('[','').replace(']','').replace("'",''), sep=' ')
        arr = arr[~np.isnan(arr)]
        q_vals = np.arange(1, len(arr) + 1)
        if len(arr) < 10:
            while len(arr) < 10:
                arr = np.append(arr, np.nan)
        if len(arr) > 10:
            arr = arr[:10]
        lyap_vals.loc[str(gauge_name) + '_' + row_name] = np.append(row_name, arr)
lyap_vals.to_csv(output_path + 'lyap_vals.csv')

lyap_lw = lyap_vals.iloc[:,1:].loc[lyap_vals['var']=='l_W']
lyap_lm = lyap_vals.iloc[:,1:].loc[lyap_vals['var']=='l_M']
lyap_lq = lyap_vals.iloc[:,1:].loc[lyap_vals['var']=='l_Q']
lyap_la = lyap_vals.iloc[:,1:].loc[lyap_vals['var']=='l_A']
lyap_md = lyap_vals.iloc[:,1:].loc[lyap_vals['var']=='m_D']
lyap_mw = lyap_vals.iloc[:,1:].loc[lyap_vals['var']=='m_W']
lyap_mm = lyap_vals.iloc[:,1:].loc[lyap_vals['var']=='m_M']
lyap_mq = lyap_vals.iloc[:,1:].loc[lyap_vals['var']=='m_Q']
lyap_ma = lyap_vals.iloc[:,1:].loc[lyap_vals['var']=='m_A']
lyap_uw = lyap_vals.iloc[:,1:].loc[lyap_vals['var']=='u_W']
lyap_um = lyap_vals.iloc[:,1:].loc[lyap_vals['var']=='u_M']
lyap_uq = lyap_vals.iloc[:,1:].loc[lyap_vals['var']=='u_Q']
lyap_ua = lyap_vals.iloc[:,1:].loc[lyap_vals['var']=='u_A']

# saving outputs
lyap_lw.to_csv(output_path + 'lyap_lw.csv')
lyap_lm.to_csv(output_path + 'lyap_lm.csv')
lyap_lq.to_csv(output_path + 'lyap_lq.csv')
lyap_la.to_csv(output_path + 'lyap_la.csv')
lyap_md.to_csv(output_path + 'lyap_md.csv')
lyap_mw.to_csv(output_path + 'lyap_mw.csv')
lyap_mm.to_csv(output_path + 'lyap_mm.csv')
lyap_mq.to_csv(output_path + 'lyap_mq.csv')
lyap_ma.to_csv(output_path + 'lyap_ma.csv')
lyap_uw.to_csv(output_path + 'lyap_uw.csv')
lyap_um.to_csv(output_path + 'lyap_um.csv')
lyap_uq.to_csv(output_path + 'lyap_uq.csv')
lyap_ua.to_csv(output_path + 'lyap_ua.csv')


# Standardize & PCA
tabs = [lowA,lowQ,lowM,lowW,meanA,meanQ,meanM,meanW,meanD,highA,highQ,highM,highW]
varies = ['lowA','lowQ','lowM','lowW','meanA','meanQ','meanM','meanW','meanD','highA','highQ','highM','highW']
explained_vars = pd.DataFrame(columns=['pca_1','pca_2','pca3'])
clust_size3 = pd.DataFrame(columns=['c1','c2','c3'])

for i in range(0,len(tabs)):
    df = tabs[i]
    df_var = varies[i]
    df = df.replace(-np.inf, np.nan)
    df = df.replace(np.inf, np.nan)
    df = df.drop(['Lyap_1','Lyap_2','Lyap_3','Lyap_4'],axis=1).dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    pca = PCA(n_components=3)
    scores = pca.fit_transform(X)
    loadings = pd.DataFrame(pca.components_.T,index=df.columns,columns=['PC1','PC2','PC3'])
    loadings.to_csv(output_path+'/loadings_'+df_var+'.csv')
    pd.DataFrame(scores).to_csv(output_path+'/scores_'+df_var+'.csv')
    explained_vars.loc[df_var] = pca.explained_variance_ratio_
explained_vars.to_csv(output_path+'/explained_vars_'+df_var+'.csv')

# Clustering
for i in range(0,len(tabs)):
    df_clu = tabs[i]
    df_var = varies[i]
    df_clu = df_clu.replace(-np.inf, np.nan)
    df_clu = df_clu.replace(np.inf, np.nan)
    df_clu = df_clu[['H_DFA', 'mfdfa_slope', 'SampEn', 'Determinism',
           'Entropy', 'Laminarity', 'TrappingTime',
           'norm_cmor0.5-1.0', 'norm_cmor1.0-1.5', 'norm_cmor1.5-2.0']].dropna()
    n_clusters=3
    cmeans = FCM(n_clusters=n_clusters, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clu)
    cmeans.fit(X)
    centroids = cmeans.centers
    membership_mat = cmeans.u
    df['cmeans_p1'] = membership_mat[:,0]
    df['cmeans_p2'] = membership_mat[:,1]
    df['cmeans_p3'] = membership_mat[:,2]
    
    df.to_csv(output_path+'/clusters_'+df_var+'.csv')
    
    
# Correlation
lAc = lowA.drop('Divergence',axis=1).corr('spearman')
lQc = lowQ.drop('Divergence',axis=1).corr('spearman')
lMc = lowM.drop('Divergence',axis=1).corr('spearman')
lWc = lowW.drop('Divergence',axis=1).corr('spearman')
mAc = meanA.drop('Divergence',axis=1).corr('spearman')
mQc = meanQ.drop('Divergence',axis=1).corr('spearman')
mMc = meanM.drop('Divergence',axis=1).corr('spearman')
mWc = meanW.drop('Divergence',axis=1).corr('spearman')
mDc = meanD.drop('Divergence',axis=1).corr('spearman')
hAc = highA.drop('Divergence',axis=1).corr('spearman')
hQc = highQ.drop('Divergence',axis=1).corr('spearman')
hMc = highM.drop('Divergence',axis=1).corr('spearman')
hWc = highW.drop('Divergence',axis=1).corr('spearman')

lAc.to_csv(output_path+'/lAc.csv')
lQc.to_csv(output_path+'/lQc.csv')
lMc.to_csv(output_path+'/lMc.csv')
lWc.to_csv(output_path+'/lWc.csv')
mAc.to_csv(output_path+'/mAc.csv')
mQc.to_csv(output_path+'/mQc.csv')
mMc.to_csv(output_path+'/mMc.csv')
mWc.to_csv(output_path+'/mWc.csv')
mDc.to_csv(output_path+'/mDc.csv')
hAc.to_csv(output_path+'/hAc.csv')
hQc.to_csv(output_path+'/hQc.csv')
hMc.to_csv(output_path+'/hMc.csv')
hWc.to_csv(output_path+'/hWc.csv')