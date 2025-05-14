# -*- coding: utf-8 -*-
"""
Created February 2025

@author: Krzysztof Raczynski
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# EDIT BLOCK
path = "path_to_mergedData"
output_path = "output_path"


# PLOTTING
# read data
lowA = pd.read_csv(path+'/lowA.csv')
lowQ = pd.read_csv(path+'/lowQ.csv')
lowM = pd.read_csv(path+'/lowM.csv')
lowW = pd.read_csv(path+'/lowW.csv')
meanA = pd.read_csv(path+'/meanA.csv')
meanQ = pd.read_csv(path+'/meanQ.csv')
meanM = pd.read_csv(path+'/meanM.csv')
meanW = pd.read_csv(path+'/meanW.csv')
meanD = pd.read_csv(path+'/meanD.csv')
highA = pd.read_csv(path+'/highA.csv')
highQ = pd.read_csv(path+'/highQ.csv')
highM = pd.read_csv(path+'/highM.csv')
highW = pd.read_csv(path+'/highW.csv')
WTMM_l_W = pd.read_csv(path+'/wtmm_lw.csv')
WTMM_l_M = pd.read_csv(path+'/wtmm_lm.csv')
WTMM_l_Q = pd.read_csv(path+'/wtmm_lq.csv')
WTMM_l_A = pd.read_csv(path+'/wtmm_la.csv')
WTMM_m_D = pd.read_csv(path+'/wtmm_md.csv')
WTMM_m_W = pd.read_csv(path+'/wtmm_mw.csv')
WTMM_m_M = pd.read_csv(path+'/wtmm_mm.csv')
WTMM_m_Q = pd.read_csv(path+'/wtmm_mq.csv')
WTMM_m_A = pd.read_csv(path+'/wtmm_ma.csv')
WTMM_u_W = pd.read_csv(path+'/wtmm_uw.csv')
WTMM_u_M = pd.read_csv(path+'/wtmm_um.csv')
WTMM_u_Q = pd.read_csv(path+'/wtmm_uq.csv')
WTMM_u_A = pd.read_csv(path+'/wtmm_ua.csv')
lyap_lw = pd.read_csv(path+'/lyap_lw.csv')
lyap_lm = pd.read_csv(path+'/lyap_lm.csv')
lyap_lq = pd.read_csv(path+'/lyap_lq.csv')
lyap_la = pd.read_csv(path+'/lyap_la.csv')
lyap_md = pd.read_csv(path+'/lyap_md.csv')
lyap_mw = pd.read_csv(path+'/lyap_mw.csv')
lyap_mm = pd.read_csv(path+'/lyap_mm.csv')
lyap_mq = pd.read_csv(path+'/lyap_mq.csv')
lyap_ma = pd.read_csv(path+'/lyap_ma.csv')
lyap_uw = pd.read_csv(path+'/lyap_uw.csv')
lyap_um = pd.read_csv(path+'/lyap_um.csv')
lyap_uq = pd.read_csv(path+'/lyap_uq.csv')
lyap_ua = pd.read_csv(path+'/lyap_ua.csv')
lAc = pd.read_csv(path+'/lAc.csv')
lQc = pd.read_csv(path+'/lQc.csv')
lMc = pd.read_csv(path+'/lMc.csv')
lWc = pd.read_csv(path+'/lWc.csv')
mAc = pd.read_csv(path+'/mAc.csv')
mQc = pd.read_csv(path+'/mQc.csv')
mMc = pd.read_csv(path+'/mMc.csv')
mWc = pd.read_csv(path+'/mWc.csv')
mDc = pd.read_csv(path+'/mDc.csv')
hAc = pd.read_csv(path+'/hAc.csv')
hQc = pd.read_csv(path+'/hQc.csv')
hMc = pd.read_csv(path+'/hMc.csv')
hWc = pd.read_csv(path+'/hWc.csv')


# fix index
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
WTMM_l_W.set_index('Unnamed: 0',drop=True,inplace=True)
WTMM_l_M.set_index('Unnamed: 0',drop=True,inplace=True)
WTMM_l_Q.set_index('Unnamed: 0',drop=True,inplace=True)
WTMM_l_A.set_index('Unnamed: 0',drop=True,inplace=True)
WTMM_m_D.set_index('Unnamed: 0',drop=True,inplace=True)
WTMM_m_W.set_index('Unnamed: 0',drop=True,inplace=True)
WTMM_m_M.set_index('Unnamed: 0',drop=True,inplace=True)
WTMM_m_Q.set_index('Unnamed: 0',drop=True,inplace=True)
WTMM_m_A.set_index('Unnamed: 0',drop=True,inplace=True)
WTMM_u_W.set_index('Unnamed: 0',drop=True,inplace=True)
WTMM_u_M.set_index('Unnamed: 0',drop=True,inplace=True)
WTMM_u_Q.set_index('Unnamed: 0',drop=True,inplace=True)
WTMM_u_A.set_index('Unnamed: 0',drop=True,inplace=True)
lyap_lw.set_index('Unnamed: 0',drop=True, inplace=True)
lyap_lm.set_index('Unnamed: 0',drop=True, inplace=True)
lyap_lq.set_index('Unnamed: 0',drop=True, inplace=True)
lyap_la.set_index('Unnamed: 0',drop=True, inplace=True)
lyap_md.set_index('Unnamed: 0',drop=True, inplace=True)
lyap_mw.set_index('Unnamed: 0',drop=True, inplace=True)
lyap_mm.set_index('Unnamed: 0',drop=True, inplace=True)
lyap_mq.set_index('Unnamed: 0',drop=True, inplace=True)
lyap_ma.set_index('Unnamed: 0',drop=True, inplace=True)
lyap_uw.set_index('Unnamed: 0',drop=True, inplace=True)
lyap_um.set_index('Unnamed: 0',drop=True, inplace=True)
lyap_uq.set_index('Unnamed: 0',drop=True, inplace=True)
lyap_ua.set_index('Unnamed: 0',drop=True, inplace=True)
lAc.set_index('Unnamed: 0', inplace=True,drop=True)
lQc.set_index('Unnamed: 0', inplace=True,drop=True)
lMc.set_index('Unnamed: 0', inplace=True,drop=True)
lWc.set_index('Unnamed: 0', inplace=True,drop=True)
mAc.set_index('Unnamed: 0', inplace=True,drop=True)
mQc.set_index('Unnamed: 0', inplace=True,drop=True)
mMc.set_index('Unnamed: 0', inplace=True,drop=True)
mWc.set_index('Unnamed: 0', inplace=True,drop=True)
mDc.set_index('Unnamed: 0', inplace=True,drop=True)
hAc.set_index('Unnamed: 0', inplace=True,drop=True)
hQc.set_index('Unnamed: 0', inplace=True,drop=True)
hMc.set_index('Unnamed: 0', inplace=True,drop=True)
hWc.set_index('Unnamed: 0', inplace=True,drop=True)


# helper functions
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.tick_params(axis='x', labelsize=10,rotation=90)
    for tick in ax.get_xticklabels():
        tick.set_fontname("times new roman")


# formatting
font = {'family':'times new roman',
        'color':'black',
        'weight':'normal',
        'size':10}

boxcol = 'slategray'
whiskcol = 'black'
boxline = 0.2
boxwid = 0.8
medline = 0.2
whiskline = 0.2

meanpointprops = dict(marker='o', markeredgecolor='black', markerfacecolor='red', markersize=1)
medianprops = dict(linestyle='-', linewidth=medline, color=whiskcol)
flierprops = dict(marker='o', markerfacecolor='gray', markersize=0.1,markeredgecolor='gray')
boxprops=dict(color=whiskcol,facecolor=boxcol,linewidth=boxline)
whiskerprops=dict(color=whiskcol,linewidth=whiskline)
labels = [' ']
title_font = {'family':'times new roman','color':'black','weight':'normal','size':8}

# FIGURES
# Figure 2
fig, [[axa1,axa2,axa3,axa4],[axb1,axb2,axb3,axb4],[axc1,axc2,axc3,axc4],[axd1,axd2,axd3,axd4],[axe1,axe2,axe3,axe4],[axf1,axf2,axf3,axf4]] = plt.subplots(6,4)
fig.set_size_inches(6.5, 6)

def plot_violin(ax, data, pos):
    violin_parts = ax.violinplot(data,positions=pos,widths = 0.8,showmeans=False, showmedians=False,showextrema=False)#, widths = boxwid, showfliers=False, patch_artist=True, showcaps=False,
    for pc in violin_parts['bodies']:
        pc.set_facecolor('slategray')
        pc.set_edgecolor('black')
    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)
    whiskers_min, whiskers_max = quartile1, quartile3
    ax.scatter(pos, medians, marker='o', color='white', s=5, zorder=3)
    ax.vlines(pos, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(pos, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
    
boxcol = 'gray'
whiskcol = 'black'
axes = [axa1,axa2,axa3,axa4,axb1,axb2,axb3,axb4,axc1,axc2,axc3,axc4,axd1,axd2,axd3,axd4,axe1,axe2,axe3,axe4,axf1,axf2,axf3,axf4]
axes_a = [axa1,axa2,axa3,axa4]
axes_b = [axb1,axb2,axb3,axb4]
axes_c = [axc1,axc2,axc3,axc4]
axes_d = [axd1,axd2,axd3,axd4]
axes_e = [axe1,axe2,axe3,axe4]
axes_f = [axf1,axf2,axf3,axf4]
boxline = 0.01
boxwid = .5
medline = 0.8
whiskline = 0.3

plot_violin(axa1, np.array(highW['SampEn'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axa1, np.array(meanW['SampEn'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axa1, np.array(lowW['SampEn'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axa2, np.array(highM['SampEn'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axa2, np.array(meanM['SampEn'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axa2, np.array(lowM['SampEn'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axa3, np.array(highQ['SampEn'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axa3, np.array(meanQ['SampEn'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axa3, np.array(lowQ['SampEn'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axa4, np.array(highA['SampEn'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axa4, np.array(meanA['SampEn'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axa4, np.array(lowA['SampEn'].replace(np.inf, np.nan).dropna()), [3])

axa1.set_ylabel(r'$\it{SampEn}$', fontdict=font)
for ax in axes_a:
    ax.set_ylim(0,4)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plot_violin(axb1, np.array(highW['H_rs'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axb1, np.array(meanW['H_rs'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axb1, np.array(lowW['H_rs'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axb2, np.array(highM['H_rs'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axb2, np.array(meanM['H_rs'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axb2, np.array(lowM['H_rs'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axb3, np.array(highQ['H_rs'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axb3, np.array(meanQ['H_rs'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axb3, np.array(lowQ['H_rs'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axb4, np.array(highA['H_rs'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axb4, np.array(meanA['H_rs'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axb4, np.array(lowA['H_rs'].replace(np.inf, np.nan).dropna()), [3])

axb1.set_ylabel(r'$\it{H_{rs}}$', fontdict=font)
for ax in axes_b:
    ax.set_ylim(0,1)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
N = np.log2(len(np.array(highW['H_rs'].replace(np.inf, np.nan).dropna())))
y1 = 0.5 - np.exp(-7.33*np.log(np.log(N))+4.21)
y2 = np.exp(-7.20*np.log(np.log(N))+4.04) + 0.5
axb1.fill_between([0,4], y2, y1,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)

N = np.log2(len(np.array(highM['H_rs'].replace(np.inf, np.nan).dropna())))
y1 = 0.5 - np.exp(-7.33*np.log(np.log(N))+4.21)
y2 = np.exp(-7.20*np.log(np.log(N))+4.04) + 0.5
axb2.fill_between([0,4], y2, y1,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)

N = np.log2(len(np.array(highQ['H_rs'].replace(np.inf, np.nan).dropna())))
y1 = 0.5 - np.exp(-7.33*np.log(np.log(N))+4.21)
y2 = np.exp(-7.20*np.log(np.log(N))+4.04) + 0.5
axb3.fill_between([0,4], y2, y1,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)

N = np.log2(len(np.array(highA['H_rs'].replace(np.inf, np.nan).dropna())))
y1 = 0.5 - np.exp(-7.33*np.log(np.log(N))+4.21)
y2 = np.exp(-7.20*np.log(np.log(N))+4.04) + 0.5
axb4.fill_between([0,4], y2, y1,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)

plot_violin(axc1, np.array(highW['H_DFA'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axc1, np.array(meanW['H_DFA'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axc1, np.array(lowW['H_DFA'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axc2, np.array(highM['H_DFA'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axc2, np.array(meanM['H_DFA'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axc2, np.array(lowM['H_DFA'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axc3, np.array(highQ['H_DFA'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axc3, np.array(meanQ['H_DFA'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axc3, np.array(lowQ['H_DFA'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axc4, np.array(highA['H_DFA'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axc4, np.array(meanA['H_DFA'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axc4, np.array(lowA['H_DFA'].replace(np.inf, np.nan).dropna()), [3])

axc1.set_ylabel(r'$\it{H_{DFA}}$', fontdict=font)
for ax in axes_c:
    ax.set_ylim(0,2)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
N = np.log2(len(np.array(highW['H_DFA'].replace(np.inf, np.nan).dropna())))
y1 = np.exp(-3.10*np.log(N)+4.77)
y2 = np.exp(-2.93*np.log(N)+4.45)
axc1.fill_between([0,4], 0.5-y2, y1+0.5,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)
axc1.fill_between([0,4], 1-y2, y1+1,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)
axc1.fill_between([0,4], 1.5-y2, y1+1.5,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)

N = np.log2(len(np.array(highM['H_DFA'].replace(np.inf, np.nan).dropna())))
y1 = np.exp(-3.10*np.log(N)+4.77)
y2 = np.exp(-2.93*np.log(N)+4.45)
axc2.fill_between([0,4], 0.5-y2, y1+0.5,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)
axc2.fill_between([0,4], 1-y2, y1+1,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)
axc2.fill_between([0,4], 1.5-y2, y1+1.5,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)

N = np.log2(len(np.array(highQ['H_DFA'].replace(np.inf, np.nan).dropna())))
y1 = np.exp(-3.10*np.log(N)+4.77)
y2 = np.exp(-2.93*np.log(N)+4.45)
axc3.fill_between([0,4], 0.5-y2, y1+0.5,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)
axc3.fill_between([0,4], 1-y2, y1+1,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)
axc3.fill_between([0,4], 1.5-y2, y1+1.5,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)

N = np.log2(len(np.array(highA['H_DFA'].replace(np.inf, np.nan).dropna())))
y1 = np.exp(-3.10*np.log(N)+4.77)
y2 = np.exp(-2.93*np.log(N)+4.45)
axc4.fill_between([0,4], 0.5-y2, y1+0.5,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)
axc4.fill_between([0,4], 1-y2, y1+1,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)
axc4.fill_between([0,4], 1.5-y2, y1+1.5,color='lightcoral',linewidth=0.0,alpha=0.25,zorder=0)

plot_violin(axd1, np.array(highW['Determinism'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axd1, np.array(meanW['Determinism'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axd1, np.array(lowW['Determinism'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axd2, np.array(highM['Determinism'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axd2, np.array(meanM['Determinism'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axd2, np.array(lowM['Determinism'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axd3, np.array(highQ['Determinism'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axd3, np.array(meanQ['Determinism'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axd3, np.array(lowQ['Determinism'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axd4, np.array(highA['Determinism'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axd4, np.array(meanA['Determinism'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axd4, np.array(lowA['Determinism'].replace(np.inf, np.nan).dropna()), [3])
axd1.set_ylabel(r'$\it{Det}$', fontdict=font)
axd1.set_ylim(0,1)
axd2.set_ylim(0,1)
axd3.set_ylim(0,1)
axd4.set_ylim(0,1)

plot_violin(axe1, np.array(highW['Laminarity'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axe1, np.array(meanW['Laminarity'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axe1, np.array(lowW['Laminarity'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axe2, np.array(highM['Laminarity'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axe2, np.array(meanM['Laminarity'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axe2, np.array(lowM['Laminarity'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axe3, np.array(highQ['Laminarity'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axe3, np.array(meanQ['Laminarity'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axe3, np.array(lowQ['Laminarity'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axe4, np.array(highA['Laminarity'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axe4, np.array(meanA['Laminarity'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axe4, np.array(lowA['Laminarity'].replace(np.inf, np.nan).dropna()), [3])
axe1.set_ylabel(r'$\it{Lam}$', fontdict=font)
axe1.set_ylim(0,1)
axe2.set_ylim(0,1)
axe3.set_ylim(0,1)
axe4.set_ylim(0,1)

plot_violin(axf1, np.array(highW['TrappingTime'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axf1, np.array(meanW['TrappingTime'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axf1, np.array(lowW['TrappingTime'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axf2, np.array(highM['TrappingTime'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axf2, np.array(meanM['TrappingTime'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axf2, np.array(lowM['TrappingTime'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axf3, np.array(highQ['TrappingTime'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axf3, np.array(meanQ['TrappingTime'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axf3, np.array(lowQ['TrappingTime'].replace(np.inf, np.nan).dropna()), [3])

plot_violin(axf4, np.array(highA['TrappingTime'].replace(np.inf, np.nan).dropna()), [1])
plot_violin(axf4, np.array(meanA['TrappingTime'].replace(np.inf, np.nan).dropna()), [2])
plot_violin(axf4, np.array(lowA['TrappingTime'].replace(np.inf, np.nan).dropna()), [3])
axf1.set_ylabel(r'$\it{TT}$', fontdict=font)
axf1.set_ylim(0,10)
axf2.set_ylim(0,10)
axf3.set_ylim(0,10)
axf4.set_ylim(0,10)

axa1.set_title('weekly',fontdict=title_font)
axa2.set_title('monthly',fontdict=title_font)
axa3.set_title('quarterly',fontdict=title_font)
axa4.set_title('annually',fontdict=title_font)

for ax in axes:
    ax.set_xticks([1,2,3])
    ax.set_xticklabels([])
    ax.set_xlim(0.5,3.5)
    ax.tick_params(axis='y', labelsize=10)
    for tick in ax.get_yticklabels():
        tick.set_fontname("times new roman")
    ax.yaxis.grid(which='major', color='#898989', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.yaxis.grid(which='minor', color='#898989', linestyle='--', linewidth=0.25)

for ax in axes_f:
    ax.set_xticks([1,2,3])
    ax.set_xticklabels([r'$\it{Q_{max}}$',r'$\it{Q_{avg}}$',r'$\it{Q_{min}}$'])
    ax.tick_params(axis='x', labelsize=10)
    for tick in ax.get_xticklabels():
        tick.set_fontname("times new roman")

plt.tight_layout()
plt.savefig(output_path + '/fig2.png',dpi=300)


# Figure 3
fig, [[axa1,axa2,axa3],[axb1,axb2,axb3],[axc1,axc2,axc3]] = plt.subplots(3,3)
fig.set_size_inches(6.5, 3)

boxcol = 'gray'
whiskcol = 'black'
axes = [axa1,axa2,axa3,axb1,axb2,axb3,axc1,axc2,axc3]
boxline = 0.01
boxwid = .8
medline = 0.8
whiskline = 0.3

def plot_boxes(ax, data):
    ax.boxplot(data,widths = boxwid, showfliers=True, patch_artist=True, showcaps=False,
                boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,
                flierprops=flierprops)
    
plot_boxes(axa1, lyap_lq.dropna())
plot_boxes(axb1, lyap_lm.dropna())
plot_boxes(axc1, lyap_lw.dropna())

plot_boxes(axa2, lyap_mq.dropna())
plot_boxes(axb2, lyap_mm.dropna())
plot_boxes(axc2, lyap_mw.dropna())

plot_boxes(axa3, lyap_uq.dropna())
plot_boxes(axb3, lyap_um.dropna())
plot_boxes(axc3, lyap_uw.dropna())

axa1.set_title(r'$\it{Q_{min}}$',fontdict=title_font)
axa2.set_title(r'$\it{Q_{avg}}$',fontdict=title_font)
axa3.set_title(r'$\it{Q_{max}}$',fontdict=title_font)

for ax in axes:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xticks([1,2,3,4,5,6,7,8,9,10])
    ax.set_xticklabels(['','','','','','','','','',''])
    ax.set_xlim(0.5,10.5)
    ax.set_ylim(-1,1)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10,rotation=90)
    for tick in ax.get_xticklabels():
        tick.set_fontname("times new roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("times new roman")
    ax.yaxis.grid(which='major', color='#898989', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.yaxis.grid(which='minor', color='#898989', linestyle='--', linewidth=0.25)
    
for ax in [axa1,axb1,axc1]:
    ax.set_ylabel(r'$\it{λ}$', fontdict=font)
    
for ax in [axc1,axc2,axc3]:
    ax.set_xticklabels([1,2,3,4,5,6,7,8,9,10])
   
axa1.text(-4.5,0,r'$\it{quarterly}$',fontdict=title_font,rotation=90,ha='center',va='center')
axb1.text(-4.5,0,r'$\it{monthly}}$',fontdict=title_font,rotation=90,ha='center',va='center')
axc1.text(-4.5,0,r'$\it{weekly}$',fontdict=title_font,rotation=90,ha='center',va='center')

plt.tight_layout()
plt.savefig(output_path + '/fig3.png',dpi=300)


# Figure 4
fig, [[axa1,axa2,axa3],[axb1,axb2,axb3],[axc1,axc2,axc3],[axd1,axd2,axd3],[axe1,axe2,axe3]] = plt.subplots(5,3)
fig.set_size_inches(6.5, 4)

axes = [axa1,axa2,axa3,axb1,axb2,axb3,axc1,axc2,axc3,axd1,axd2,axd3,axe1,axe2,axe3]

def plot_boxes(ax, data, pos):
    ax.boxplot(data,positions=pos, widths = 0.8, showfliers=True, patch_artist=True, showcaps=False,
                boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,
                flierprops=flierprops)

plot_boxes(axa3, WTMM_l_A, [1,2,3,4,5,6,7,8,9])
plot_boxes(axb3, WTMM_l_Q, [1,2,3,4,5,6,7,8,9])
plot_boxes(axc3, WTMM_l_M, [1,2,3,4,5,6,7,8,9])
plot_boxes(axd3, WTMM_l_W, [1,2,3,4,5,6,7,8,9])
plot_boxes(axa2, WTMM_m_A, [1,2,3,4,5,6,7,8,9])
plot_boxes(axb2, WTMM_m_Q, [1,2,3,4,5,6,7,8,9])
plot_boxes(axc2, WTMM_m_M, [1,2,3,4,5,6,7,8,9])
plot_boxes(axd2, WTMM_m_W, [1,2,3,4,5,6,7,8,9])
plot_boxes(axe2, WTMM_m_D, [1,2,3,4,5,6,7,8,9])
plot_boxes(axa1, WTMM_u_A, [1,2,3,4,5,6,7,8,9])
plot_boxes(axb1, WTMM_u_Q, [1,2,3,4,5,6,7,8,9])
plot_boxes(axc1, WTMM_u_M, [1,2,3,4,5,6,7,8,9])
plot_boxes(axd1, WTMM_u_W, [1,2,3,4,5,6,7,8,9])

for ax in [axa1,axa2,axa3]:
    ylim = 1.75
    ax.set_ylim(-0.5, ylim)
    
for ax in [axb1,axb2,axb3,axc1,axc2,axc3]:
    ylim = 1.5
    ax.set_ylim(-0.5, ylim)
    
for ax in [axd1,axd2,axd3,axe1,axe2,axe3]:
    ylim = 1.25
    ax.set_ylim(-0.5, ylim)
    
for ax in [axa1,axb1,axc1, axd1, axe2]:
    ax.set_ylabel(r'$\it{s}$', fontdict=font) 

for ax in axes:
    ax.plot([3.5,3.5],[-1,2],lw=0.2,c='black')
    ax.plot([6.5,6.5],[-1,2],lw=0.2,c='black')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xticks([1,2,3,4,5,6,7,8,9])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['','','','','','','','',''],rotation=90)
    ax.set_xlim(0,10)
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='x', labelsize=6)
    for tick in ax.get_xticklabels():
        tick.set_fontname("times new roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("times new roman")
    ax.yaxis.grid(which='major', color='#898989', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.yaxis.grid(which='minor', color='#b0b0b0', linestyle='--', linewidth=0.25)

for ax in [axe1,axe2,axe3]:
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax.set_xticks([1,2,3,4,5,6,7,8,9])
    ax.set_xticklabels([r'$\it{{0.5, 1.0}}$',r'$\it{{1.0, 1.5}}$',r'$\it{{1.5, 2.0}}$',
                        r'$\it{{0.5, 1.0}}$',r'$\it{{1.0, 1.5}}$',r'$\it{{1.5, 2.0}}$',
                        r'$\it{{0.5, 1.0}}$',r'$\it{{1.0, 1.5}}$',r'$\it{{1.5, 2.0}}$'],rotation=90)
    ax.text(2,-2.8, 'max', horizontalalignment='center', fontdict=font)
    ax.text(5,-2.8, 'mean', horizontalalignment='center', fontdict=font)
    ax.text(8,-2.8, 'norm', horizontalalignment='center', fontdict=font)
    
axa1.text(-4,0.625, 'annually', verticalalignment='center',rotation=90, fontdict=font)
axb1.text(-4,0.5, 'quarterly', verticalalignment='center',rotation=90, fontdict=font)
axc1.text(-4,0.5, 'monthly', verticalalignment='center',rotation=90, fontdict=font)
axd1.text(-4,0.375, 'weekly', verticalalignment='center',rotation=90, fontdict=font)
axe1.text(-4,0.375, 'daily', verticalalignment='center',rotation=90, fontdict=font)

axa1.text(5,2, r'$\it{Q_{max}}$', horizontalalignment='center', fontdict=font)
axa2.text(5,2, r'$\it{Q_{avg}}$', horizontalalignment='center', fontdict=font)
axa3.text(5,2, r'$\it{Q_{min}}$', horizontalalignment='center', fontdict=font)

plt.tight_layout()
plt.savefig(output_path + '/fig4.png',dpi=300)


# Figure 5
fig, [[axa1,axa2,axa3,axa4],[axb1,axb2,axb3,axb4],[axc1,axc2,axc3,axc4]] = plt.subplots(3,4)
fig.set_size_inches(8, 6.5)

colormap = 'seismic'
def plot_heatmap(ax, data):
    mask = np.zeros_like(data)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(data, vmin=-1, vmax=1,mask=mask, cmap=colormap, cbar=False, ax=ax,linewidths=0)
    
axes = [axa1,axa2,axa3,axa4,axb1,axb2,axb3,axb4,axc1,axc2,axc3,axc4,axd1,axd2,axd3,axd4]

plot_heatmap(axa1, hWc.drop('Entropy',axis=1).drop('Entropy',axis=0))
plot_heatmap(axa2, hMc.drop('Entropy',axis=1).drop('Entropy',axis=0))
plot_heatmap(axa3, hQc.drop('Entropy',axis=1).drop('Entropy',axis=0))
plot_heatmap(axa4, hAc.drop('Entropy',axis=1).drop('Entropy',axis=0))

plot_heatmap(axb1, mWc.drop('Entropy',axis=1).drop('Entropy',axis=0))
plot_heatmap(axb2, mMc.drop('Entropy',axis=1).drop('Entropy',axis=0))
plot_heatmap(axb3, mQc.drop('Entropy',axis=1).drop('Entropy',axis=0))
plot_heatmap(axb4, mAc.drop('Entropy',axis=1).drop('Entropy',axis=0))

plot_heatmap(axc1, lWc.drop('Entropy',axis=1).drop('Entropy',axis=0))
plot_heatmap(axc2, lMc.drop('Entropy',axis=1).drop('Entropy',axis=0))
plot_heatmap(axc3, lQc.drop('Entropy',axis=1).drop('Entropy',axis=0))
plot_heatmap(axc4, lAc.drop('Entropy',axis=1).drop('Entropy',axis=0))

axa1.set_title('weekly',fontdict=title_font)
axa2.set_title('monthly',fontdict=title_font)
axa3.set_title('quarterly',fontdict=title_font)
axa4.set_title('annual',fontdict=title_font)

axa1.text(-8,11.5,r'$\it{Q_{max}}$',fontdict=title_font,rotation=90,ha='center',va='center')
axb1.text(-8,11.5,r'$\it{Q_{avg}}$',fontdict=title_font,rotation=90,ha='center',va='center')
axc1.text(-8,11.5,r'$\it{Q_{min}}$',fontdict=title_font,rotation=90,ha='center',va='center')

for ax in [axa1,axb1,axc1]:
    ax.text(-6,14.5,r'$\it_{max}$',fontdict=font,rotation=90,ha='center',va='center')
    ax.text(-6,17.5,r'$\it_{norm}$',fontdict=font,rotation=90,ha='center',va='center')
    ax.text(-6,20.5,r'$\it_{mean}$',fontdict=font,rotation=90,ha='center',va='center')
    
for ax in [axc1,axc2,axc3,axc4]:
    ax.text(14.5,27,r'$\it_{max}$',fontdict=font,ha='center',va='center')
    ax.text(17.5,27,r'$\it_{norm}$',fontdict=font,ha='center',va='center')
    ax.text(20.5,27,r'$\it_{mean}$',fontdict=font,ha='center',va='center')

for ax in axes:
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xticklabels([])

for ax in [axa1,axb1,axc1]:
    ax.set_yticks([i+0.5 for i in range(0,22)])
    ax.set_yticklabels([r'$\it{H_{DFA}}$', r'$\it{H_{RS}}$', r'$\it{genH_s}$', 
                        r'$\it{MFDFA_{s}}$', r'$\it{SampEn}$', r'$\it{λ_1}$',
                        r'$\it{λ_2}$', r'$\it{λ_3}$', r'$\it{λ_4}$', r'$\it{RR}$', 
                        r'$\it{Det}$',r'$\it{Lam}$', r'$\it{TT}$', 
                        r'$\it_{0.5-1.0}$',r'$\it_{1.0-1.5}$', r'$\it_{1.5-2.0}$', 
                        r'$\it_{0.5-1.0}$',r'$\it_{1.0-1.5}$', r'$\it_{1.5-2.0}$', 
                        r'$\it_{0.5-1.0}$',r'$\it_{1.0-1.5}$', r'$\it_{1.5-2.0}$'])
    ax.tick_params(axis='y', length=0, labelsize=6)
    for tick in ax.get_yticklabels():
        tick.set_fontname("times new roman")
    
for ax in [axc1,axc2,axc3,axc4]:
    ax.set_xticks([i+0.5 for i in range(0,22)])
    ax.set_xticklabels([r'$\it{H_{DFA}}$', r'$\it{H_{RS}}$', r'$\it{genH_s}$', 
                        r'$\it{MFDFA_{s}}$', r'$\it{SampEn}$', r'$\it{λ_1}$',
                        r'$\it{λ_2}$', r'$\it{λ_3}$', r'$\it{λ_4}$', r'$\it{RR}$', 
                        r'$\it{Det}$',r'$\it{Lam}$', r'$\it{TT}$', 
                        r'$\it_{0.5-1.0}$',r'$\it_{1.0-1.5}$', r'$\it_{1.5-2.0}$', 
                        r'$\it_{0.5-1.0}$',r'$\it_{1.0-1.5}$', r'$\it_{1.5-2.0}$', 
                        r'$\it_{0.5-1.0}$',r'$\it_{1.0-1.5}$', r'$\it_{1.5-2.0}$'])
    ax.tick_params(axis='x', length=0, labelsize=6,rotation=90)
    for tick in ax.get_xticklabels():
        tick.set_fontname("times new roman")

for ax in axes:
    bins_mag = [2,4,5,9,13,16,19]
    for mag in bins_mag:
        ax.axhline(mag, color='black', zorder=100, linewidth=0.3)
        ax.axvline(mag, color='black', zorder=100, linewidth=0.3)
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(0.3) 
    ax.set_ylabel('')

norm = Normalize(vmin=-1, vmax=1)
sm = ScalarMappable(norm=norm, cmap=colormap)
sm.set_array([])
cbar_ax = fig.add_axes([1, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax)  
cbar.ax.set_title(r'$\it{ρ}$', fontdict=title_font,pad=10,loc='center')

cbar.ax.tick_params(labelsize=6)
for label in cbar.ax.get_yticklabels():
    label.set_fontfamily("times new roman")

plt.tight_layout()
plt.savefig(output_path + '/fig5.png',dpi=300,bbox_inches='tight')