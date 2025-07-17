import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot as upr
import awkward as ak
from matplotlib.colors import LogNorm
import shutil
import os
import sys
import mplhep as hep

# Add module paths
sys.path.append(os.path.join(os.getcwd(), "Modules"))
import importlib
import system_and_data as sd
import plotting_functions as pf

# Reload modules (if modified)
importlib.reload(sd)
importlib.reload(pf)

def pass_veto(data):
    pt = data['theL1Obj.pt']
    stub_count_norm = data['theL1Obj.commonStubCount_norm']
    
    if pt <= 2:
        return False  
    elif 2 < pt <= 4:
        return stub_count_norm < 0.06
    elif 4 < pt <= 6:
        return stub_count_norm < 0.27
    elif 6 < pt <= 8:
        return stub_count_norm < 0.6
    elif 8 < pt <= 10:
        return stub_count_norm < 0.74
    else:
        return stub_count_norm < 0.95


# Paths to data and output figures
DATA_PATH = '/scratch/rkomuda/MagisteriumCMS14_2_0_pre2/Analysis/'
FIG_PATH = '/scratch/rkomuda/MagisteriumCMS14_2_0_pre2/Analysis/fig_png_veto/'


# Tree name and branch names in ROOT files
TREE_NAME = "tOmtf;3"
BRANCH_L1 = 'l1ObjColl/theL1Obj/theL1Obj.*'
BRANCH_GEN = 'genColl/theColl/theColl._*'

# Input file names
FILENAME_SINGLEMU_DISP = 'new_SingleMu_displaced_correction.root'
FILENAME_DISP_DISP = 'new_new_displaced_displaced.root'

# Global settings for efficiency plots
PT_CUTS = [0, 5, 12, 20]

# Load generated muon data
data_gen_singlemu = sd.load_data(FILENAME_SINGLEMU_DISP, DATA_PATH, TREE_NAME, BRANCH_GEN)
data_gen_disp = sd.load_data(FILENAME_DISP_DISP, DATA_PATH, TREE_NAME, BRANCH_GEN)

data_singlemu= sd.load_data(FILENAME_SINGLEMU_DISP, DATA_PATH, TREE_NAME, BRANCH_L1)
data_displaced = sd.load_data(FILENAME_DISP_DISP, DATA_PATH, TREE_NAME, BRANCH_L1)

# Select only SA muons (type 16) for displaced dataset
data_displaced_SA = data_displaced[data_displaced['theL1Obj.type'] == 16]
data_displaced_SA = sd.match_gen_muons(data_displaced_SA, data_gen_disp)
# Select only SA muons (type 16) for prompt dataset
data_singlemu_SA = data_singlemu[data_singlemu['theL1Obj.type'] == 16]
data_singlemu_SA = sd.match_gen_muons(data_singlemu_SA, data_gen_singlemu)
# Select Tracker Muons (type 15)
data_TK = data_displaced[data_displaced['theL1Obj.type'] == 15]
data_TK = sd.match_gen_muons(data_TK, data_gen_disp)
# Refresh figure directories
sd.refresh_fig_dir(FIG_PATH, refresh=True)

pf.plot_mean_comparison(
    [data_singlemu_SA,data_displaced_SA], ['SingleMu sample','Displaced sample'], 'theColl._pt', 'theL1Obj.commonStubCount',
    bins=np.arange(0, 100, 2), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Common stub count', title=r'SAMuon:displaced',
    fig_path=FIG_PATH, save=True,log=True
)

data_singlemu_SA['theL1Obj.commonStubCount_norm'] = data_singlemu_SA['theL1Obj.commonStubCount'] / data_singlemu_SA['theL1Obj.totalStubCount']
data_displaced_SA['theL1Obj.commonStubCount_norm'] = data_displaced_SA['theL1Obj.commonStubCount'] / data_displaced_SA['theL1Obj.totalStubCount']

pf.plot_mean_comparison(
    [data_singlemu_SA,data_displaced_SA], ['SingleMu sample','Displaced sample'], 'theColl._pt', 'theL1Obj.commonStubCount_norm',
    bins=np.arange(0, 100, 2), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Common stub count / total stub count', title=r'SAMuon:displaced',
    fig_path=FIG_PATH, save=True,log=True,density=True

)

pf.plot_efficiency_ptCuts_single_dataset(
    data_displaced_SA, data_gen_disp, 'SAMuon:displaced', 'theColl._pt',
    bins=np.arange(1, 100, 1), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Efficiency', title=r'Displaced sample BV',
    fig_path=FIG_PATH, save=True, ptCuts=PT_CUTS
)
pf.plot_efficiency_ptCuts_single_dataset(
    data_singlemu_SA, data_gen_singlemu, 'SAMuon:displaced', 'theColl._pt',
    bins=np.arange(1, 100, 1), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Efficiency', title=r'SingleMu sample BV',
    fig_path=FIG_PATH, save=True, ptCuts=PT_CUTS
)



data_singlemu_veto = data_singlemu_SA[data_singlemu_SA.apply(pass_veto, axis=1)]
data_displaced_veto = data_displaced_SA[data_displaced_SA.apply(pass_veto, axis=1)]


pf.plot_efficiency_ptCuts_single_dataset(
    data_singlemu_veto, data_gen_singlemu, 'SAMuon:displaced', 'theColl._pt',
    bins=np.arange(1, 100, 1), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Efficiency', title=r'SingleMu sample',
    fig_path=FIG_PATH, save=True, ptCuts=PT_CUTS
)
pf.plot_efficiency_ptCuts_single_dataset(
    data_displaced_veto, data_gen_disp, 'SAMuon:displaced', 'theColl._pt',
    bins=np.arange(1, 100, 1), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Efficiency', title=r'Displaced sample',
    fig_path=FIG_PATH, save=True, ptCuts=PT_CUTS
)

pf.plot_efficiency_comparison(
    [data_singlemu_SA, data_singlemu_veto],[data_gen_singlemu,data_gen_singlemu], ['SingleMu sample', 'SingleMu sample AV'], 'theColl._pt',
    bins=np.arange(1, 100, 1), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Efficiency', title=r'SingleMu sample',
    fig_path=FIG_PATH, save=True, ptCut=0
)
pf.plot_efficiency_comparison(
    [data_displaced_SA, data_displaced_veto],[data_gen_disp,data_gen_disp], ['Displaced sample', 'Displaced sample AV'], 'theColl._pt',
    bins=np.arange(1, 100, 1), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Efficiency', title=r'Displaced sample',
    fig_path=FIG_PATH, save=True, ptCut=0
)



data_singlemu_SA_pT_10 = data_singlemu_SA[data_singlemu_SA['theColl._pt'] > 10]
data_displaced_SA_pT_10 = data_displaced_SA[data_displaced_SA['theColl._pt'] > 10]
# [2,4] pT bin
data_singlemu_SA_pT_2_4 = data_singlemu_SA[(data_singlemu_SA['theColl._pt'] > 2) & (data_singlemu_SA['theColl._pt'] <= 4)]
data_displaced_SA_pT_2_4 = data_displaced_SA[(data_displaced_SA['theColl._pt'] > 2) & (data_displaced_SA['theColl._pt'] <= 4)]
# [8,10] pT bin
data_singlemu_SA_pT_8_10 = data_singlemu_SA[(data_singlemu_SA['theColl._pt'] > 8) & (data_singlemu_SA['theColl._pt'] <= 10)]
data_displaced_SA_pT_8_10 = data_displaced_SA[(data_displaced_SA['theColl._pt'] > 8) & (data_displaced_SA['theColl._pt'] <= 10)]
# [4,6] pT bin
data_singlemu_SA_pT_4_6 = data_singlemu_SA[(data_singlemu_SA['theColl._pt'] > 4) & (data_singlemu_SA['theColl._pt'] <= 6)]
data_displaced_SA_pT_4_6 = data_displaced_SA[(data_displaced_SA['theColl._pt'] > 4) & (data_displaced_SA['theColl._pt'] <= 6)]
# [6,8] pT bin
data_singlemu_SA_pT_6_8 = data_singlemu_SA[(data_singlemu_SA['theColl._pt'] > 6) & (data_singlemu_SA['theColl._pt'] <= 8)]  
data_displaced_SA_pT_6_8 = data_displaced_SA[(data_displaced_SA['theColl._pt'] > 6) & (data_displaced_SA['theColl._pt'] <= 8)]


pf.histogram_1D_comparison(
    [data_singlemu_SA_pT_10, data_displaced_SA_pT_10], ['SingleMu sample', 'Displaced sample'], 'theL1Obj.commonStubCount_norm',
    bins=np.arange(0, 1.2, 0.1), xlabel=r'Normalized common stub count',
    ylabel='Counts', title=r'SAMuon:displaced $p_T$ > 10 GeV',
    fig_path=FIG_PATH, save=True
)

pf.histogram_1D_comparison(
    [data_singlemu_SA_pT_2_4, data_displaced_SA_pT_2_4], ['SingleMu sample', 'Displaced sample'], 'theL1Obj.commonStubCount_norm',
    bins=np.arange(0, 1.1, 0.1), xlabel=r'Normalized common stub count',
    ylabel='Counts', title=r'SAMuon:displaced $p_T$ [2,4] GeV',
    fig_path=FIG_PATH, save=True
)

pf.histogram_1D_comparison(
    [data_singlemu_SA_pT_4_6, data_displaced_SA_pT_4_6], ['SingleMu sample', 'Displaced sample'], 'theL1Obj.commonStubCount_norm',
    bins=np.arange(0, 1.1, 0.1), xlabel=r'Normalized common stub count',
    ylabel='Counts', title=r'SAMuon:displaced $p_T$ [4,6] GeV',
    fig_path=FIG_PATH, save=True
)

pf.histogram_1D_comparison(
    [data_singlemu_SA_pT_6_8, data_displaced_SA_pT_6_8], ['SingleMu sample', 'Displaced sample'], 'theL1Obj.commonStubCount_norm',
    bins=np.arange(0, 1.1, 0.1), xlabel=r'Normalized common stub count',
    ylabel='Counts', title=r'SAMuon:displaced $p_T$ [6,8] GeV',
    fig_path=FIG_PATH, save=True
)

pf.histogram_1D_comparison(
    [data_singlemu_SA_pT_8_10, data_displaced_SA_pT_8_10], ['SingleMu sample', 'Displaced sample'], 'theL1Obj.commonStubCount_norm',
    bins=np.arange(0, 1.1, 0.1), xlabel=r'Normalized common stub count',
    ylabel='Counts', title=r'SAMuon:displaced $p_T$ [8,10] GeV',
    fig_path=FIG_PATH, save=True
)


