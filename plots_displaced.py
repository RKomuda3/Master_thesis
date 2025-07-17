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

# Paths to data and output figures
DATA_PATH = '/scratch/rkomuda/MagisteriumCMS14_2_0_pre2/Analysis/'
FIG_PATH_SA = '/scratch/rkomuda/MagisteriumCMS14_2_0_pre2/Analysis/fig_png_SA_disp/'
FIG_PATH_TK = '/scratch/rkomuda/MagisteriumCMS14_2_0_pre2/Analysis/fig_png_TK_disp/'

# Tree name and branch names in ROOT files
TREE_NAME = "tOmtf;3"
BRANCH_L1 = 'l1ObjColl/theL1Obj/theL1Obj.*'
BRANCH_GEN = 'genColl/theColl/theColl._*'

# Input file names
FILENAME_PROMPT_COLL = 'new_new_displaced_prompt.root'
FILENAME_DISP_COLL = 'new_new_displaced_displaced.root'

# Global settings for efficiency plots
PT_CUTS = [0, 5, 12, 20]

# Load generated muon data
data_gen_disp = sd.load_data(FILENAME_DISP_COLL, DATA_PATH, TREE_NAME, BRANCH_GEN)
data_gen_prompt = sd.load_data(FILENAME_PROMPT_COLL, DATA_PATH, TREE_NAME, BRANCH_GEN)

# Load L1 object data
data_prompt = sd.load_data(FILENAME_PROMPT_COLL, DATA_PATH, TREE_NAME, BRANCH_L1)
data_displaced = sd.load_data(FILENAME_DISP_COLL, DATA_PATH, TREE_NAME, BRANCH_L1)

# Select only SA muons (type 16) for displaced dataset
data_displaced_SA = data_displaced[data_displaced['theL1Obj.type'] == 16]
data_displaced_SA = sd.match_gen_muons(data_displaced_SA, data_gen_disp)

# Select only SA muons (type 16) for prompt dataset
data_prompt_SA = data_prompt[data_prompt['theL1Obj.type'] == 16]
data_prompt_SA = sd.match_gen_muons(data_prompt_SA, data_gen_prompt)

# Select Tracker Muons (type 15)
data_TK = data_displaced[data_displaced['theL1Obj.type'] == 15]
data_TK = sd.match_gen_muons(data_TK, data_gen_disp)


# Refresh figure directories
sd.refresh_fig_dir(FIG_PATH_SA, refresh=True)
sd.refresh_fig_dir(FIG_PATH_TK, refresh=True)

pf.histogram_1D_comparison(
    [data_displaced_SA], ['SAMuon:displaced'], 'theColl._abs_dxy',
    bins=np.arange(0, 200, 5), xlabel=r'$|d_{xy}| \ [cm]$',
    ylabel='Counts', title=r'Displaced sample',
    fig_path=FIG_PATH_SA, save=True
)
pf.histogram_1D_comparison(
    [data_prompt_SA], ['SAMuon:prompt'], 'theColl._abs_dxy',
    bins=np.arange(0, 200, 5), xlabel=r'$|d_{xy}| \ [cm]$',
    ylabel='Counts', title=r'Displaced sample',
    fig_path=FIG_PATH_SA, save=True
)

# Efficiency plot for tracker muons
pf.plot_efficiency_ptCuts_single_dataset(
    data_TK, data_gen_disp, 'TKMuon', 'theColl._pt',
    bins=np.arange(1, 100, 1), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Tracking efficiency', title=r'Displaced sample',
    fig_path=FIG_PATH_TK, save=True, ptCuts=PT_CUTS
)

pf.plot_efficiency_ptCuts_single_dataset(
    data_TK, data_gen_disp, 'TKMuon', 'theColl._abs_dxy',
    bins=np.arange(-0.05,5,0.5), xlabel=r'$|d_{xy}| \ [cm]$',
    ylabel='Tracking efficiency', title=r'Displaced vs $|d_{xy}|$',
    fig_path=FIG_PATH_TK, save=True, ptCuts=PT_CUTS
)

pf.plot_3_eta_ranges(
    data_TK, data_gen_disp, 'TKMuon', 'theColl._pt',
    bins=np.arange(0,100, 1), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Efficiency', title='Displaced sample',
    fig_path=FIG_PATH_TK, save=True, ptCuts=PT_CUTS
)

pf.plot_efficiency_ptCuts_single_dataset(
    data_displaced_SA, data_gen_disp, 'SAMuon:displaced', 'theColl._pt',
    bins=np.arange(1, 100, 1), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Efficiency', title=r'Displaced sample',
    fig_path=FIG_PATH_SA, save=True, ptCuts=PT_CUTS
)

pf.plot_efficiency_ptCuts_single_dataset(
    data_displaced_SA, data_gen_disp, 'SAMuon:displaced', 'theColl._abs_dxy',
    bins=np.arange(-0.05,100,5), xlabel=r'$|d_{xy}| \ [cm]$',
    ylabel='Efficiency', title=r'Displaced vs $|d_{xy}|$',
    fig_path=FIG_PATH_SA, save=True, ptCuts=PT_CUTS
)
# Efficiency plot for 3 eta ranges
pf.plot_3_eta_ranges(
    data_displaced_SA, data_gen_disp, 'SAMuon:displaced', 'theColl._pt',
    bins=np.arange(0,100, 1), xlabel=r'$gen.p_{T} \, [GeV]$',
    ylabel='Efficiency', title='Displaced sample',
    fig_path=FIG_PATH_SA, save=True, ptCuts=PT_CUTS
)

pf.plot_3_eta_ranges(
    data_TK, data_gen_disp, 'TKMuon', 'theColl._abs_dxy',
    bins=np.linspace(0, 100, 25), xlabel=r'$|d_{xy}| \, [cm]$',
    ylabel='Efficiency', title='Efficiency vs $|d_{xy}|$',
    fig_path=FIG_PATH_TK, save=True, ptCuts=PT_CUTS
)

pf.plot_3_eta_ranges(
    data_displaced_SA, data_gen_disp, 'SAMuon:displaced', 'theColl._abs_dxy',
    bins=np.linspace(0, 100, 25), xlabel=r'$|d_{xy}| \, [cm]$',
    ylabel='Efficiency', title='Efficiency vs $|d_{xy}|$',
    fig_path=FIG_PATH_SA, save=True, ptCuts=PT_CUTS 
)

pf.plot_3_eta_ranges(
    data_prompt_SA, data_gen_prompt, 'SAMuon:prompt', 'theColl._pt',
    bins=np.arange(0,100, 1), xlabel=r'$gen.p_{T} \, [GeV]$',
    ylabel='Efficiency', title='Displaced sample',
    fig_path=FIG_PATH_SA, save=True, ptCuts=PT_CUTS
) 



# pf.histogram_1D_comparison(
#     [data_displaced_SA], ['SAMuon:displaced'], 'theColl._pt',
#     bins=np.arange(0, 200, 1), xlabel=r'$gen.p_{T} \ [GeV]$',
#     ylabel='Counts', title=r'vs $gen.p_{T}$',
#     fig_path=FIG_PATH_SA, save=True 
# )

pf.histogram_1D_comparison(
    [data_prompt_SA,data_displaced_SA], ['SAMuon:prompt','SAMuon:displaced'], 'theL1Obj.commonStubCount',
    bins=np.arange(0,10,1), xlabel=r'Common stub count',
    ylabel='Counts', title=r'Displaced sample',
    fig_path=FIG_PATH_SA, save=True
)

# pf.histogram_1D_comparison(
#     [data_displaced_SA], ['SAMuon:displaced'], 'theL1Obj.commonStubQuality',
#     bins=np.arange(0,20,1), xlabel=r'$Common Stub Quality$',
#     ylabel='Counts', title=r'$Common Stub Quality$',
#     fig_path=FIG_PATH_SA, save=True
# )

pf.histogram_1D_comparison(
    [data_prompt_SA,data_displaced_SA], ['SAMuon:prompt','SAMuon:displaced'], 'theL1Obj.totalStubCount',
    bins=np.arange(0,10,1), xlabel=r'Total stub count',
    ylabel='Counts', title=r'Displaced sample',
    fig_path=FIG_PATH_SA, save=True
)
# pf.histogram_1D_comparison(
#     [data_displaced_SA], ['SAMuon:displaced'], 'theL1Obj.totalStubQuality',
#     bins=np.arange(0,20,1), xlabel=r'$Total Stub Quality$',
#     ylabel='Counts', title=r'Displaced sample',
#     fig_path=FIG_PATH_SA, save=True
# )

pf.plot_mean_comparison(
    [data_prompt_SA,data_displaced_SA], ['SAMuon:prompt','SAMuon:displaced'], 'theColl._pt', 'theL1Obj.commonStubCount',
    bins=np.arange(0, 100, 2), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Common stub count', title=r'Displaced sample',
    fig_path=FIG_PATH_SA, save=True,log=True
)
pf.plot_mean_comparison(
    [data_displaced_SA], ['SAMuon:displaced'], 'theColl._pt', 'theL1Obj.commonStubQuality',
    bins=np.arange(0, 100, 2), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Total Stub Quality', title=r'Displaced sample',
    fig_path=FIG_PATH_SA, save=True
)

data_displaced_SA['commonStubCount_normalized'] = (data_displaced_SA['theL1Obj.commonStubCount'] / data_displaced_SA['theL1Obj.totalStubCount'])
data_displaced_SA['commonStubQuality_normalized'] = (data_displaced_SA['theL1Obj.commonStubQuality'] / data_displaced_SA['theL1Obj.totalStubQuality'])

data_prompt_SA['commonStubCount_normalized'] = (data_prompt_SA['theL1Obj.commonStubCount'] / data_prompt_SA['theL1Obj.totalStubCount'])
pf.plot_mean_comparison(
    [data_prompt_SA,data_displaced_SA], ['SAMuon:prompt','SAMuon:displaced'], 'theColl._pt', 'commonStubCount_normalized',
    bins=np.arange(0, 100, 2), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='common Stub Count normalized', title=r'Displaced sample',
    fig_path=FIG_PATH_SA, save=True, density=True,log=True
)

pf.plot_mean_comparison(
    [data_displaced_SA], ['SAMuon:displaced'], 'theColl._pt', 'commonStubQuality_normalized',
    bins=np.arange(1, 50, 1), xlabel=r'$gen.p_{T} \ [GeV]$',
    ylabel='Common Stub Quality normalized', title=r'Displaced sample',
    fig_path=FIG_PATH_SA, save=True, density=True,log=True
)
