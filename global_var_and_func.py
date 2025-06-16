import numpy as np 
import pandas as pd 
import xarray as xr 
import cartopy 
import glob
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import scipy 
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, make_union
import warnings
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.offsetbox as offsetbox
import seaborn.objects as so
from seaborn import axes_style
import matplotlib as mpl
import pickle
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


warnings.filterwarnings("ignore")

#=================================================================================================================================================================
#Global Variables 
root_path='/data/keeling/a/rytam2/ccf_model_spread/data/preprocessed/'
kernel_path='/data/keeling/a/rytam2/a/met_kernels/gcms/'
obs_kernel_path='/data/keeling/a/rytam2/a/met_kernels/obs/'

textbox_labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u']
zonal_plt_color = ['#E69F00','#0072B2','#CC79A7','#009E73']
column_titles = ['AMIP', 'Historical', '4x$CO_{2}$-fast', '4x$CO_{2}$-slow', 'Zonal Mean']

modelfullname_list = ['CCSM4','CanESM2','CanESM5','E3SM-1-0','GFDL-CM4','HadGEM2','HadGEM3-GC31-LL','IPSL-CM6A-LR','MIROC-ES2L','MIROC-ESM',
                     'MIROC5','MIROC6','MPI-ESM','MRI-CGCM3','MRI-ESM2-0','UKESM1-0-LL']
model_list=['ccsm4','canam4','canesm5','e3sm','gfdl','hg2','hg3','ipsl','mies2l','miesm','mi5','mi6','mpi','mrcgcm','mresm','ukesm']
rename_dict = dict(zip(modelfullname_list, model_list))

ccf_str = ['tot', 'sst', 'eis', 'tadv', 'rh', 'omega','ws']
exp_str = ['amip','hist','fast','slow']

#Kernel Data
with open('kernel.pkl', 'rb') as f:
    (sst_kernels, eis_kernels, tadv_kernels, rh_kernels, omega_kernels, ws_kernels,
     sst_obs_kernels, eis_obs_kernels, tadv_obs_kernels, rh_obs_kernels, omega_obs_kernels, ws_obs_kernels) = pickle.load(f)

#==================================================================================================================================================================

# Global Functions 
def renamevarname(var):
    return var.rename(rename_dict)

def get_data(exp):
    ts = xr.open_mfdataset(root_path+'ts_*'+exp+'_CMIP5&6_*.nc')
    eis = xr.open_mfdataset(root_path+'eis_*'+exp+'*_CMIP5&6_*.nc')
    tadv = xr.open_mfdataset(root_path+'tadv_*'+exp+'*_CMIP5&6_*.nc')*24*3600
    rh = xr.open_mfdataset(root_path+'hur_*'+exp+'*_CMIP5&6_*.nc')
    omega = xr.open_mfdataset(root_path+'wap_*'+exp+'*_CMIP5&6_*.nc')*864
    ws = xr.open_mfdataset(root_path+'ws_*'+exp+'*_CMIP5&6_*.nc')
    
    if exp=='4xCO2':
        ts,eis,tadv,rh,omega,ws = renamevarname(ts),renamevarname(eis),renamevarname(tadv),renamevarname(rh),renamevarname(omega),renamevarname(ws)
    return ts,eis,tadv,rh,omega,ws

def extract_abrupt(ds_abrupt): 
    ds_fast = ds_abrupt.isel(time=slice(0,240))
    ds_slow = ds_abrupt.isel(time=slice(240,1752))#1800)) #4/24:EDIT END INDEX to 1800 AFTER CMIP6 4XCO2 DATA DOWNLOAD
    return ds_fast, ds_slow

def remove_time_mean(CCF_tseries):
    # CCF_anom - applies removal of seasonal cycle to all 16 models in the dataset
    return CCF_tseries.groupby("time.month") - CCF_tseries.groupby("time.month").mean()

def spatial_weighted_mean(ds):
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    ds_weighted = ds.weighted(weights)
    return ds_weighted.mean(('lat','lon'))

def ensemble_mean(x):
    if "model" in x.dims:
        return x.mean('model')
    else:
        return x.to_array(dim='model').mean('model')

def hat_and_zonal(var):
    # gets MMM and MMM-zonal mean 
    var_hat = ensemble_mean(var)
    var_hat_zonal = var_hat.mean("lon")
    return var_hat, var_hat_zonal

def make_textbox(axes, string):

    box1 = offsetbox.TextArea(string,textprops=dict(fontsize=14,ha='left',fontweight='bold'))
    anchored_box = offsetbox.AnchoredOffsetbox(loc=3,
                                 child=box1, pad=0.2,
                                 frameon=False,
                                 bbox_to_anchor=(0,1),
                                 bbox_transform=axes.transAxes,
                                 borderpad=.2)
    axes.add_artist(anchored_box)
    
    return

def melt(*args):
    return [pd.melt(df, id_vars=['exp', 'model'],
        value_vars=ccf_str,
        var_name='var',
        value_name='dR/dT').set_index("idx", inplace=True) for df in args]

def get_mean_df(df,exp,var):
    df_subset = df.loc[df['exp'].isin([exp]) & df['var'].isin([var])]
    dRdT_mean = df_subset['dR/dT'].mean()
    
    new_row = {'exp':exp,
               'model':'mean',
               'var':var,
               'dR/dT':dRdT_mean,
               'var_numeric':df_subset['var_numeric'].unique()[0],
               'var_offset': df_subset['var_offset'].unique()[0]}
    df.loc[len(df)] = new_row
    return df


# Main Calculation Functions ==================================================================================================
## landmask variable 
with open('kernel.pkl', 'rb') as f:
    kernels = pickle.load(f)
sst_kernels=kernels[0]
landmask=(~np.isnan(sst_kernels.ccsm4)).astype('int').where((sst_kernels.lat>-60)&(sst_kernels.lat<60))


# !!!!! OLD CODE start ======
### Glob Temp Change Trend
# ts_amip = xr.open_mfdataset(root_path+'ts_'+'amip'+'_CMIP5&6_*.nc')
# glob_ts_anom = spatial_weighted_mean(ts_amip) #weighted mean 
# glob_ts_poly = glob_ts_anom.polyfit(dim="time",deg=1)
# glob_ts_trends = glob_ts_poly.sel(degree=1).drop_vars("degree").rename(dict(zip(glob_ts_poly.data_vars, model_list))) #dK/dt (degC/year)

# def ccf_trends(landmask, ds): 
#     ds_anom = ds.map(remove_time_mean)*landmask #monthly anonaly tseries at all location 
#     ds_poly = ds_anom.polyfit(dim="time",deg=1)
#     ds_trends = ds_poly.sel(degree=1).drop_vars("degree").rename(dict(zip(ds_poly.data_vars, model_list))) #dK/dt (degC/year)
#     return ds_trends


# def ccf_changes_perglobts(landmask, ds_amip, ds_hist, ds_fast, ds_slow): 
#     # returns datasets of 1) unit of CCF change per degree of warming (K/K, [lat, lon]) 2) model ensemble of (1), 3) zonal mean of (2); 
#     # 4) weighted glob mean of (1); and model-ensemble mean of (4)
#     ds_trend_amip = ccf_trends(landmask, ds_amip)
#     ds_trend_hist = ccf_trends(landmask, ds_hist)
#     ds_trend_fast = ccf_trends(landmask, ds_fast)
#     ds_trend_slow = ccf_trends(landmask, ds_slow)

#     ds_trends = xr.concat([ds_trend_amip,ds_trend_hist,ds_trend_fast,ds_trend_slow],dim=xr.DataArray(["amip","hist","fast","slow"],dims="exp",name="experiment"))
    
#     ds_perglobts = ds_trends/glob_ts_trends # get unit change of CCF(ts) per change in glob ts dK/dK (K/K)
#     ds_perglobts_hat = ensemble_mean(ds_perglobts) #model mean (hat)
#     ds_perglobts_hat_zonal = ds_perglobts_hat.mean("lon")#zonal-model mean 
#     ds_perglobts_glob = spatial_weighted_mean(ds_perglobts) #spatially weighted avged trends for each model 
#     ds_perglobts_hat_glob = spatial_weighted_mean(ds_perglobts_hat) #spatially weighted avged trends of model ensemble

#     return ds_perglobts, ds_perglobts_hat, ds_perglobts_hat_zonal, ds_perglobts_glob, ds_perglobts_hat_glob  
# !!!!! OLD CODE END ======


# Goal: loop thorugh model and replace each CCF (time,lat,lon)'s time coordinates to temp changes - varying dKs for each models!! then take trends (dCCF/dT - [unit/K])
# Using SST as example 

# takes input: var to be regressed (lat,lon,time,exp; 16 var from each models) , TS from each experiment (lat,lon,time; 16 var from each models) 

# Goal: loop thorugh model and replace each CCF (time,lat,lon)'s time coordinates to temp changes - varying dKs for each models!! then take trends (dCCF/dT - [unit/K])
# Using SST as example 

# takes input: var to be regressed (lat,lon,time,exp; 16 var from each models) , TS from each experiment (lat,lon,time; 16 var from each models) 
#==================================================================================================================
#### NOTE: FUNCTION RETIRED: ENSO signal masks global warming signal 
# def regression(glob_ts,var_anom):
#     combined = []
#     for i in model_list:
#         # Replace time dimension with change in Temp and take regression 
#         da = var_anom.rename({"time":'dT'})
#         da = da[i].assign_coords(dT=glob_ts.sel(model=i).data)

#         mask = da.notnull().compute()
#         da_masked = da.where(mask, drop=True)
        
#         da_poly = da_masked.polyfit(dim="dT",deg=1)
#         dVar_dT_model = da_poly.sel(degree=1).drop_vars("degree").polyfit_coefficients 
#         combined.append(dVar_dT_model)

#     ds_combined = xr.concat(combined, dim="model").assign_coords(model=("model", model_list))
#     return ds_combined
#==================================================================================================================

def regression(var_anom):
    combined = []
    for i in model_list:
        # print(i)
        mask = var_anom[i].notnull().compute()
        da_masked = var_anom[i].where(mask, drop=True)
        
        da_poly = da_masked.polyfit(dim="time",deg=1)
        dVar_dT_model = da_poly.sel(degree=1).drop_vars("degree").polyfit_coefficients 
        combined.append(dVar_dT_model)

    ds_combined = xr.concat(combined, dim="model").assign_coords(model=("model", model_list))
    return ds_combined
    
def var_perglobts(ts_exp,var,smooth): # WORKING 
    #var_perglobts 
    ts_anom = ts_exp.map(remove_time_mean)
    glob_ts_anom = spatial_weighted_mean(ts_anom) #weighted mean tseries #includes land # dataarray (model,time)
    
    # CCF time series
    var_anom = var.map(remove_time_mean)*landmask #removed seasonality and land (lat,lon,time)
    
    if smooth==True: 
        # smooth with running mean of 5 years to minimize ENSO effect 
        ### TODO: Check Tseries and Impact 
        dTg_rolling = glob_ts_anom.rolling(time=12*5, center=True).mean()
        var_rolling = var_anom.rolling(time=12*5, center=True).mean()
    
        dTg_dt = regression(dTg_rolling) # REPETITIVE CALC FOR dTg_dt
        dvar_dt = regression(var_rolling)
    
    elif smooth == False: 
        dTg_dt = regression(glob_ts_anom)
        dvar_dt = regression(var_anom)
    
    return dvar_dt/dTg_dt

def ccf_changes_perglobts(ts_amip, var_amip,\
                          ts_hist, var_hist,\
                          ts_fast, var_fast,\
                          ts_slow, var_slow,smth): 
    perglobts_amip = var_perglobts(ts_amip,var_amip,smooth=smth)
    print('amip done!')
    perglobts_hist = var_perglobts(ts_hist,var_hist,smooth=smth)
    print('hist done!')
    perglobts_fast = var_perglobts(ts_fast,var_fast,smooth=smth)
    print('fast done!')
    perglobts_slow = var_perglobts(ts_slow,var_slow,smooth=smth)
    print('slow done!')
    ds_perglobts = xr.concat([perglobts_amip,perglobts_hist,perglobts_fast,perglobts_slow],dim=xr.DataArray(["amip","hist","fast","slow"],dims="exp",name="experiment"))
    
    perglobts_glob = spatial_weighted_mean(ds_perglobts) #spatially weighted avged trends for each model [model,exp]
    # perglobts_hat = ensemble_mean(ds_perglobts) #model mean (hat) [lat, lon,exp]
    # perglobts_hat_zonal = perglobts_hat.mean("lon")#zonal-model mean [lat,exp]
    # perglobts_hat_glob = ensemble_mean(perglobts_glob) #spatially weighted avged trends of model ensemble [exp]

    return ds_perglobts#, perglobts_glob, perglobts_hat, perglobts_hat_zonal, perglobts_hat_glob


def get_feedback(ccf_perglobts, mod_kernel,obs_kernels=None): 
    dR_ccf_perglobts = ccf_perglobts*mod_kernel # permodel*permodel 
    
    if obs_kernels is not None:
        dR_ccf_perglobts_ceres = ccf_perglobts*obs_kernels['ceres']
        dR_ccf_perglobts_isccp = ccf_perglobts*obs_kernels['isccp']
        dR_ccf_perglobts_modis = ccf_perglobts*obs_kernels['modis']
        dR_ccf_perglobts_patmosx = ccf_perglobts*obs_kernels['patmosx']
        dR_ccf_perglobts_mmkern = ccf_perglobts*ensemble_mean(mod_kernel)
        # print('obs kernel calc done')
        return dR_ccf_perglobts,dR_ccf_perglobts_ceres,dR_ccf_perglobts_isccp,dR_ccf_perglobts_modis,dR_ccf_perglobts_patmosx,dR_ccf_perglobts_mmkern
    
    return dR_ccf_perglobts

def ccf_feedbacks(dRsst,dReis,dRtadv,dRrh,dRomega,dRws):
    dR_tot = dRsst+dReis+dRtadv+dRrh+dRomega+dRws
    dR_arr = xr.concat([dR_tot, dRsst, dReis, dRtadv, dRrh, dRomega, dRws],
                           dim=xr.DataArray(["tot","sst","eis","tadv","rh","omega","ws"], dims="ccf", name="ccf"))
        
    if isinstance(dR_arr, xr.Dataset):
        return dR_arr.to_array('model').to_dataset(dim='ccf')
    elif isinstance(dR_arr, xr.DataArray):
        return dR_arr.to_dataset(dim='ccf')

def prime_mmm_calc(kernel,ccf_perglobts):
    kernel_mmm = ensemble_mean(kernel)
    kernel_prime = (kernel - kernel_mmm).to_array(dim='model')
    ccf_perglobts_mmm = ensemble_mean(ccf_perglobts)
    ccf_perglobts_prime = ccf_perglobts - ccf_perglobts_mmm
    
    dR_kp_cp = kernel_prime * ccf_perglobts_prime
    dR_km_cp = kernel_mmm * ccf_perglobts_prime
    dR_kp_cm = kernel_prime * ccf_perglobts_mmm
    dR_km_cm = kernel_mmm * ccf_perglobts_mmm

    return dR_kp_cp, dR_km_cp, dR_kp_cm, dR_km_cm

def indv_var_calc(kernel,ccf_perglobts,dR_ik_ic,ccf):
    # goal: return a [lat,lon,exp,comb] where comb is the combination of ik-ic, km-cp, kp-cm, diff 
    # func should be applied to all 6 ccfs except the total 
    dR_ccf_kp_cp,dR_ccf_km_cp,dR_ccf_kp_cm,dR_ccf_km_cm = prime_mmm_calc(kernel,ccf_perglobts)
    
    var_dR_ik_ic = dR_ik_ic.var('model').compute() #plot
    
    var_dR_kp_cp = dR_ccf_kp_cp.var('model') 
    var_dR_km_cp = dR_ccf_km_cp.var('model').compute() #plot
    var_dR_kp_cm = dR_ccf_kp_cm.var('model').compute() #plot 
    
    var_dR_aprox = var_dR_km_cp+var_dR_kp_cm#holds the assumption that kp-cp is near 0; var(km-cm) is 0
    var_dR_sum = var_dR_kp_cp + var_dR_km_cp + var_dR_kp_cm  #for comparison 
    var_diff = var_dR_ik_ic[ccf] - var_dR_aprox.compute() #plot 

    var_ds = xr.concat([var_dR_ik_ic[ccf], var_dR_km_cp, var_dR_kp_cm, var_diff, var_dR_sum],
                           dim=xr.DataArray(["ik_ic","km_cp","kp_cm","indv-aprx","aprx+kp_cp"], dims="comb", name="comb"))

    return var_ds