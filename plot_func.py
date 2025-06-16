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
from cartopy.util import add_cyclic_point 

from global_var_and_func import *


warnings.filterwarnings("ignore")


# glob_var for plotting ==============================================================================================================
ccf_str = ['tot', 'sst', 'eis', 'tadv', 'rh', 'omega','ws']
exp_str = ['amip','hist','fast','slow']

x_pos_map = {label: pos for pos, label in enumerate(ccf_str)}


# Data Wrangling on Dataframe =================================================================================================================================================================
def melt(*args):
    return [pd.melt(df, id_vars=['exp', 'model'],
        value_vars=ccf_str,
        var_name='var',
        value_name='dR/dT') for df in args]

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

def x_pos_calc(df):
    x_pos_map = {label: pos for pos, label in enumerate(df['var'].unique())}
    df['var_numeric'] = df['var'].map(x_pos_map)
    df['var_offset'] = df['var_numeric'] + df['exp'].map({'amip': -0.225, 'hist': -0.075, 'fast': 0.075, 'slow': 0.225}) # best for non-staggering errorbars
    # df['var_offset'] = df['var_numeric'] + df['exp'].map({'amip': -0.3, 'hist': -0.1, 'fast': 0.1, 'slow': 0.3}) # best for staggering errorbars
    return df

def rename_exp(df):
    return df.replace({'amip':'AMIP','hist':'historical','fast': '4xCO2-fast', 'slow': '4xCO2-slow'})

def mean_error_summary(df):
    return df.groupby(['exp','var','var_numeric','var_offset'])['dR/dT'].agg(['mean', 'std']).reset_index()

def perglobts_add_cyclic_point(da):
    """
    Inputs
    da: xr.DataArray with dimensions (time,lat,lon)
    """
    # Use add_cyclic_point to interpolate input data
    lon_idx = da.dims.index('lon')
    wrap_data, wrap_lon = add_cyclic_point(da, coord=da.lon, axis=lon_idx)
    
    coords = {
        "exp":da.coords["exp"],
        "model":da.coords["model"],
        "lat":da.coords["lat"],
        "lon": wrap_lon
    }
    
    # Generate output DataArray with new data but same structure as input
    output_da = xr.DataArray(data=wrap_data, 
                           coords = coords, 
                           dims=da.dims, 
                           attrs=da.attrs)
    
    return output_da 

def dR_add_cyclic_point(da):
    """
    coords diff from perglobts. 
    """
    # Use add_cyclic_point to interpolate input data
    lon_idx = da.dims.index('lon')
    wrap_data, wrap_lon = add_cyclic_point(da, coord=da.lon, axis=lon_idx)
    
    coords = {
        "exp":da.coords["exp"],
        "lat":da.coords["lat"],
        "lon": wrap_lon
    }
    
    # Generate output DataArray with new data but same structure as input
    output_da = xr.DataArray(data=wrap_data, 
                           coords = coords, 
                           dims=da.dims, 
                           attrs=da.attrs)
    
    return output_da 


# # Calculate Mean for all df
# for i in [df_ikic,df_ceres,df_modis,df_isccp,df_patmos,df_mmccf,df_mmkern]:
#     for j in range(len(ccf_str)):
#         for k in range(len(exp)):
#             print(exp_str[k],ccf_str[j])
#             i = get_mean_df(i,exp_str[k],ccf_str[j])

#Plotting funcs =================================================================================================================================================================
# def add_column_titles(fig, titles, y_position=0.9, fontsize=12, fontweight='bold'):
#     """
#     Adds column titles to a figure.
    
#     Parameters:
#     - fig: Matplotlib figure object.
#     - titles: List of titles for each column.
#     - y_position: y-coordinate for the titles (default is 0.9).
#     - fontsize: Font size for the titles (default is 12).
#     - fontweight: Font weight for the titles (default is 'bold').
#     """
#     num_titles = len(titles)
#     for i, title in enumerate(titles):
#         x_position = (i + 1) / (num_titles + 1)  # Evenly spaced titles
#         fig.text(x_position, y_position, title, ha='center', va='center', fontsize=fontsize, fontweight=fontweight)


def rename_modname(dfs):
    renamed_dfs = []
    for df in dfs:
        df = df.copy()  
        df['model'] = df['model'].replace(dict(zip(model_list, modelfullname_list)))
        renamed_dfs.append(df)
    return renamed_dfs


def add_row_titles_from_gridspec(fig, row_titles, gs, x_position=0.09, **kwargs):
    """
    Adds row titles aligned with rows of a GridSpec layout.

    Parameters:
    - fig: matplotlib.figure.Figure
    - row_titles: list of row title strings
    - gs: matplotlib.gridspec.GridSpec object
    - x_position: float, x coordinate in figure space for the titles
    - kwargs: any additional arguments passed to fig.text (e.g. fontsize)
    """
    total_rows = gs.nrows

    for i, title in enumerate(row_titles):
        # Get the bbox (bounding box) for the leftmost cell in this row
        cell = gs[i, 0]
        # Compute vertical center of this row in figure coordinates
        y_center = (cell.get_position(fig).y0 + cell.get_position(fig).y1) / 2
        fig.text(x_position, y_center, title, va='center', ha='left',
                 rotation=90, **kwargs)


#### Calculate x positions in Fig 1 for dataframes 
def x_pos_calc(df):
    x_pos_map = {label: pos for pos, label in enumerate(df['var'].unique())}
    df['var_numeric'] = df['var'].map(x_pos_map)
    df['var_offset'] = df['var_numeric'] + df['exp'].map({'amip': -0.225, 'hist': -0.075, 'fast': 0.075, 'slow': 0.225})
    return df


def stagger_not(df,stag): #stag takes 'yes'/'no'; also require change of values in x_pos_calc 
    if stag == 'yes':
        return [j.assign(var_offset=j['var_offset'] + 0.1) for j in df]
    elif stag == 'no':
        return df
