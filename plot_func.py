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
    
# Functions modifying Dataframe (data wrangling, title/cell value modifications) =================================================================================================================================================================
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
    #### Calculate x positions in Fig 1 for dataframes 
    x_pos_map = {label: pos for pos, label in enumerate(df['var'].unique())}
    df['var_numeric'] = df['var'].map(x_pos_map)
    df['var_offset'] = df['var_numeric'] + df['exp'].map({'amip': -0.225, 'hist': -0.075, 'fast': 0.075, 'slow': 0.225}) # best for non-staggering errorbars
    # df['var_offset'] = df['var_numeric'] + df['exp'].map({'amip': -0.3, 'hist': -0.1, 'fast': 0.1, 'slow': 0.3}) # best for staggering errorbars
    return df

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
    Solve Projection Discontinuity problem by interpolation when lon centered at 180 deg. 
    """
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


def mean_error_summary(df):
    return df.groupby(['exp','var','var_numeric','var_offset'])['dR/dT'].agg(['mean', 'std']).reset_index()

# Figures convention and aesthetics =================================================================================================================================================================

def rename_exp(df):
    return df.replace({'amip':'AMIP','hist':'historical','fast': '4xCO2-fast', 'slow': '4xCO2-slow'})

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



