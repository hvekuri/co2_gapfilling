import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from matplotlib import colors
import geopandas as gpd
import seaborn as sns 

def plot_fig1():
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    res1 = pd.read_csv('Results/res_XGBoost.csv')

    # Add some scatter to see more points
    res1.loc[(res1.LAT>20) & (res1.LAT<70), 'LON'] = res1['LON'] + np.random.normal(loc=0, scale=5, size=len(res1))
    res1['LAT'] = res1['LAT'] + np.random.normal(loc=0, scale=.5, size=len(res1))

    divnorm = colors.TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=.1)

    night = res1[res1.Time == 'Night']
    tot = res1[res1.Time=='Total']
    day = res1[res1.Time == 'Day']


    fig, ax = plt.subplots(3,1, figsize=[15,10])

    fig.subplots_adjust(wspace=0, hspace=0)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[(world.pop_est > 0) & (world.name != "Antarctica")]
    world = world[['continent', 'geometry']]
    continents = world.dissolve(by='continent')

    for axx in [ax[0], ax[1],ax[2]]:
        continents.plot(facecolor="whitesmoke", edgecolor="k", ax=axx)
        axx.set_aspect('auto')
        axx.set_ylim(0, 79)
    ax[0].text(0.15, 0.1, 'XGB, Day', ha='center',
                  va='center', transform=ax[0].transAxes, fontsize=20)
    ax[1].text(0.15, 0.1, 'XGB, Night', ha='center',
                  va='center', transform=ax[1].transAxes, fontsize=20)
    ax[2].text(0.15, 0.1, 'XGB, Total', ha='center',
                  va='center', transform=ax[2].transAxes, fontsize=20)              

    cm = 'bwr'
    al = .9

    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])

    im = ax[0].scatter(x=day.LON, y=day.LAT, c=day.Gap_bias,
                     cmap=cm,  s=30, norm=divnorm,  alpha=al)
    ax[1].scatter(x=night.LON, y=night.LAT, c=night.Gap_bias,
                     cmap=cm,  s=30, norm=divnorm, alpha=al)
    ax[2].scatter(x=tot.LON, y=tot.LAT, c=tot.Gap_bias,
                     cmap=cm,  s=30,  norm=divnorm, alpha=al)
    fig.subplots_adjust(right=0.84)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.01, 0.8])
    fig.colorbar(im, cax=cbar_ax, extend='both')
    cbar_ax.set_ylabel(
       'Bias [g C m$^{-2}$ d$^{-1}$]', labelpad=20, fontsize=20)
    fig.text(0.5, 0.01, 'Longitude', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=20)
    plt.show()


def plot_fig3():
    data = pd.read_csv('Results_north/res_north.csv')
    col = 'Error'
    ylab = 'Error [g C m$^{-2}$ y$^{-1}$]'
    data['Gaps'] = np.round(data.Gaps, 2)
    data['Gapfiller_gaps'] = data['Gapfiller']+' '+data.Gaps.astype(str)+'%'
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    fig, ax = plt.subplots(1, figsize=[15,10])
    mypal = {'XGBoost 30%': 'dimgrey','XGBoost 50%': sns.set_hls_values('dimgrey', l=.5), 'XGBoost 70%': sns.set_hls_values('dimgrey', l=.9)}
    sns.boxplot(x=data.Site, y=data[col], hue=data['Gapfiller_gaps'],  ax=ax, flierprops = dict(markersize=3), palette=mypal)
    ax.set_xlabel("Site", fontsize=20)
    ax.set_ylabel(ylab, fontsize=20)
    ax.axhline(y=0, c='k')
    ax.legend(bbox_to_anchor=(1.01, 1), loc=2, prop={'size': 14})
    fig.subplots_adjust(right=.85)
    plt.show()