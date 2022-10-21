import CO2_gapfill
import os
import pandas as pd
import numpy as np
import plots
from scipy import stats
import random

def opt_xgb_hyperparams_fluxnet():
    """
    Optimizes hyperparameters used in XGB by selecting randomly 10 datasets and selecting the mode from grid search
    """
    filenames = [filen for filen in os.listdir("FLUXNET_2015/") if 'csv' in filen]

    res = pd.DataFrame(columns=["colsample_bytree", "max_depth", "min_child_weight", "subsample"])
    rand_files = random.sample(filenames, 10)

    for i, filen in enumerate(rand_files):
        data = pd.read_csv("FLUXNET_2015/"+filen, index_col=0)
        data.index = data.index.astype(str).str[:12]
        data.index = pd.to_datetime(data.index, format="%Y%m%d%H%M")
        data.replace({-9999: np.nan}, inplace=True)
        data = CO2_gapfill.add_time_vars(data)

        hyp = CO2_gapfill.optimize_hyperparameters(data, ['SW_IN_F', 'VPD_F_MDS', 'TA_F_MDS', 'Month_sin',
            'Month_cos', 'Hour_sin', 'Hour_cos', 'Time'], 'NEE')
        for p in ["colsample_bytree", "max_depth", "min_child_weight", "subsample"]:
            res.loc[i, p] = hyp.get(p)
        print(res)
    
    res = res.mode(axis=0)
    res.to_csv('Results/hyperparams.csv')

    hyperparams = {'colsample_bytree':res.loc[0, 'colsample_bytree'], 'max_depth':int(res.loc[0, 'max_depth']), 'min_child_weight':int(res.loc[0, 'min_child_weight']), "subsample":res.loc[0, "subsample"]}
    return hyperparams

def gapfill_XGB(path,res_path, hyperparams, x_cols, y_col, folds):
    """
    Gapfills data using XGBoost
    """
    for filename in os.listdir(path):
        if 'csv' in filename:
            data = pd.read_csv(path+filename, index_col=0)
            data.index = data.index.astype(str).str[:12]
            data.index = pd.to_datetime(data.index, format="%Y%m%d%H%M")
            data.replace({-9999: np.nan}, inplace=True)
            data = CO2_gapfill.add_time_vars(data)
            gapfilled_data = CO2_gapfill.cv_preds(data, x_cols+['Month_sin', 'Month_cos', 'Hour_sin', 'Hour_cos', 'Time'], y_col, hyperparams, folds)
            gapfilled_data.to_csv(res_path+filename[:-4]+'_xgb.csv')

def calc_res(path, gpf):
    """
    Calculates nighttime, daytime and total bias of XGBoost
    """
    sites = pd.read_excel('sites.xlsx')

    results = pd.DataFrame(columns=['Site',  'Year', 'LAT', 'LON', 'Time', 'Gap_bias'])

    i=0
    orig_path = 'FLUXNET_2015/'
    
    for filename in os.listdir(path):
        print(filename)
        site = filename[4:10]
        lat = sites[sites.SITE_ID == site].LOCATION_LAT.values[0]
        lon = sites[sites.SITE_ID == site].LOCATION_LONG.values[0]

        data = pd.read_csv(path+filename)
        data.replace({-9999:np.nan}, inplace=True)
        year = filename[-12:-8]
        data = data[data.NEE.notnull()] 
        data['Gap_bias'] = data.modelled_NEE-data.NEE
        day = data[data.SW_IN_F>20]
        night = data[data.SW_IN_F<20]        
        day_bias = np.nanmean(day.Gap_bias)
        night_bias = np.nanmean(night.Gap_bias)
        tot_bias = np.nanmean(data.Gap_bias)
        
        results.loc[i] = site, year, lat,  lon, 'Day', day_bias
        i+=1
        results.loc[i] = site, year, lat, lon, 'Night', night_bias
        i+=1
        results.loc[i] = site, year, lat, lon, 'Total', tot_bias
        i+=1

    results['LAT_bin'] = pd.cut(results.LAT, bins = [0,10,20,30,40,50,60,70,80], labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])
  
    # Unit to g C m-2 d-1
    results['Gap_bias'] = results.Gap_bias*0.04401*12/44/1000*86400

    results.to_csv('Results/res_'+gpf+'.csv')



def main():
    # Make directories for results
    os.mkdir('Results/')
    os.mkdir('Results/XGBoost/')

    # Optimize XGBoost hyperparameters
    hyperparams = opt_xgb_hyperparams_fluxnet()

    # Gap-fill, 100-fold CV for measured points
    gapfill_XGB('FLUXNET_2015/','Results/XGBoost/', hyperparams, ['SW_IN_F', 'VPD_F_MDS', 'TA_F_MDS'], 'NEE', 100)
    
    # Calculate results
    calc_res('Results/XGBoost/', 'XGBoost')

    # Make plot
    plots.plot_fig1()

if __name__ == "__main__":
    main()