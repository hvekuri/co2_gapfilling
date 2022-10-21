import os 
import pandas as pd 
from synthetic_data import make_synth_data
import CO2_gapfill
import datetime as dt
import numpy as np
import plots

def make_synth():
    """
    Makes synthetic datasets
    """
    os.mkdir('Synthetic_data/')
    data_path = 'Data_north/'
    
    x_cols = ['TA_F_MDS', 'TS_F_MDS_1', 'SW_IN_F', 'VPD_F_MDS']

    for filename in os.listdir(data_path):
        for i in range(1, 6):
            data = pd.read_csv(
                data_path+filename, index_col=0)
            data.index = pd.to_datetime(
                data.index, format="%Y%m%d%H%M")
            data.replace({-9999:np.nan},inplace=True)
            synth_data = make_synth_data(
                data, x_cols, 'NEE')
            # mumol -> mg
            synth_data['NEE'] = synth_data.NEE*0.04401
            synth_data.to_csv(
                'Synthetic_data/'+ filename[:-4]+str(i)+".csv")


def list_gaps():
    """
    Makes a list of all gaps
    """
    data_path = 'Data_north/'
    for i, filename in enumerate(os.listdir(data_path)):
        gaps = pd.DataFrame()
        df = pd.read_csv(data_path+filename, index_col=0)
        df.index = pd.to_datetime(df.index, format="%Y%m%d%H%M")
        df.loc[df.NEE<-999, 'NEE'] = np.nan
        df = df[['NEE']]
        df1 = df.NEE.isnull().astype(int).groupby(
            df.NEE.notnull().astype(int).cumsum()).sum()
        # Determine the different groups of NaNs. Only keep the 1st. The 0's are non-NaN values, the 1's are the first in a group of NaNs.
        b = df.isna()
        df2 = b.cumsum() - b.cumsum().where(~b).ffill().fillna(0).astype(int)
        df2 = df2.loc[df2['NEE'] <= 1]
        # Set index from the non-zero 'NaN-count' to the index of the first NaN
        df3 = df1.loc[df1 != 0]
        df3.index = df2.loc[df2['NEE'] == 1].index
        # Update the values from df3 (which has the right values, and the right index), to df2
        df2.update(df3)
        df2.columns = ['Gap_len']
        df = df.join(df2)
        df = df[(df.Gap_len <= 3*48) & (df.Gap_len > 0)
                & (df.Gap_len.notnull())]
        df = df[['Gap_len']]
        df['Time'] = df.index.time
        gaps = pd.concat([gaps, df])
        
    return gaps




def gap_lottery(data, gap_list, i, p_gaps):
    """
    Makes 1 gap scenario
    """
    while(len(data[data[i].notnull()])/len(data))*100 < p_gaps:
        gap = gap_list.sample(1)
        row_nr = data[(data[i].isna()) & (
            data.Time == gap.Time.values[0])].sample(1).index[0]

        while data.loc[row_nr:row_nr+gap.Gap_len.values[0]][i].isnull().sum() == 0:
            gap = gap_list.sample(1)
            row_nr = data[(data[i].isna()) & (
                data.Time == gap.Time.values[0])].sample(1).index[0]
        data.loc[row_nr:row_nr+gap.Gap_len.values[0], i] = 1

    return data

def make_gap_scenarios():
    """
    Makes 10 gap scenarios for 30, 50 and 70% data coverages
    """
    os.mkdir('Results_north/Gap_scenarios/')
    gaps = list_gaps()
    gaps.Time = pd.to_datetime(gaps.Time, format="%H:%M:%S").dt.time

    for p_gaps in [30, 50, 70]:
        gap_frame = pd.DataFrame(index=np.arange(
            0, 17568), columns=np.arange(0, 10, 1))
        gap_frame['Time'] = pd.date_range(dt.datetime(
            2020, 1, 1), dt.datetime(2020, 12, 31, 23, 30), freq='30min').time
        for i in range(10):
            gap_frame = gap_lottery(gap_frame, gaps, i, p_gaps)
        gap_frame.to_csv(
            'Results_north/Gap_scenarios/'+"gaps_"+str(p_gaps)+'.csv')



def insert_gaps_to_data(data, gap_scenario, col):
    """
    Inserts artificial gaps into dataframe
    """
    # Some years have 1 day less
    gap_list = gap_scenario[str(col)].values[:len(data)]
    data['Artificial_Gap'] = gap_list
    data.loc[data.Artificial_Gap ==
                 1, 'Hidden_NEE'] = data.NEE.copy()
    data.loc[data.Artificial_Gap == 1, 'NEE'] = np.nan

    return data



def gapfill():
    """
    Optimizes hyperparameters using grid search, gap-fills data and does 10-fold cross-validation for measured data
    """
    x_cols = ['SW_IN_F', 'TA_F_MDS', 'VPD_F_MDS', 'Month_sin',
            'Month_cos', 'Hour_sin', 'Hour_cos', 'Time']
    y_col = 'NEE'

    for filename in os.listdir('Synthetic_data/'):
        orig_data = pd.read_csv('Synthetic_data/'+filename, index_col=0)
        orig_data.index = pd.to_datetime(orig_data.index)
        orig_data = CO2_gapfill.add_time_vars(orig_data)
        hyperparams = CO2_gapfill.optimize_hyperparameters(orig_data, x_cols, y_col)
        for gap_perc in ['30','50','70']:
            gap_file = pd.read_csv('Results_north/Gap_scenarios/'+"gaps_"+str(gap_perc)+'.csv')
            for i in np.arange(0, 10, 1):
                data = orig_data.copy()
                data['complete_NEE'] = data.NEE.copy()
                data = insert_gaps_to_data(data,  gap_file, i)             
                gapfilled_data = CO2_gapfill.cv_preds(data, x_cols, y_col, hyperparams, 10)
                gapfilled_data['gapfilled_NEE'] = gapfilled_data.NEE.copy()
                gapfilled_data.loc[gapfilled_data.NEE.isnull(), 'gapfilled_NEE'] = gapfilled_data.modelled_NEE.copy()
                gapfilled_data.to_csv('Synthetic_gapfilled/'+filename[:-4]+str(i)+'_'+str(gap_perc)+'_xgb.csv')

def calc_errors():
    """
    Calculates errors of the gap-filled C balances
    """
    path = 'Synthetic_gapfilled/'
    i=0
    res = pd.DataFrame(columns=['Site', 'Gaps', 'Gapfiller', 'Error', 'True_balance'])
    for filename in [filen for filen in os.listdir(path) if 'csv' in filen]:
        data = pd.read_csv(path+filename)
        n_gaps = int(np.round(len(data[data.NEE.isnull()])/len(data),2)*100)
        error = ((data.gapfilled_NEE.sum()*1.8) - \
        (data.complete_NEE.sum()*1.8))*(12/44)
        true_bal = data.complete_NEE.sum()*1.8*12/44
        res.loc[i] = filename[4:10], n_gaps, 'XGBoost', error, true_bal
        i += 1

    res.to_csv('Results_north/res_north.csv')

def main():
    # Make synthetic data sets
    make_synth() 

    # Make gap scenarios
    os.mkdir('Results_north/')
    make_gap_scenarios()

    # Gap-fill
    os.mkdir('Synthetic_gapfilled/')
    gapfill()

    # Calculate errors
    calc_errors()
    
    # Make plot
    plots.plot_fig3()

if __name__ == "__main__":
    main()