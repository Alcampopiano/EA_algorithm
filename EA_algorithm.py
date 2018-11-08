"""
To use this software, cd to the folder containing the code and type the following in a python console:
import EA_algorithm as ea
df_stu, df_sch = ea.main()

The example tables that were downloaded along with the code include:
1) example_data.csv - the input data set
2) example_map.csv - the mapped values for each area of measurement (these row labels must match the column headers in the dataset)
3) params.csv - this hold all the parameters (file names, amount to allocate, clipping values, etc...)

Copyright (C) 2017  Allan Campopiano

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
"""

import os
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
from scipy import stats

def init(params_df, map_df):

    df=pd.read_csv(params_df['fname'][0])

    # LOWER CASING COLUMNS IN PARAMS FILE
    df_stu=pd.DataFrame(columns=[])
    cols=map_df.index

    df_stu['student']=df['Student Name']
    df_stu['school']=df['School']
    df_stu['family']=df['Family of Schools']
    df_stu['grade']=df['Grade']

    for c in cols:
        nums=[]

        for stu in range(len(df)):

            if df[c][stu] is not np.nan:

                try:

                    if params_df['L4_replace'][0] in df[c][stu]:
                        n = map_df.loc[c]['level_4']

                    elif params_df['L3_replace'][0] in df[c][stu]:
                        n = map_df.loc[c]['level_3']

                    elif params_df['L2_replace'][0] in df[c][stu]:
                        n = map_df.loc[c]['level_2']

                    elif params_df['L1_replace'][0] in df[c][stu]:
                        n = map_df.loc[c]['level_1']

                    else:
                        n=np.nan

                    nums.append(n)

                except:

                    if params_df['L4_replace'][0] == df[c][stu]:
                        n = map_df.loc[c]['level_4']

                    elif params_df['L3_replace'][0] == df[c][stu]:
                        n = map_df.loc[c]['level_3']

                    elif params_df['L2_replace'][0] == df[c][stu]:
                        n = map_df.loc[c]['level_2']

                    elif params_df['L1_replace'][0] == df[c][stu]:
                        n = map_df.loc[c]['level_1']

                    else:
                        n = np.nan

                    nums.append(n)


            elif df[c][stu] is np.nan:
                nums.append(np.nan)

        df_stu[c]=nums


    return df_stu

def get_grand_avgs(df_stu, map_df):

    grp_labels = ['group ' + str(g) for g in map_df['group'].unique()]
    group_nums = map_df['group'].unique()

    for lab, num in zip(grp_labels, group_nums):
        inds = map_df[map_df['group'] == num].index
        df_stu[lab]=df_stu[inds].mean(axis=1)

    df_stu['mean']=df_stu[grp_labels].agg('mean', axis=1)

    return df_stu

def makeLog(params_df, map_df, df_stu, df_sch):

    stamp=str(datetime.datetime.now()).split('.')[0]
    fname1='student_level_report_' + stamp + '.csv'
    fname2 = 'school_level_report_' + stamp + '.csv'
    fname1=fname1.replace(':', '_')
    fname1=fname1.replace(' ', '_')
    fname2=fname2.replace(':', '_')
    fname2=fname2.replace(' ', '_')

    new_log=pd.DataFrame([], columns=[])
    new_log['work_dir'] = pd.Series(os.getcwd())
    new_log['fname'] = params_df['fname']
    new_log['mapname'] = params_df['mapname']
    new_log['limit'] = params_df['limit']
    new_log['real_limit'] = params_df['real_limit']
    new_log['tolerance'] = params_df['tolerance']
    new_log['number_allocated'] = df_sch['numEAs'].sum()
    new_log['balance'] = params_df['limit']-df_sch['numEAs'].sum()
    new_log['EAs_to_add'] = df_sch['numEAs'].sum() - params_df['limit']
    new_log['date'] = stamp
    new_log['map']=map_df.to_string()
    new_log['clip_upper'] = params_df['clip_upper'][0]
    new_log['clip_lower'] = params_df['clip_lower'][0]
    new_log['output_files']=fname1 + '\n' + fname2 + stamp + '.csv'

    # output files
    df_stu.to_csv(fname1, index=False)
    df_sch.to_csv(fname2)

    if 'log.csv' in os.listdir():
        log_df=pd.read_csv('log.csv')
        log_df=log_df.append(new_log, ignore_index=True)
        log_df.to_csv('log.csv', index=False)

    else:
        new_log.to_csv('log.csv', index=False)

def main():

    # load params
    params_df=pd.read_csv('params.csv')

    # paths
    # os.chdir(work_dir)
    os.chdir(params_df['working_directory'][0])

    # inital values
    map_df=pd.read_csv(params_df['mapname'][0], index_col=0)

    #stop_flag=False

    # read data, set up initial df
    df_stu = init(params_df, map_df)

    # get initital averages
    df_stu = get_grand_avgs(df_stu, map_df)

    # redistribute
    df_stu, df_sch=redistribute(df_stu, params_df)

    # fill log file
    makeLog(params_df, map_df, df_stu, df_sch)

    # while not stop_flag:
    #     # iter until optimized
    #     df_sch, df_stu, stop_flag = adjust_values(df_stu, params_df, map_df, stop_flag=stop_flag)

    plt.ion()
    makeBar(df_sch)
    makeKernel(df_stu)

    return df_stu, df_sch

def makeFakeData(fname):

    df=pd.read_csv(fname)

    rand_list=[]
    for i in range(len(df)):

        x=''.join([random.choice(string.ascii_letters[:26]) for j in range(20)])
        rand_list.append(x)

    cols=[c for c in df.columns if ('Student Name' not in c) and ('Student Date of Birth' not in c) and ('Special Education Consultant' not in c)]

    df['Student Name']=rand_list
    df['Student Date of Birth'] = rand_list
    df['Special Education Consultant'] = rand_list

    for c in cols:
        rnum=np.random.randint(0, len(df), size=len(df))
        df[c]=df[c].values[rnum]

    df.to_csv('example_data.csv', index=False)

def makeBar(df_sch):

    ix=[]
    for i in range(len(df_sch.index)):
        #ix.append(df_sch.index[i][0])
        ix.append(df_sch.index[i])


    df = pd.DataFrame({'labs': ix, 'data': df_sch['numEAs'].values})
    df=df.sort_values('data')

    fig, ax = plt.subplots(1)
    ax.bar(range(0, len(ix)), df['data'])
    ax.set_xticks(range(len(ix)))
    ax.set_xticklabels(df['labs'], rotation=90)
    plt.subplots_adjust(bottom=.4)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    plt.show()

def makeKernel(df_stu):

    fig, ax = plt.subplots(1)
    data=df_stu['numEAs']
    density = stats.kde.gaussian_kde(data[~data.isnull()])
    density.set_bandwidth(.3)
    x = np.arange(0., 1, .01)
    ax.plot(x, density(x))
    plt.show()

def redistribute(df_stu, params_df):

    limit=params_df['limit'][0]
    cur_total=df_stu['mean'].sum()
    cur_diff=limit-cur_total
    df_stu['numEAs']=df_stu['mean']
    new_sum=df_stu['numEAs'].sum()
    zero_or_nan_records = (df_stu['mean'] == 0) | (df_stu['mean'].isna())

    while new_sum != limit:
        if cur_diff>=0:
            to_redistribute = cur_diff / len(df_stu)
            df_stu['numEAs'] = df_stu['numEAs'] + to_redistribute
            df_stu.loc[zero_or_nan_records, 'numEAs'] = 0
            df_stu['numEAs'].clip_upper(params_df['clip_upper'][0], inplace=True)

        else:
            to_redistribute = abs(cur_diff) / len(df_stu)
            df_stu['numEAs'] = df_stu['numEAs'] - to_redistribute
            df_stu['numEAs'].clip_upper(params_df['clip_lower'][0], inplace=True)

        new_sum=df_stu['numEAs'].sum().round(5)
        cur_diff = limit - new_sum

    # nearest half int
    df_sch = df_stu.groupby(['school'])[['numEAs']].sum()
    df_sch=df_sch.apply(lambda x: round(x * 2) / 2)
    #df_sch = df_sch.apply(lambda x: np.floor(x * 2) / 2)

    return df_stu, df_sch











