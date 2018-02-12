"""
To use this software, cd to the folder containing the code and type the following in a python console:
import EA_algorithm as ea
df_stu, df_sch = ea.main('path/to/the/datafiles/')


The example tables that were downloaded along with the code include:
1) example_data.csv - the input data set
2) example_map.csv - the mapped values for each area of measurement (these row labels must match the column headers in the dataset)
3) params.csv - this hold all the parameters (file names, amount to allocate, etc...)
"""

import os
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
from scipy import stats

def init(params_df, map_df):

    df=pd.read_csv(params_df['fname'][0])
    df_stu=pd.DataFrame(columns=[])

    # cols=df.columns
    # cols=[c for c in cols if 'Overall Level' in c]
    # cols.sort()
    cols=map_df.index

    df_stu['student']=df['Student Name']
    df_stu['school']=df['School']
    df_stu['family']=df['Family of Schools']
    df_stu['grade']=df['Grade']
    df_stu['student']=df['Student Name']
    df_stu['DOB']=df['Student Date of Birth']

    #for c, m in zip(cols, map_df.index):
    for c in cols:
        nums=[]

        for stu in range(len(df)):

            if df[c][stu] is not np.nan:

                if 'Level 4:' in df[c][stu]:
                    n = map_df.loc[c]['level 4']

                elif 'Level 3:' in df[c][stu]:
                    n = map_df.loc[c]['level 3']

                elif 'Level 2:' in df[c][stu]:
                    n = map_df.loc[c]['level 2']

                elif 'Level 1:' in df[c][stu]:
                    n = map_df.loc[c]['level 1']

                else:
                    n=np.nan

                nums.append(n)

            elif df[c][stu] is np.nan:
                nums.append(np.nan)

        df_stu[c]=nums


    return df_stu

def adjust_values(df_stu, params_df, map_df, stop_flag=False):


    df_stu['mean']=df_stu.agg('mean', axis=1)
    df_sch=df_stu.groupby(['school', 'family'])[['mean']].sum()
    df_stu.drop('mean', axis=1, inplace=True)
    df_sch.rename(columns={'mean':'prop_of_EAs'}, inplace=True)

    total=df_sch['prop_of_EAs'].round().sum()
    diff=abs(params_df['limit'][0]-total)
    numeric_cols = [c for c in df_stu if df_stu[c].dtype.kind == 'f']

    if diff < params_df['tolerance'][0]:
        df_stu['mean'] = df_stu.agg('mean', axis=1)
        df_sch['num_of_EAs']=df_sch['prop_of_EAs'].round()
        df_sch.drop('prop_of_EAs', axis=1, inplace=True)

        # fill log file
        makeLog(params_df, total, map_df, df_stu, df_sch)

        stop_flag=True

    elif total<params_df['limit'][0]:
        #print('up')
        df_stu[numeric_cols] += .001

    elif total>params_df['limit'][0]:
        #print('down')
        df_stu[numeric_cols] -= .001
        df_stu[numeric_cols] = df_stu[numeric_cols].clip_lower(0)

    return df_sch, df_stu, stop_flag

def makeLog(params_df, total, map_df, df_stu, df_sch):

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
    new_log['number_allocated'] = total
    new_log['date'] = stamp
    new_log['map']=map_df.to_string()
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

def main(work_dir):

    # paths
    os.chdir(work_dir)

    # load params
    params_df=pd.read_csv('params.csv')

    # inital values
    map_df=pd.read_csv(params_df['mapname'][0], index_col=0)

    stop_flag=False

    # read data, set up initial df
    df_stu = init(params_df, map_df)

    while not stop_flag:
        # iter until optimized
        df_sch, df_stu, stop_flag = adjust_values(df_stu, params_df, map_df, stop_flag=stop_flag)


    makeBar(df_sch)
    makeKernel(df_sch)

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
        ix.append(df_sch.index[i][0])


    df = pd.DataFrame({'labs': ix, 'data': df_sch['num_of_EAs'].values})
    df=df.sort_values('data')

    fig, ax = plt.subplots(1)
    ax.bar(range(0, len(ix)), df['data'])
    ax.set_xticks(range(len(ix)))
    ax.set_xticklabels(df['labs'], rotation=90)
    plt.subplots_adjust(bottom=.4)
    ax.grid(axis='y')
    ax.set_axisbelow(True)

def makeKernel(df_stu):

    fig, ax = plt.subplots(1)
    data=df_stu['mean']
    density = stats.kde.gaussian_kde(data)
    density.set_bandwidth(.3)
    x = np.arange(0., 1, .01)
    ax.plot(x, density(x))


