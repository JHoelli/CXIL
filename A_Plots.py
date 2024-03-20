import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
# TODO Move Plot Functions from .ipynb to here 

def simple_line_plot(dir,filter, save=None):
    plt.figure()
    
    for a in os.listdir(dir) :
        
        if filter in a and not a.endswith('.png'): 
            data=pd.read_csv(f'{dir}/{a}')
            plt.plot(data, label=f'{a}')
    plt.legend()
    plt.show()
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
def interpretaility_plt(dir,filter, save=None):
    '''Split after Baseline, Model Half, Model Full empty !'''
    df= pd.read_csv(dir)
    df_sort=df.sort_values('model')

  
    plt.figure(figsize=(30,10))
    #df['baseline']=np.zeros(len(df))
    #df[['model']=='iter'] ['baseline']='baseline'  
    #df[['model']=='it'] ['baseline']='baseline'  
    if type(filter) is str:
        gfg=sns.boxplot(data=df_sort, x="model", y=f"{filter}")
    else: 
        data = df_sort[ ['model',filter[0]]]
        data['method'] = np.repeat(filter[0], len(data))
        data2 = df_sort[['model',filter[1]]]
        data2.rename(columns={f"{filter[1]}": f"{filter[0]}"}, inplace=True)
        data2['method'] = np.repeat(filter[1], len(data))
        df=pd.concat([data,data2],ignore_index=True)
        gfg=sns.boxplot(data=df, x="model", y=f"{filter[0]}", hue='method')
    gfg.set_ylim(0, 1)
    if save is None:
        plt.show()
    else:
        plt.savefig(save, transparent=True)

def interpretaility_double_plt(dir, save=None):
    df= pd.read_csv(dir)
    rank = pd.DataFrame([])
    rank['value']=df['rank']
    rank['model']=df['model']
    rank['metric']= np.repeat('rank', len(df['rank']))

    rank_mass=pd.DataFrame([])
    rank_mass['value']=df['rank_mass']
    rank_mass['model']=df['model']
    rank_mass['metric']= np.repeat('mass', len(df['rank_mass']))
    df_new=pd.concat([rank, rank_mass], ignore_index=True, join='inner')
    print(df_new.columns)
    print(len(df_new))
  
    plt.figure(figsize=(30,10))
    
    f=sns.boxplot(data=df_new, x="model", y=f"value", hue='metric')
    f.set_ylim(0, 1.5)
    #sns.boxplot(data=df_new, x="model", y=f"value", hue='metric')
    if save is None:
        plt.show()
    else:
        plt.savefig(save, transparent=True)

def table_data_from_running(dir,filter):
    data=pd.DataFrame([])

    acc=[]
    name=[]
    for a in os.listdir(dir) :
        
        if filter in a and not a.endswith('.png'): 
            d=pd.read_csv(f'{dir}/{a}')
            print(d['0'].values[-1])
            acc.append(d['0'].values[-1])
            name.append(a)
    data['name']=name
    data['data']=acc
    print(data)
def table_data(dir,filter):
    data=pd.read_csv(f'{dir}')
    print(data)

def plot_table_data(dir, filter,save = None):
    df= pd.read_csv(dir) 
   
    df_new=df[df['model'].str.contains("empty")]
    df_new['model']=df_new['model'].str.replace('empty', '')
    df_new['model']=df_new['model'].str.replace('model', '')
    df_new['model']=df_new['model'].str.replace('__', '_')
    print(df_new['model'])
    sns.lineplot(data=df_new, x="model", y=f"{filter}", label=str('empty'))

    df_new=df[df['model'].str.contains("half")]
    df_new['model']=df_new['model'].str.replace('half', '')
    df_new['model']=df_new['model'].str.replace('model', '')
    df_new['model']=df_new['model'].str.replace('__', '_')
    print(df_new['model'])
    sns.lineplot(data=df_new, x="model", y=f"{filter}",label=str('half'))

    df_new=df[df['model'].str.contains("full")]
    df_new['model']=df_new['model'].str.replace('full', '')
    df_new['model']=df_new['model'].str.replace('model', '')
    df_new['model']=df_new['model'].str.replace('__', '_')
    print(df_new['model'])
    ax=sns.lineplot(data=df_new, x="model", y=f"{filter}",label=str('full'))
    ax.set_ylim(0, 1)
    #TODO
    ax.axhline(df[df['model']=='model_full'][f'{filter}'].values[0],color = "black", linestyle = "dashed")
    try:
        ax.axhline(df[df['model']=='iter'][f'{filter}'].values[0],color = "grey", linestyle = "dashed")
    except:
        print('No Iter Found')
    #sns.lineplot(data=df_new, x="model", y=f"{filter}", label=str('baseline_batch'))
    #sns.lineplot(data=df_new, x="model", y=f"{filter}", label=str('baseline_iterative'))      
    if save is None:
        plt.show()
    else:
        plt.savefig(save, transparent=True)


def interaction_efficiency(path,metric='f1', save=None, logscale=False):
    '''
    Running Acc
    '''
    plt.figure()
    mi=10000000
    for name in os.listdir(path):
        if metric in name:
            data= pd.read_csv(f'{path}/{name}').drop(columns=['Unnamed: 0'])
            if len(data)<mi: 
                mi = len(data)
            na=name.split('_')[-1].replace('.csv','')
            #print(data)
            plt.plot(data.iloc[:mi], label=f'{na}')
    plt.legend()
    if logscale:
        #{ “linear”, “log”, “symlog”, “logit”, … }
        plt.xscale("log")
    plt.savefig(save)

    

def filter_to_latex(dir, filter):
    '''Split after Baseline, Model Half, Model Full empty !'''

    tab=''
    header=''
    df= pd.read_csv(dir)
    df_sort=df.sort_values('model')

    for m in np.unique(df_sort['model']):
        header +=f'{m} &' 
        slice=df_sort[df_sort['model']==m]
        mean=round(slice[filter].mean(),4)
        std =round(slice[filter].std(),4)
        tab += f'${mean} \pm {std}$ &'
    print(header)
    print(tab)

def filter_to_latex_over_all_Datasets(dir, filter, half=False):
    '''Split after Baseline, Model Half, Model Full empty !'''
    #TODO CHECK SORTING !
    baselines =['baseline', 'empty', 'half', 'full'] 
    head=''
    if half:
        baselines =['baseline', 'empty', 'full']
        dir = dir[~dir['model'].str.contains('half')]
    dir['model']=dir['model'].str.replace('iter','baseline')
    dir['model']=dir['model'].str.replace('model_','')
    dir['model']=dir['model'].str.replace('model','')
    for model in baselines:
        head +=f'& {model}'
    for interaction in np.unique(dir['interaction']):
        for model in np.unique(dir['model']):
            if model not in baselines:
                head +=f'& {model}'
    head +=' \\\\ \\hline '
    head=head.replace('_', ' ')
    body=''

    for dataset in np.unique(dir['dataset']):
        body+=dataset

         # Baselines 
        for model in baselines:
            slice= dir[dir['model']==model]
            slice=slice[slice['dataset']==dataset]
            mean=round(slice[filter].mean(),4)
            std =round(slice[filter].std(),4)
            tab = f'${mean} \pm {std}$ '
            body += f' & {tab}'
         # Models
        for interaction in np.unique(dir['interaction']):
            #print(interaction)
            slice= dir[dir['interaction']==interaction]
            #print(slice)

            for model in np.unique(dir['model']):
                if model not in baselines:
                    slice2= slice[slice['model']==model]
                    slice2=slice2[slice2['dataset']==dataset]
                    mean=round(slice2[filter].mean(),4)
                    std =round(slice2[filter].std(),4)
                    tab = f'${mean} \pm {std}$ '
                    body += f' & {tab}'
        body += ' \\\\ \\hline '

    text = head + body 
    print(text)
    

def acc_over_all_dataset(dir, filter, half = False):
    #TODO Average over all roots ! 
    baselines =['baseline', 'empty', 'half', 'full']
    head=''
    if half:
        baselines =['baseline', 'empty', 'full']
        dir = dir[~dir['model'].str.contains('half')]
    dir['model']=dir['model'].str.replace('iter','baseline')
    dir['model']=dir['model'].str.replace('model_','')
    dir['model']=dir['model'].str.replace('model','')
    for model in baselines:
        head +=f'& {model}'
    for interaction in np.unique(dir['interaction']):
        for model in np.unique(dir['model']):
            if model not in baselines:
                head +=f'& {model}'
    head +=' \\\\ \\hline '
    head=head.replace('_', ' ')
    body=''

    #print(np.unique(dir['model']))
    for dataset in np.unique(dir['dataset']):
        body+=dataset

        # Baselines 
        for model in baselines:
            slice= dir[dir['model']==model]
            slice=round(slice[slice['dataset']==dataset][filter].values[0],4)
            body += f' & ${slice}$'


        # Models
        for interaction in np.unique(dir['interaction']):
            #print(interaction)
            slice= dir[dir['interaction']==interaction]
            #print(slice)
            for model in np.unique(dir['model']):

                if model not in baselines:
                    #print(model)

                    slice2= slice[slice['model']==model]
                    try:
                        slice2=round(slice2[slice2['dataset']==dataset][filter].values[0],4)
                    except: 
                        slice2 = '0'
                    body += f' & ${slice2}$'
        body += ' \\\\  \\hline '

    text = head + body 

    print(text)


def collected_acc_over_all_dataset(data, filter, exclude_empty =True):
    

    df=data.groupby(['dataset','interaction','model'], as_index=False).agg({f'{filter}':['mean','std']})
    print(df.head(5))
    baselines =['baseline', 'empty', 'full']
    head=''

    df['model']=df['model'].str.replace('iter','baseline')
    df['model']=df['model'].str.replace('model_','')
    df['model']=df['model'].str.replace('model','')
    if exclude_empty:
        baselines =['baseline', 'full']
        df=df[df['model']!='empty']
    #print(df)

    # TODO IS HERE A SORT BY NECESSAEY ?

    for model in baselines:
        head +=f'& {model}'
    for interaction in np.unique(df['interaction']):
        for model in np.unique(df['model']):
            if model not in baselines:
                head +=f'& {model}'
    head +=' \\\\ \\hline '
    head=head.replace('_', ' ')
    body=''

    #print(np.unique(dir['model']))
    for dataset in np.unique(df['dataset']):
        body+=dataset

        # Baselines 
        for model in baselines:
            slice= df[df['model']==model]
            slice_a=round(slice[slice['dataset']==dataset][filter]['mean'].values[0],2)
            slice_b=round(slice[slice['dataset']==dataset][filter]['std'].values[0],2)
            body += f' & ${slice_a} \\pm {slice_b}$'


        # Models
        for interaction in np.unique(df['interaction']):
            #print(interaction)
            slice= df[df['interaction']==interaction]
            #print(slice)
            for model in np.unique(df['model']):

                if model not in baselines:
                    #print(model)

                    slice2= slice[slice['model']==model]
                    try:
                        slice2_a=round(slice2[slice2['dataset']==dataset][filter]['mean'].values[0],2)
                        slice2_b=round(slice2[slice2['dataset']==dataset][filter]['std'].values[0],2)
                    except: 
                        slice2_a = '0'
                        slice2_b= '0'
                    body += f' & ${slice2_a} \\pm {slice2_b}$'
        body += ' \\\\  \\hline '

    text = head + body 

    print(text)

def collect_data(dir_caipi, dir_rrr,data, model,full_data_class=None, full_data_inter=None, aggregate = True): 
    '''
    Enable Dataset Aggregation.. 

    '''
    if dir_caipi is not None:
        for file in os.listdir(dir_caipi):
            if file.startswith('classification'):
                #CAIPI
                d=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/{data}/caipi/{caipi_param}/{model}/{file}')
                d['model_type'] =np.repeat(f'{model}',len(d))
                d['model']=d['model'].str.replace('_\d\d', '', regex=True)
                d['dataset'] =np.repeat(f'{data}',len(d))
                d['interaction']=np.repeat(f'CAIPI',len(d))
                seed=file.split('_')[-1]
                d['seed']=np.repeat(seed,len(d))
                if full_data_class is None:

                    full_data_class=d

                else: 
                    full_data_class=pd.concat([full_data_class,d],ignore_index=True)

            elif file.startswith('interpret'):
                di=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/{data}/caipi/{caipi_param}/{model}/{file}')
                di['model_type'] =np.repeat(f'{model}', len(di))
                di['model']=di['model'].str.replace('_\d\d', '', regex=True)
                di['dataset'] =np.repeat(f'{data}', len(di))
                di['interaction']=np.repeat(f'CAIPI',len(di))
                seed=file.split('_')[-1]
                di['seed']=np.repeat(seed,len(di))


                if full_data_inter is None:
                    full_data_inter=di
                else: 
                    full_data_inter=pd.concat([full_data_inter,di],ignore_index=True)
    if dir_rrr is not None:

        #RRR
        for file in os.listdir(dir_rrr):
            if file.startswith('classification'):
                d=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/{data}/RRR/{model}/{file}')
                d['model_type'] =np.repeat(f'{model}',len(d))
                d['dataset'] =np.repeat(f'{data}',len(d))
                d['interaction']=np.repeat(f'RRR',len(d))
                d['model']=d['model'].str.replace('_\d\d', '', regex=True)
                seed=file.split('_')[-1]
                d['seed']=np.repeat(seed,len(d))
                if full_data_class is None:

                    full_data_class=d

                else: 
                    full_data_class=pd.concat([full_data_class,d],ignore_index=True)
            elif file.startswith('interpret'): 
                di=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/{data}/RRR/{model}/{file}')
                di['model_type'] =np.repeat(f'{model}', len(di))
                di['dataset'] =np.repeat(f'{data}', len(di))
                di['interaction']=np.repeat(f'RRR',len(di))
                seed=file.split('_')[-1]
                di['seed']=np.repeat(seed,len(di))
                di['model']=di['model'].str.replace('_\d\d', '', regex=True)
                if full_data_inter is None:
                    full_data_inter=di
                else: 
                    full_data_inter=pd.concat([full_data_inter,di],ignore_index=True)
            else: 
                continue
    return full_data_class,full_data_inter

if __name__=='__main__':
    #data='SimulatedTestingData_Middle_Harmonic_F_1_TS_50'
    #data='decoy_mnist'
    caipi_param= 'MeanAndStdTransformer'
    #caipi_param= 'randomTransformer'
    #caipi_param= 'MeanandstdTrans'
    run_caipi=True
    run_rrr=True
    #data='iris_cancer'
    datasets=['real']#['time10','tabular','toy_classification','decoy_mnist']#['real']#['time10','tabular','toy_classification','decoy_mnist']#['real']#['time10','tabular','toy_classification','decoy_mnist']#['tabular','decoy_mnist','time10','toy_classification']#,'toy_classification']#'toy_classification',
    
    #data='SimulatedTestingData_Middle_Harmonic_F_1_TS_50'
    #data='toy_classification'
    #Results/decoy_mnist/caipi/MeanAndStdTransformer/SimpleMLP
    models=['resnet']#['SimpleMLP']
    #model='CNN'
    full_data_class = None
    full_data_inter = None
    for a in datasets: 
        data=a

        for b in models:
            model=b
            #f'/media/jacqueline/Data/XIL/Results/{data}/RRR/{model}/'
            full_data_class, full_data_inter = collect_data(f'/media/jacqueline/Data/XIL/Results/{data}/caipi/{caipi_param}/{model}', f'/media/jacqueline/Data/XIL/Results/{data}/RRR/{model}/',data, model,full_data_class, full_data_inter)
    print(full_data_class)
    #import sys 
    #sys.exit(1)
    print(collected_acc_over_all_dataset(full_data_class, 'f1'))
    #print('Rank MASS')
    #print(collected_acc_over_all_dataset(full_data_inter, 'rank_mass'))
    #print('Rank ACC')
    #print(collected_acc_over_all_dataset(full_data_inter, 'rank'))
    #print('AuC')
    #print(collected_acc_over_all_dataset(full_data_inter, 'auc'))
            #if run_caipi:
            #    filter_to_latex(f'/media/jacqueline/Data/XIL/Results/{data}/caipi/{caipi_param}/{model}/interpretability_metrics.csv',filter='rank_mass')
            #    interaction_efficiency(path=f'/media/jacqueline/Data/XIL/Results/{data}/caipi/{caipi_param}/{model}/',metric='f1', save=f'/media/jacqueline/Data/XIL/Results/Plots/Finetuning/{data}_caipi_{caipi_param}_{model}_F1_interact_{data}.png')
            #    plt.close()
            #    table_data(f'/media/jacqueline/Data/XIL/Results/{data}/caipi/{caipi_param}/{model}/classification_metrics.csv',filter='f1')
            #    plot_table_data(f'/media/jacqueline/Data/XIL/Results/{data}/caipi/{caipi_param}/{model}/classification_metrics.csv',filter='f1', save=f'/media/jacqueline/Data/XIL/Results/Plots/Finetuning/{data}_caipi_{caipi_param}_{model}_F1_{data}.png')
            #    plt.close()
            #    interpretaility_plt(f'/media/jacqueline/Data/XIL/Results/{data}/caipi/{caipi_param}/{model}/interpretability_metrics.csv','auc', save=f'/media/jacqueline/Data/XIL/Results/Plots/Finetuning/{data}_caipi_{caipi_param}_{model}_auc_{data}.png')
            #    plt.close()
            #    interpretaility_plt(f'/media/jacqueline/Data/XIL/Results/{data}/caipi/{caipi_param}/{model}/interpretability_metrics.csv',['rank','rank_mass'], save=f'/media/jacqueline/Data/XIL/Results/Plots/Finetuning/{data}_caipi_{caipi_param}_{model}_rank_{data}.png')
            #    plt.close()
                #interpretaility_double_plt(f'/media/jacqueline/Data/XIL/Results/{data}/caipi/{caipi_param}/{model}/interpretability_metrics2.csv', save=f'/media/jacqueline/Data/XIL/Results/{data}/caipi/{caipi_param}/{model}/Double.png')
                #plt.close()
            #if run_rrr:
            #    interaction_efficiency(path=f'/media/jacqueline/Data/XIL/Results/{data}/RRR/{model}/',metric='f1', save=f'/media/jacqueline/Data/XIL/Results/Plots/Finetuning/{data}_RRR_{model}_F1_interact_{data}.png')
            #    plt.close()
            #    table_data(f'/media/jacqueline/Data/XIL/Results/{data}/RRR/{model}/classification_metrics.csv',filter='f1')
            #    plot_table_data(f'/media/jacqueline/Data/XIL/Results/{data}/RRR/{model}/classification_metrics.csv',filter='f1', save=f'/media/jacqueline/Data/XIL/Results/Plots/Finetuning/{data}_RRR_{model}_F1_{data}.png')
            #    plt.close()
            #    interpretaility_plt(f'/media/jacqueline/Data/XIL/Results/{data}/RRR/{model}/interpretability_metrics.csv','auc', save=f'/media/jacqueline/Data/XIL/Results/Plots/Finetuning/{data}_RRR_{model}_Auc_{data}.png')
            #    plt.close()
            #    interpretaility_plt(f'/media/jacqueline/Data/XIL/Results/{data}/RRR/{model}/interpretability_metrics.csv',['rank','rank_mass'], save=f'/media/jacqueline/Data/XIL/Results/Plots/Finetuning/{data}_RRR_{model}_rank_{data}.png')
            #    plt.close()
            #    #interpretaility_double_plt(f'/media/jacqueline/Data/XIL/Results/{data}/RRR/{model}/interpretability_metrics2.csv', save=f'/media/jacqueline/Data/XIL/Results/{data}/RRR/{model}/Double.png')
            #    #plt.close()

            #if run_caipi:
                #CAIPI
            #    d=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/{data}/caipi/{caipi_param}/{model}/classification_metrics.csv')
            #    d['model_type'] =np.repeat(f'{model}',len(d))
            #    d['dataset'] =np.repeat(f'{data}',len(d))
            #    d['interaction']=np.repeat(f'CAIPI',len(d))
            #    di=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/{data}/caipi/{caipi_param}/{model}/interpretability_metrics.csv')
            #    di['model_type'] =np.repeat(f'{model}', len(di))
            #    di['dataset'] =np.repeat(f'{data}', len(di))
            #    di['interaction']=np.repeat(f'CAIPI',len(di))

            #    if full_data_class is None:

            #        full_data_class=d
            #        full_data_inter=di
            #    else: 
            #        print('DI', di.head(5))
            #        print('full_data_inter', full_data_inter.head(5))
            #        full_data_class=pd.concat([full_data_class,d],ignore_index=True)
            #        full_data_inter=pd.concat([full_data_inter,di],ignore_index=True)
            #if run_rrr:
                #RRR
            #    d=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/{data}/RRR/{model}/classification_metrics.csv')
            #    d['model_type'] =np.repeat(f'{model}',len(d))
            #    d['dataset'] =np.repeat(f'{data}',len(d))
            #    d['interaction']=np.repeat(f'RRR',len(d))
            #    di=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/{data}/RRR/{model}/interpretability_metrics.csv')
            #    di['model_type'] =np.repeat(f'{model}', len(di))
            #    di['dataset'] =np.repeat(f'{data}', len(di))
            #    di['interaction']=np.repeat(f'RRR',len(di))


            #    if full_data_class is None:

            #        full_data_class=d
            #        full_data_inter=di
            #    else: 
            #        print('DI', di.head(5))
            #        print('full_data_inter', full_data_inter.head(5))
            #        full_data_class=pd.concat([full_data_class,d],ignore_index=True)
            #        full_data_inter=pd.concat([full_data_inter,di],ignore_index=True)
    #print(full_data_class) 
    #print(full_data_inter)  
    #acc_over_all_dataset(full_data_class, 'f1',True)
    #filter_to_latex_over_all_Datasets(full_data_inter, 'rank_mass',True)
    #filter_to_latex_over_all_Datasets(full_data_inter, 'rank',True)