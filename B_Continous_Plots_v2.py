import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
def collect_data(dir_CAIPI, dir_rrr,dir_sanity,data, model,full_data_class=None, full_data_inter=None, aggregate = True, interpret=False): 
    '''
    Enable Dataset Aggregation.. 

    '''
    print(model)

    CAIPI_param='MeanAndStdTransformer'
    if dir_CAIPI is not None:
        for file in os.listdir(dir_CAIPI):
            if file.startswith('classification'):
                #CAIPI
                d=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/Continous/{data}/CAIPI/{CAIPI_param}/{model}/{file}')
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

            elif interpret and  file.startswith('interpret'):
                di=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/Continous/{data}/CAIPI/{CAIPI_param}/{model}/{file}')
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
                d=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/Continous/{data}/RRR/{model}/{file}')
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
            elif interpret and file.startswith('interpret'): 
                di=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/Continous/{data}/RRR/{model}/{file}')
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
    if dir_sanity is not None:
        #RRR
        for file in os.listdir(dir_sanity):

            if file.startswith('classification'):
                d=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/Continous/{data}/Sanity/{model}/{file}')
                d['model_type'] =np.repeat(f'{model}',len(d))
                d['dataset'] =np.repeat(f'{data}',len(d))
                d['interaction']=np.repeat(f'Sanity',len(d))
                d['model']=d['model'].str.replace('_\d\d', '', regex=True)
                seed=file.split('_')[-1]
                d['seed']=np.repeat(seed,len(d))
                if full_data_class is None:

                    full_data_class=d

                else: 
                    full_data_class=pd.concat([full_data_class,d],ignore_index=True)
            elif interpret and file.startswith('interpret'): 
                di=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/Continous/{data}/Sanity/{model}/{file}')
                di['model_type'] =np.repeat(f'{model}', len(di))
                di['dataset'] =np.repeat(f'{data}', len(di))
                di['interaction']=np.repeat(f'Sanity',len(di))
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


def collect_data_taskwise(dir_CAIPI, dir_rrr,dir_sanity,data, model,full_data_class=None, full_data_inter=None, aggregate = True, interpret=False): 
    '''
    Enable Dataset Aggregation.. 

    '''
    print(model)
    CAIPI_param='MeanAndStdTransformer'
    
    if dir_CAIPI is not None:
        for file in os.listdir(dir_CAIPI):
            if file.startswith('model') and file.endswith('csv'):
                #CAIPI
                d=pd.read_csv(f'/media/jacqueline/Data/XIL/Results/Continous/{data}/CAIPI/{CAIPI_param}/{model}/{file}')
                lt= 'LT' in file
                df=pd.DataFrame([])
                df['LT'] =np.repeat(f'{lt}',5)
                df['tasknumber']=[1,2,3,4,5]
                df['value']=[d['task_0'].values[-1],d['task_1'].values[-1],d['task_2'].values[-1],d['task_3'].values[-1],d['task_4'].values[-1]]
                #df['task1']=np.repeat(d['task_0'].values[-1],1)
                #df['task2']=np.repeat(d['task_1'].values[-1],1)
                #df['task3']=np.repeat(d['task_2'].values[-1],1)
                #df['task4']=np.repeat(d['task_3'].values[-1],1)
                #df['task5']=np.repeat(d['task_4'].values[-1],1)
                
                model_name=file.split('_')[1]
                df['model']=np.repeat(f'{model_name}',5)
                
                df['dataset'] =np.repeat(f'{data}',5)
                df['interaction']=np.repeat(f'CAIPI',5)
                seed=file.split('_')[-1]
                df['seed']=np.repeat(seed,5)
                if full_data_class is None:

                    full_data_class=df

                else: 
                    full_data_class=pd.concat([full_data_class,df],ignore_index=True)
    print(full_data_class)
    return full_data_class

def latex_table(data, filter):
    #TODO Sort Alpahbetical
    print(data['dataset'].unique())
    datasets=data['dataset'].unique()#np.unique(data['dataset'].values).sort()
    models =data['model'].str.replace('_LT','').unique()# np.unique(data['model'].values).sort()
    interactions=['CAIPI', 'RRR', 'Sanity']

    string_full='& '
    for d in datasets:
         d=d.replace('_',' ')
         # TODO ADD Multi Column 
         string_full+= '& \\multicolumn{2}{|c|}{'
         string_full+=f'{d}'
         string_full+='}'
    string_full += f' \\\\ \\hline '
    string_full+= '&'
    for i in datasets:
        string_full+= '& &LT'
    string_full += f' \\\\ \\hline '

    for interact in interactions:
        filter_int=data[data['interaction']== interact]
        string_full+= "\\multirow{6}{*}{"
        string_full+= f"{interact}"
        string_full+="}}"
        save= True
        for model in models:            #string_full += f'{model}'
            filter_int_model = filter_int[filter_int['model'].str.contains(model)]
            model=model.replace('model', '')
            model=model.replace('_',' ')
            #if save:
            #    string_full += f' {model}'
            #    save=False
            #else:
            string_full += f'& {model}'


            for d in datasets:
                filter_int_model_data=filter_int_model[filter_int_model['dataset']==d]
                #TODO Calculation
                #mean= 0
                #std= 0
                df=filter_int_model_data[~filter_int_model_data['model'].str.contains('LT')].agg({f'{filter}':['mean','std']})
                df2=filter_int_model_data[filter_int_model_data['model'].str.contains('LT')].agg({f'{filter}':['mean','std']})
                print(df)
                print(df2)
                #print(df
                mean=round(df[filter]['mean'],2)
                std=round(df[filter]['std'],2)

                mean2=round(df2[filter]['mean'],2)
                std2=round(df2[filter]['std'],2)
                 
                string_full += f'& ${mean} \\pm {std} $ &  ${mean2} \\pm {std2}$'
        
            string_full += ' \\\\ \\cline{2-8}'
        string_full += ' \\hline'

    print(string_full)



def latex_table_real(data, filter):
    #TODO Sort Alpahbetical
    print(data['dataset'].unique())
    datasets=data['dataset'].unique()#np.unique(data['dataset'].values).sort()
    models =data['model'].str.replace('_LT','').unique()# np.unique(data['model'].values).sort()
    interactions=['CAIPI', 'RRR', 'Sanity']
    model_interact=['ADAM', 'SGD','replay']
    if 'task' in data['dataset'].unique():
        model_2=['ADAM', 'SGD', 'StartingModel', 'UpperBound']
    else: 
        model_2=['ADAM', 'SGD', 'UpperBound']
    line=''
    line_lt=''
    string_full=''
    model_string=''
    for interact in interactions:
        filter_int=data[data['interaction']== interact]
        string_full+= "& \\multicolumn{3}{|c|}{"
        string_full+= f"{interact}"
        string_full+="}"

        #TODO ADD NEW LINW
        save= True
       
        for model in models:
            print(model)
            filter_int_model = filter_int[filter_int['model'].str.contains(model)]
            print(filter_int_model)
            model=model.replace('model', '')
            model=model.replace('_','')
            print(model)
            if interact in ['CAIPI', 'RRR'] and model in model_interact:  
                model_string += f'& {model}'
            elif interact in ['Sanity'] and model in model_2:
                model_string += f'& {model}'
            else: 
                continue
    
            #TODO ADD NEW LINW
    
            for d in datasets:
                filter_int_model_data=filter_int_model[filter_int_model['dataset']==d]
                #TODO Calculation
                #mean= 0
                #std= 0
                print(filter_int_model_data)
                df=filter_int_model_data[~filter_int_model_data['model'].str.contains('LT')].agg({f'{filter}':['mean','std']})
                df2=filter_int_model_data[filter_int_model_data['model'].str.contains('LT')].agg({f'{filter}':['mean','std']})
        
                mean=round(df[filter]['mean'],2)
                std=round(df[filter]['std'],2)
                print(mean)

                mean2=round(df2[filter]['mean'],2)
                std2=round(df2[filter]['std'],2)
                print(mean2)
                        
                line += f'& ${mean} \\pm {std}$ '
                line_lt+=f' &  ${mean2} \\pm {std2}$'
    string_full+=f'\\\\ \hline {model_string} \\\\ \hline '
    string_full += '-'
    string_full += line
    string_full += '\\\\ \\hline '
    string_full += 'LT'
    string_full += line_lt
    string_full += '\\\\ \\hline '

    print(string_full)



    
def plot_taskwise_acc(dir, figsize=(9,3),save = None):
    #df= None
    #linetype=[]
    
    #try:
    #    d_name=data.replace('model_','')
    #except:
    #    d_name=data.replace('model_','')
    #d["Unnamed: 0"]=d["Unnamed: 0"].astype(str)
    
    #if 'LT' in d_name:
    #    d['linetype'] = np.repeat('LT', len(d))
    #    d_name=d_name.replace('_LT','')
    #else: 
    #    d['linetype'] = np.repeat('', len(d))
    #    d['approach']=np.repeat(d_name, len(d))
               
    plt.figure(figsize=figsize)
    #print(df)
    #print(df.columns)
    sns.lineplot(data=dir, x="tasknumber", y=f"value", hue="model",style="LT")
    #fig, ax = plt.subplots()

    #x = np.linspace(0, 2 * np.pi, 50)
    #y = np.sin(x) + np.random.randn(len(x)) * 0.03

    #yerr0 = y - (0.1 + np.random.randn(len(x)) * 0.03)
    #yerr1 = y + (0.1 + np.random.randn(len(x)) * 0.03)

    #ax.plot(x, y, color='C0')
    #plt.fill_between(x, yerr0, yerr1, color='C0', alpha=0.5)
    
    plt.tight_layout()

    if save is None:
        plt.show()
    else:
        plt.savefig(save, transparent=True)



if __name__=='__main__':
    CAIPI_param= 'MeanAndStdTransformer'
    #/media/jacqueline/Data/XIL/Results/Continous/continous_tabular/CAIPI/MeanAndStdTransformer/SimpleMLP/classification_metrics_11.csv
    run_CAIPI=True
    run_rrr=True
    #data='iris_cancer'
    datasets=['task_real']#['domain_real']#['continous_tabular','continous_time']
    
    #data='SimulatedTestingData_Middle_Harmonic_F_1_TS_50'
    #data='toy_classification'
    #Results/decoy_mnist/CAIPI/MeanAndStdTransformer/SimpleMLP
    models=['resnet']#['SimpleMLP']
    #model='CNN'
    full_data_class = None
    full_data_inter = None
    for a in datasets: 
        data=a

        for b in models:
            model=b
            #print(model)
            #print(data)
            #dir_CAIPI, dir_rrr,dir_sanity
            full_data_class, full_data_inter = collect_data(dir_CAIPI=f'/media/jacqueline/Data/XIL/Results/Continous/{data}/CAIPI/{CAIPI_param}/{model}',dir_rrr=f'/media/jacqueline/Data/XIL/Results/Continous/{data}/RRR/{model}',dir_sanity=f'/media/jacqueline/Data/XIL/Results/Continous/{data}/Sanity/{model}/',data=data,model=model, full_data_class=full_data_class, full_data_inter=full_data_inter)
    #print(full_data_class)
    #latex_table(full_data_class, 'f1')
    latex_table_real(full_data_class, 'f1')
    #plot_taskwise_acc(full_data_class,ind='f1', figsize=(9,3),save = './Results/real_itemwise.png')
    #full_data_class = None
    #full_data_inter = None
    #data=collect_data_taskwise(dir_CAIPI=f'/media/jacqueline/Data/XIL/Results/Continous/{data}/CAIPI/{CAIPI_param}/{model}',dir_rrr=f'/media/jacqueline/Data/XIL/Results/Continous/{data}/RRR/{model}',dir_sanity=f'/media/jacqueline/Data/XIL/Results/Continous/{data}/Sanity/{model}/',data=data,model=model, full_data_class=full_data_class, full_data_inter=full_data_inter)
    #plot_taskwise_acc(data, figsize=(9,3),save = None)
    #import sys 