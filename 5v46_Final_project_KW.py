# -*- coding: utf-8 -*-
'''
Psych 5V46 Final Project
Instructor: Dr. Stephen Emrich

This script contains functions that generate data that mocks a rat breeding.

This is followed by syntax to generate mock behavioral and biomarker data relevant to my own thesis project.

Next, there is a function that conducts independent T-tests on an variable between Ethanols and Controls from the Final dataset

Lastly, there is syntax to visualize the behavioral and biomarker data, as well as Dendrograms and scatter plots to identify and visualize clusters in the dataset.
                                                                               
                                                                               
'''
import numpy as np

import pandas as pd

import random

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import scipy.cluster.hierarchy as sch 

from sklearn.cluster import AgglomerativeClustering as ac

from sklearn.preprocessing import normalize


#                                                                              Breeding

def Real_breed():

    print("Please enter the number of Dams:")

    Input1=input()

    Dam_count = int(Input1)

    print("Please enter the file name followed with.csv (e.g Data.csv)")

    Filename=input()
    
    Treatment_D = ['Control', 'Ethanol']

    Sex_D = ['M','F']

    df = pd.DataFrame(columns=['Treatment', 'Dam', 'Pups_sex', 'Pups_rank', 'Tested'])
        
    
    for i in range(Dam_count):

        Treatment=random.choice(list(Treatment_D))

        Pups_num=round(random.uniform(1,12))
        
        Tested=0

        for ii in range(Pups_num):
 
            new_row = pd.Series({'Treatment': Treatment, 'Dam': i, 'Pups_sex': random.choice(Sex_D), 'Pups_rank': (ii+1), 'Tested':Tested})
            
            df = df.append(new_row, ignore_index=True)
    
    df.to_csv(Filename)

#Next is the weaning stage where the first male and female from each dam is selcted to proceed to testing

#I spent too much time struggling with building the syntax to do realistic scenario of 'the wean' so I modified the breeding...

# in the syntax below to give me exactly 1 male and 1 female. 
   

# Modified Breeding to produce exactly 1 male and 1 female from each Dam. The following portions of the script will be based on the Modified Breeding. 

#If you're testing out the function, the number you want to enter is 100 or more.

#%reset #To reset the variables, please type this '%reset' in the Console and press enter. Type the letter "y" and press enter.


def Mod_breed():

    print("Please enter the number of Dams:")

    Input1=input()

    Dam_count = int(Input1)

    print("Please enter the file name followed with.csv (e.g Data.csv)")

    Filename=input()
    
    Treatment_D = ['Control', 'Ethanol']

    df = pd.DataFrame(columns=['Treatment', 'Dam', 'Pups_sex', 'Tested'])
         
    for i in range(Dam_count):
    
        Treatment=random.choice(list(Treatment_D))
        
        Pups_num=1
        
        Tested=1
        
        for ii in range(Pups_num):
            
                new_rowM = pd.Series({'Treatment': Treatment, 'Dam': i, 'Pups_sex': "M",'Tested':Tested})
        
                df = df.append(new_rowM, ignore_index=True)
                
                new_rowF = pd.Series({'Treatment': Treatment, 'Dam': i, 'Pups_sex': "F", 'Tested':Tested})
                
                df = df.append(new_rowF, ignore_index=True)
                
    df.to_csv(Filename)

#End of breeding/weaning   



#                                                                              Behavior testing

# Below is the code to generate mock behavior data

df=pd.read_csv('breeding.csv', usecols=['Unnamed: 0', 'Treatment', 'Dam', 'Pups_sex'])

df=df.rename(columns={"Unnamed: 0": "Index"})

#Elevated plus maze is a 10 min test for anxiety behaviors

#creating mock EPM_open_entry data

Min_EPM_Open_entry=0

df['EPM_Open_entry'] = [ round(np.random.normal(loc=7, scale = 3,size=None)) if i=='Ethanol' else

                         round(np.random.normal(loc=3, scale = 2,size=None)) for i in df.Treatment]

df['EPM_Open_entry'] = [ Min_EPM_Open_entry if i<Min_EPM_Open_entry else i for i in df.EPM_Open_entry ]

#creating mock EPM_open_time data; 10 minutes in the maze

Max_EPM_Open_time=10

Min_EPM_Open_time=0

df['EPM_Open_time'] = [ (np.random.normal(loc=6, scale = 2.5,size=None)) if i=='Ethanol' else

                        (np.random.normal(loc=4, scale = 2,size=None)) for i in df.Treatment]

df['EPM_Open_time'] = [ Max_EPM_Open_time if i>Max_EPM_Open_time else i for i in df.EPM_Open_time ]

df['EPM_Open_time'] = [ Min_EPM_Open_time if i<Min_EPM_Open_time else i for i in df.EPM_Open_time ]

#Forced swim test is a 5 minute test for depressive behaviors

#creating mock FST_first_freeze data; 

Min_FST_first_freeze=0

df['FST_first_freeze'] = [ (np.random.normal(loc=1, scale = 0.5,size=None)) if i=='Ethanol' else

                           (np.random.normal(loc=2.5, scale = 0.5,size=None)) for i in df.Treatment]

df['FST_first_freeze'] = [ Min_FST_first_freeze if i<Min_FST_first_freeze else i for i in df.FST_first_freeze ]

#creating mock FST_freeze_time; 

#total immobolized in the water

Max_FST_freeze_time=5

Min_FST_freeze_time=0

df['FST_freeze_time'] = [ (np.random.normal(loc=3, scale = 0.75,size=None)) if i=='Ethanol' else

                          (np.random.normal(loc=2, scale = 0.5,size=None)) for i in df.Treatment]

df['FST_freeze_time'] = [ Max_FST_freeze_time if i>Max_FST_freeze_time else i for i in df.FST_freeze_time ]

df['FST_freeze_time'] = [ Min_FST_freeze_time if i<Min_FST_freeze_time else i for i in df.FST_freeze_time ]

#standardization to create emotionality score 

# To standardize, subtract observed value by the mean and divide by the standard deviation

EPM_Open_entry_mean=df.EPM_Open_entry.mean() 

EPM_Open_entry_std=np.std(df.EPM_Open_entry) 

EPM_Open_time_mean=df.EPM_Open_time.mean() 

EPM_Open_time_std=np.std(df.EPM_Open_time) 

FST_first_freeze_mean=df.FST_first_freeze.mean() 

FST_first_freeze_std=np.std(df.FST_first_freeze) 

FST_freeze_time_mean=df.FST_freeze_time.mean() 

FST_freeze_time_std=np.std(df.FST_freeze_time) 

#standardization

df['EPM_Open_entry_stzd'] = [ ((i-EPM_Open_entry_mean)/EPM_Open_entry_std) for i in df.EPM_Open_entry]

df['EPM_Open_time_stzd'] = [ (((i-EPM_Open_time_mean)/EPM_Open_time_std)*-1) for i in df.EPM_Open_time] #reverse coded, such as higher time spent in open arms will add to emotionality score

df['FST_first_freeze_stzd'] = [ (((i-FST_first_freeze_mean)/FST_first_freeze_std)*-1) for i in df.FST_first_freeze] #reverse coded, such that less time of immobility will add to emotionality score

df['FST_freeze_time_stzd'] = [ (((i-FST_freeze_time_mean)/FST_freeze_time_std)) for i in df.FST_freeze_time] 

#Compile standardized scores from behavioural battery to create emotionality score

#Ethanol treatments should have higher emotionality scores compared to controls

df['Emotionality_score']= [(df.EPM_Open_entry_stzd[i] + df.EPM_Open_time_stzd[i] + df.FST_first_freeze_stzd[i] + df.FST_freeze_time_stzd[i]) for i in df.Index]

#end of  generating mock behavioral data

#                                                                              Assaying inflammatory biomarker data 

#The distributions of the data were based on distributions from an older project

df['IFNgamma']= [random.randint(0, 44) for i in df.Index]

df['IL6'] = [ (np.random.normal(loc=0.85, scale = 1.3,size=None)) if i=='Ethanol' else

              (np.random.normal(loc=0.75, scale = 1.3,size=None)) for i in df.Treatment]

df['IL6'] = [ i*-1 if i<0 else i for i in df.IL6 ]

df['IL10'] = [ (np.random.normal(loc=7, scale = 7,size=None)) if i=='Ethanol' else

               (np.random.normal(loc=6, scale = 6,size=None)) for i in df.Treatment]

df['IL10'] = [ i*-1 if i<0 else i for i in df.IL10 ]

#end of generating mock biomarker data

# Save all the mock data to prepare for data visualization and analysis

df.to_csv('Final_dataset.csv')
     
#                                                                              Data analysis

# The ttest function conducts independent T-test of any dependent variables between the Ethanol and Control treatment group

df=pd.read_csv('Final_dataset.csv', usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

def ttest():

    print('''Please enter the name of the variable you want to compare between Ethanols and Controls, \n If you want to test more than one variable, please seperate variable name by a comma \n e.g EPM_Open_entry,EPM_Open_time,FST_freeze_time,FST_first_freeze''')

    Input1 =input() 

    print("Please enter the alpha e.g 0.05")

    Input2=input()

    Var_list= Input1.split(',')

    sig=float(Input2)

    results = [] 

    for i, e in enumerate( Var_list):
            
        E=df[df['Treatment']=='Ethanol'][e]

        C=df[df['Treatment']=='Control'][e]
        
        t, p = stats.ttest_ind(E, C)

        if p < sig:

            p= ('<'+ str(sig))
            
        results.append(f'Test# {i+1} Treatment vs Control:Independent variable({e}) T-value: {t.round(2)}, P-value: {p}')

    print(results)

#                                                                              Data visualization

#Behaviour visualization

fig, axes = plt.subplots(2, 2)

sns.barplot(ax=axes[0,0], x=df.Treatment, y=df.EPM_Open_entry, data=df, ci='sd').title.set_text('Open entry:PAE vs Control')

sns.barplot(ax=axes[0,1], x=df.Treatment, y=df.EPM_Open_time, data=df, ci='sd').title.set_text('Open arm time:PAE vs Control')

sns.barplot(ax=axes[1,0], x=df.Treatment, y=df.FST_first_freeze, data=df, ci='sd').title.set_text('First freeze:PAE vs Control')

sns.barplot(ax=axes[1,1], x=df.Treatment, y=df.FST_freeze_time, data=df, ci='sd').title.set_text('Freeze time:PAE vs Control')

plt.tight_layout()


#Immune biomarker visualization

fig, axes = plt.subplots(1,3)

sns.scatterplot(ax=axes[0], x=df.Emotionality_score, y=df.IFNgamma, data=df).title.set_text('IFNγ:PAE vs Control')

sns.scatterplot(ax=axes[1], x=df.Emotionality_score, y=df.IL6, data=df).title.set_text('IL-6:PAE vs Control')

sns.scatterplot(ax=axes[2], x=df.Emotionality_score, y=df.IL10, data=df).title.set_text('IL-10:PAE vs Control')

plt.tight_layout()


#Emotionality score scatter

#Plot 1

sns.set_style(style="white") 

sns.catplot(x=df.Treatment, y=df.Emotionality_score, data=df)

#Plot 2

df=pd.read_csv('Final_dataset.csv', usecols=[2,13])

df['Treatment'] = [1 if i=='Control' else 2 for i in df.Treatment] #1 = control, 2 = ethanol

cluster = ac(n_clusters=2, affinity='euclidean', linkage='ward') #Applying parameters for scatterplot 

cluster.fit_predict(df) # 0's are for first cluster, 1's is for second cluster

plt.figure(figsize=(10, 7))  

plt.scatter(df['Treatment'], df['Emotionality_score'], c=cluster.labels_) # The purple/black indicates those at risk while the yellow/white indicates those that are resilient


#what I wanted to add onto the catplot but couldn't because catplot/stripplot wont accept the axes.

plt.title('Risk vs Resilience')

plt.axhline(y=0, color='k', linestyle='-')

_ = plt.plot([0, 0], color="k")

plt.ylim(-6,8)

plt.show()


#data preparation for clustering
df=pd.read_csv('Final_dataset.csv', usecols=[2,4,5,6,7,8,13,14,15,16])

df['Treatment'] = [1 if i=='Control' else 2 for i in df.Treatment] #1 = control, 2 = ethanol

df['Pups_sex'] = [1 if i=='M' else 2 for i in df.Pups_sex] #1=male, 2=ethanol

data_scaled= normalize(df)

data_scaled = pd.DataFrame(data_scaled, columns=df.columns)


#Dendrogram with heatmap using Scipy

plt.figure(figsize=(25, 12))  

plt.title("Dendrograms") 

dendrogram = sch.dendrogram(sch.linkage(data_scaled, method='ward'))

plt.axhline(y=3, color='b', linestyle='--') #added line show 2 clusters based on longest vertical line 


#Dendrogram with heatmap using seaborn

sns.clustermap(data_scaled, cmap="mako")

plt.show()


#visualizing clusters using a scatterplot

cluster = ac(n_clusters=2, affinity='euclidean', linkage='ward') #Applying parameters for scatterplot 

cluster.fit_predict(data_scaled) # 0's are for first cluster, 1's is for second cluster

plt.figure(figsize=(10, 7))

plt.scatter(data_scaled['IL10'], data_scaled['Emotionality_score'], c=cluster.labels_) 


#end