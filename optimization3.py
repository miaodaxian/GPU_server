#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os,sys
import math
from sklearn import linear_model
import pandas as pd
import numpy as np
import array
from scipy.stats import rankdata
import datetime
import pyOpt
print(os.getcwd())


# In[2]:


fannie_lppub_file='/home/k2uxam/data/fannie_lppub.txt'
print(pd.read_csv(fannie_lppub_file,nrows=5,sep="|"))
fannie_lppub=pd.read_csv(fannie_lppub_file,sep="|")
training_start_date='2001-01-01'
training_end_date='2011-12-01'

fannie_lppub.shape[0]
fannie_lppub['AQSN_DTE']=pd.to_datetime(fannie_lppub['FRST_DTE'])


# In[3]:


fannie_lppub_sub=fannie_lppub[(fannie_lppub.AQSN_DTE >= training_start_date) & (fannie_lppub.AQSN_DTE<=training_end_date)]
fannie_lppub_sub.shape[0]
fannie_lppub_sub.NUM_BO.unique()
fannie_lppub_sub['NUM_BO']=np.where(fannie_lppub_sub['NUM_BO']>1,2,1)
fannie_lppub_sub=fannie_lppub_sub[['FICO','OCLTV','DTI','ORIG_AMT','NUM_BO','DLQ60_12','LOAN_ID','FNMA_LN','PURPOSE','ORIG_RT','AQSN_DTE']]
fannie_lppub_sub['CLTV']=fannie_lppub_sub.OCLTV
fannie_lppub_sub=fannie_lppub_sub.dropna()
fannie_lppub_sub.shape
training_data=fannie_lppub_sub


# In[4]:


outSample_start_date='2005-01-01'
#outSample_start_date='2010-01-01'
outSample_end_date='2012-01-01'
fannie_lppub_outSample=fannie_lppub[(fannie_lppub.AQSN_DTE>= outSample_start_date) & (fannie_lppub.AQSN_DTE <= outSample_end_date)]
fannie_lppub_outSample['NUM_BO']=np.where(fannie_lppub_outSample['NUM_BO']>1,2,1)
fannie_lppub_outSample=fannie_lppub_outSample[['FICO','OCLTV','DTI','ORIG_TRM','ORIG_AMT','NUM_BO','DLQ60_12','LOAN_ID','FNMA_LN','PURPOSE','ORIG_RT','AQSN_DTE']]
fannie_lppub_outSample=fannie_lppub_outSample[(fannie_lppub_outSample.OCLTV>=60) & (fannie_lppub_outSample.OCLTV<=97) & (fannie_lppub_outSample.ORIG_TRM<=240)]
fannie_lppub_outSample=fannie_lppub_outSample.dropna()
fannie_lppub_outSample['CLTV']=fannie_lppub_outSample.OCLTV
fannie_lppub_outSample['CLTV_bucket']=np.where(fannie_lppub_outSample['OCLTV']<80,"G1","G2")
testing_data=fannie_lppub_outSample.dropna()
print(training_data.shape)
print(testing_data.shape)


# In[5]:


econHPI_period=[]
econIR_period=[]
econUNEMPLOY_period=[]
econ_data_file='/home/k2uxam/data/all_econ_vars2.csv'
econ_data=pd.read_csv(econ_data_file,sep="|")

if len(econHPI_period)>0 :
    for econ_period in econHPI_period :
        print(econ_period)
        econ_data['CSUSHPINSA_BACK']=np.concatenate((np.repeat(econ_data['CSUSHPINSA'][1],econ_period),np.array(econ_data['CSUSHPINSA'][1:(econ_data.shape[0]+1-econ_period)])),axis=0)
        HPA_LAG="HPA_LAG"+str(econ_period)
        econ_data[HPA_LAG]=(econ_data['CSUSHPINSA']-econ_data['CSUSHPINSA_BACK'])/econ_data['CSUSHPINSA_BACK']
    
if len(econIR_period)>0 :
    for econ_period in econIR_period :
        print(econ_period)
        econ_data['MORTGAGE30US_BACK']=np.concatenate((np.repeat(econ_data['MORTGAGE30US'][1],econ_period),np.array(econ_data['MORTGAGE30US'][1:(econ_data.shape[0]+1-econ_period)])),axis=0)
        ir_LAG="IR_LAG"+str(econ_period)
        econ_data[ir_LAG]=econ_data['MORTGAGE30US_BACK']

if len(econUNEMPLOY_period)>0 :
    for econ_period in econUNEMPLOY_period :
        print(econ_period)
        econ_data['UNRATE_BACK']=np.concatenate((np.repeat(econ_data['UNRATE'][1],econ_period),np.array(econ_data['UNRATE'][1:(econ_data.shape[0]+1-econ_period)])),axis=0)
        unemploy_LAG="UNRATE_LAG"+str(econ_period)
        econ_data[unemploy_LAG]=econ_data['UNRATE_BACK']
        
select_econ=econ_data
select_econ['DATE']=pd.to_datetime(select_econ['DATE'])
training_data=pd.merge(training_data,select_econ,left_on='AQSN_DTE',right_on='DATE',how='left')
testing_data=pd.merge(testing_data,select_econ,left_on="AQSN_DTE",right_on="DATE",how='left')


# In[6]:


training_cohort=['AQSN_DTE']
testing_cohort=['AQSN_DTE','CLTV_bucket']
weight=['ORIG_AMT']

indep_var=['FICO','DTI','CLTV']
target_var=['DLQ60_12']
outputVar=indep_var+target_var

generationSize=30
populationSize=60

lower_bound=[min(min(training_data['FICO']),min(testing_data['FICO'])),min(min(training_data['FICO']),min(testing_data['FICO'])),min(min(training_data['DTI']),min(testing_data['DTI'])),min(min(training_data['DTI']),min(testing_data['DTI'])),min(min(training_data['CLTV']),min(testing_data['CLTV'])),min(min(training_data['CLTV']),min(testing_data['CLTV']))]
upper_bound=[max(max(training_data['FICO']),max(testing_data['FICO'])),max(max(training_data['FICO']),max(testing_data['FICO'])),max(max(training_data['DTI']),max(testing_data['DTI'])),max(max(training_data['DTI']),max(testing_data['DTI'])),max(max(training_data['CLTV']),max(testing_data['CLTV'])),max(max(training_data['CLTV']),max(testing_data['CLTV']))]

inputSize=len(lower_bound)
breakSize=int(inputSize/2)

num_cohort_limit=100
num_in_cohort=500
#obj_fn=4
#outputSize=4
filename="/home/k2uxam/python2/cohort_config_bm.Rdata"


# In[7]:


def mergeSortInversions(arr):
    if len(arr) == 1:
        return arr, 0
    else:
        a = arr[:int(len(arr)/2)]
        b = arr[int(len(arr)/2):]
        a, ai = mergeSortInversions(a)
        b, bi = mergeSortInversions(b)
        c = []
        i = 0
        j = 0
        inversions = 0 + ai + bi
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            c.append(a[i])
            i += 1
        else:
            c.append(b[j])
            j += 1
            inversions += (len(a)-i)
    #print(inversions)
    c += a[i:]
    c += b[j:]
    #print(c)
    return c, inversions


# In[8]:


def kendallTauDistance(x,y) :
    rankY=rankdata(y).astype(int)
    #print("here",rankY)
    orderX=np.array(x).argsort()+1
    #print(orderX)
    inverseInput=orderX[rankY-1]
    #print(inverseInput)
    c,inv= mergeSortInversions(list(inverseInput))
    #print(inv)
    return inv
#x=[4,2,3,8,4,3]
#y=[3,2,3,4,1]
#kendallTauDistance(x,y)


# In[9]:


def attrBucket(cohort_min,cohort_max,i,breaks):
    minV=cohort_min[i]
    maxV=cohort_max[i]
    numVar=int(len(breaks)/2)
    twoBreaks=[breaks[i*2],breaks[i*2+1]]
    if twoBreaks[0]<twoBreaks[1] :
        return [minV,twoBreaks[0],twoBreaks[1],maxV]
    elif twoBreaks[0]==twoBreaks[1]:
        return [minV,twoBreaks[0],maxV]
    else:
        return [twoBreaks[1],twoBreaks[0]]


# In[10]:


def wavg(group, avg_name, weight_name):

    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w*1.0).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


# In[11]:


class cohort_pred:
    def fitness(self,breaks) :
        print("start the pred function")
        num_sttr=int(len(breaks)/2)
        cohort_min=lower_bound[::2]
        cohort_max=upper_bound[::2]
        breakList=[None]*breakSize
        for i in range(0,breakSize) :
            breakList[i]=attrBucket(cohort_min,cohort_max,i,breaks)
        
        for i in range(0,breakSize) :
            breakVector=breakList[i]
            varTemp=indep_var[i]+'_bucket'
            training_data[varTemp]=pd.cut(training_data[indep_var[i]],bins=breakVector)
        print("breaks set up")
    
        training_data_clean=training_data.dropna()
    
        train_dist=999
        train_MSE=999
        train_maxErr=999
        test_dist=999
        test_MSE=999
        test_maxErr=999
            
        if training_data_clean.shape[0] > 0:
            print("enough records")
            bucketVars= [col for col in training_data_clean if 'bucket' in col]
            groupbyVar=training_cohort+bucketVars
            print("outputVar ",outputVar)
            data_fico=training_data_clean.groupby(groupbyVar).apply(wavg,"FICO","ORIG_AMT")
            data_cltv=training_data_clean.groupby(groupbyVar).apply(wavg,"CLTV","ORIG_AMT")
            data_dti=training_data_clean.groupby(groupbyVar).apply(wavg,"DTI","ORIG_AMT")
            data_dlq=training_data_clean.groupby(groupbyVar).apply(wavg,"DLQ60_12","ORIG_AMT")
            cohort_data= pd.DataFrame(data=dict(s1=data_fico, s2=data_dti,s3=data_cltv,s4=data_dlq))
            count=pd.DataFrame(training_data_clean.groupby(groupbyVar).count().iloc[:,1])
            cohort_data['COUNT']=count['OCLTV']
            #cohort_data = ["WA_FICO","WA_DTI","WA_CLTV","WA_DLQ60_12"]
    
            test_fico=testing_data.groupby(testing_cohort).apply(wavg,"FICO","ORIG_AMT")
            test_cltv=testing_data.groupby(testing_cohort).apply(wavg,"CLTV","ORIG_AMT")
            test_dti=testing_data.groupby(testing_cohort).apply(wavg,"DTI","ORIG_AMT")
            test_dlq=testing_data.groupby(testing_cohort).apply(wavg,"DLQ60_12","ORIG_AMT")
            cohort_test= pd.DataFrame(data=dict(s1=test_fico, s2=test_dti,s3=test_cltv,s4=test_dlq))
    
            cohort_data=cohort_data[cohort_data.COUNT>=num_in_cohort]
            #cohort_test = ["WA_FICO","WA_DTI","WA_CLTV","WA_DLQ60_12"]
    
            cohort_data.columns=['WA_FICO','WA_DTI','WA_CLTV','WA_DLQ60_12','COUNT']
            cohort_test.columns=['WA_FICO','WA_DTI','WA_CLTV','WA_DLQ60_12']
    
            estVar='WA_DLQ60_12'
            cohort_data['logit']=np.log(cohort_data[estVar]/(1-cohort_data[estVar]))
            cohort_data=cohort_data.replace([np.inf, -np.inf], np.nan).dropna()
    
            num_training_cohorts=cohort_data.shape[0]
    
            if(num_training_cohorts>=num_cohort_limit) :
                print("here2")
                model=linear_model.LinearRegression()
                model.fit(cohort_data[cohort_data.columns.values[0:3].tolist()],cohort_data[cohort_data.columns.values[3:4].tolist()])
                train_pred=model.predict(cohort_data[cohort_data.columns.values[0:3].tolist()])
                cohort_data['train_pred']=np.exp(train_pred)/(1+np.exp(train_pred))
        
                normFactor=0.5*(cohort_data.shape[0] *(cohort_data.shape[0]-1))
                train_dist=kendallTauDistance((np.array(cohort_data['train_pred']).argsort()+1),(np.array(cohort_data['WA_DLQ60_12']).argsort()+1))/normFactor
                train_MSE=((cohort_data['train_pred']-cohort_data['WA_DLQ60_12'])**2).mean()
                train_maxErr=max(abs(cohort_data['train_pred']-cohort_data['WA_DLQ60_12']))
        
                test_pred=model.predict(cohort_test[cohort_test.columns.values[0:3].tolist()])
                cohort_test['test_pred']=np.exp(test_pred)/(1+np.exp(test_pred))
        
                normFactor=0.5*(cohort_test.shape[0] *(cohort_test.shape[0]-1))
                test_dist=kendallTauDistance((np.array(cohort_test['test_pred']).argsort()+1),(np.array(cohort_test['WA_DLQ60_12']).argsort()+1))/normFactor
                test_MSE=((cohort_test['test_pred']-cohort_test['WA_DLQ60_12'])**2).mean()
                test_maxErr=max(abs(cohort_test['test_pred']-cohort_test['WA_DLQ60_12']))

        print("find optimal values")
        return([train_MSE,test_MSE])
        #return(train_dist,train_MSE,train_maxErr,test_dist,test_MSE,test_maxErr)
    def get_nobj(self):
        return 2
    def get_bounds(self):
        return (lower_bound,upper_bound)
    def get_name(self):
        return("Cohort Optimization")
    


# In[12]:


import pygmo as pg
prob = pg.problem(cohort_pred())


# In[13]:


print(prob)


# In[14]:


#testing
prob.fitness([758, 698,50,50,  52,  52])


# In[ ]:


from datetime import datetime
start_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S') #start


# In[15]:


# create population
pop = pg.population(prob, size=60)
# select algorithm
algo = pg.algorithm(pg.nsga2(gen=5))
# run optimization
pop = algo.evolve(pop)
# extract results
fits, vectors = pop.get_f(), pop.get_x()


# In[16]:



ending_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S') #end


# In[10]:


import pickle


# In[11]:


with open('opt3_results.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([start_time,ending_time,fits,vectors], f)


# In[12]:


with open('opt3_results.pkl') as f:  # Python 3: open(..., 'rb')
    start_t,end_t,fit_results,vector_results = pickle.load(f)


# In[ ]:




