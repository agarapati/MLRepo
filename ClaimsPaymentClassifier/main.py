# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:03:48 2020

@author: garapati
"""

#Read the two first two lines of the file.
import csv
import re
#import importlib
import  DynamicRecArray 
import  operator
import  matplotlib.pyplot as plt
import  numpy as np
from sklearn import tree
from sklearn import preprocessing

#importlib.reload(DynamicRecArray)

#with open('C:/Users/garapati/Desktop/SMU_Course_work/ML2/HW2/HW2/Week 5/claim.sample.csv', 'r') as f:
#    print(f.readline())
#    print(f.readline())
    
f = open('C:/Users/garapati/Desktop/SMU_Course_work/ML2/HW2/HW2/Week 5/claim.sample.csv')   
a=[] 
j_code_dict = {}
p_code_dict ={}
types=[('V1', 'S8'),
 ('Claim.Number', 'f8'),
 ('Claim.Line.Number', 'i4'),
 ('Member.ID', 'i4'),
 ('Provider.ID', 'S14'),
 ('Line.Of.Business.ID', 'S6'),
 ('Revenue.Code', 'S6'),
 ('Service.Code', 'S6'),
 ('Place.Of.Service.Code', 'S4'),
 ('Procedure.Code', 'S9'),
 ('Diagnosis.Code', 'S7'),
 ('Claim.Charge.Amount', 'f8'),
 ('Denial.Reason.Code', 'S5'),
 ('Price.Index', 'S3'),
 ('In.Out.Of.ork', 'S3'),
 ('Reference.Index', 'S3'),
 ('Pricing.Index', 'S3'),
 ('Capitation.Index', 'S3'),
 ('Subscriber.Payment.Amount', 'f8'),
 ('Provider.Payment.Amount', 'f8'),
 ('Group.Index', 'i4'),
 ('Subscriber.Index', 'i4'),
 ('Subgroup.Index', 'i4'),
 ('Claim.Type', 'S3'),
 ('Claim.Subscriber.Type', 'S3'),
 ('Claim.Pre.Prince.Index', 'S3'),
 ('Claim.Current.Status', 'S4'),
 ('Network.ID', 'S14'),
 ('Agreement.ID', 'S14')]
csv_f = csv.reader(f)
for row in csv_f:
    if (re.search("^J",row[9])) :
        if (bool(j_code_dict.get(row[9] ))) :
            #j_code_dict[row[9]] =  DynamicRecArray.CodeArray((types),row[9]) ##create the object and append the row
            j_code_dict[row[9]].append(tuple(row))
            
        else:
           # print(row)
            j_code_dict[row[9]] =  DynamicRecArray.CodeArray((types),row[9])
            j_code_dict[row[9]].append(tuple(row))
            
        if (bool(p_code_dict.get(row[4] ))) :
            #j_code_dict[row[9]] =  DynamicRecArray.CodeArray((types),row[9]) ##create the object and append the row
            p_code_dict[row[4]].append(tuple(row))
            
        else:
           # print(row)
            p_code_dict[row[4]] =  DynamicRecArray.ProviderArray((types),row[4])
            p_code_dict[row[4]].append(tuple(row))    
            
            
####we have no created a database of All JCodes .. stored in 202 objects .. Lets answer the questiins
##     Question 1A,1B,1C
def Q1 (j_code_dict):
    code_lines=[]
    in_network_claims=[]
    for key in j_code_dict:
        code_lines.append(j_code_dict[key].length)
        in_network_claims.append(j_code_dict[key].amount_in_network)  
    print ("Number of recored with jcodes=",sum(code_lines))    
    print ("Total InNetwork Claims paid  =",sum(in_network_claims)) 
    for x in sorted(j_code_dict.values(),key=operator.attrgetter('payment_to_providers'),reverse=True)[0:5]:
            print ("The Top five Jcodes based on paytmets to providers is " , x.code,x.payment_to_providers)

####Question 2

def Q2 (p_code_dict):
    data = {"un_paid_claims":[], "paid_claims":[], "Provider":[]}
    for key in p_code_dict:
        data["un_paid_claims"].append(p_code_dict[key].CountofPayments[0])
        data["paid_claims"].append(p_code_dict[key].CountofPayments[1])
        data["Provider"].append(p_code_dict[key].code)
        
    plt.figure(figsize=(20,20))
    plt.title('Scatter Plot', fontsize=20)
    plt.xlabel('uppaid', fontsize=15)
    plt.ylabel('paid', fontsize=15)
    #plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.scatter(data["un_paid_claims"], data["paid_claims"], marker = 'o')

# add labels
    for label, x, y in zip(data["Provider"], data["un_paid_claims"], data["paid_claims"]):
        plt.annotate(label, xy = (x, y))       
        
        
def ArrayMerge(j_code_dict):
     data=np.empty(1, dtype=types)
     for key in j_code_dict:
        data= np.concatenate((data,j_code_dict[key].data))
     return ( data[1:])    
 
def createLabelledData(data):
  #  new_dt = np.dtype(data.dtype.descr + [('LABEL', 'S8')])
   # b = np.zeros(data.shape,new_dt)
    #for col_name in  data.dtype.names:
   #     b[col_name] = data[col_name]
    #b['LABEL']= data['Provider.Payment.Amount'] > 0
  #  i=19 ##remove the provide.payment.amount from the data set
    #name = list(b.dtype.names)
    #new_name = name[:i]+name[i+1:]
    #labelledData = b[new_name]
   # labelledData_name = list(data.dtype.descr)
    data = ArrayMerge(j_code_dict)
    col_to_encode = {'Member.ID': 'int',
                     'Provider.ID':'cat',
                     'Line.Of.Business.ID':'int',
                     'Revenue.Code':'int',
                     'Service.Code':'int',
                     'Place.Of.Service.Code':'cat',
                     'Procedure.Code':'cat',
                     'Diagnosis.Code':'cat',
                     'Claim.Charge.Amount':'int',
                     'Denial.Reason.Code':'cat',
                     'Price.Index':'cat',
                     'In.Out.Of.Network':'cat',
                     'Reference.Index': 'cat',
                     'Pricing.Index':'cat',
                     'Capitation.Index':'cat',
                     'Subscriber.Payment.Amount':'cat',
                     'Provider.Payment.Amount':'binary',
                     'Group.Index':'cat',
                     'Subscriber.Index':'cat',
                     'Subgroup.Index':'cat',
                     'Claim.Type':'cat',
                     'Claim.Subscriber.Type':'cat',
                     'Claim.Pre.Prince.Index':'cat',
                     'Claim.Current.Status':'cat',
                     'Network.ID':'cat',
                     'Agreement.ID':'cat'                
                               }
    for key , value in col_to_encode.items():
        if (value=='cat'):
            le = preprocessing.LabelEncoder()
            x  = le.fit(data[value])
        if (value == 'int'):
            x =  np.data[value].asType=int
        if (value == 'binary'):
            x = np.where(data['value'] > 0, 1 ,0 )
            
    
    
    return(x)    

                     

                     

                     
                     
                     
                     
                     
                     
                     
                     
                     
   
    
    
     
    
    
     
    
        

    
    
    
    
     
     
            


            
        
        