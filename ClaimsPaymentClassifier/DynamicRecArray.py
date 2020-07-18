# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:21:40 2020
code_class
@author: garapati
The below code is copied from 
https://stackoverflow.com/questions/16246643/adding-records-to-a-numpy-record-array
Some methods that were relevent to the project were added by garapati

Tried to use class inheritance to create a seperate table for providerID 
"""
import numpy as np

class CodeArray:
    def __init__(self, dtype,code):
        #self.dtype = np.dtype(dtype)
        self.dtype  = dtype
        self.length = 0
        self.size = 10
        self.code = code
        #self._data = np.empty(dtype=self.dtype,self.size)
        self._data = np.empty(self.size, dtype=self.dtype)

    def __len__(self):
        return self.length

    def append(self, rec):
        if self.length == self.size:
            self.size = int(1.5*self.size)
            self._data = np.resize(self._data, self.size)
        self._data[self.length] = rec
        self.length += 1

    def extend(self, recs):
        for rec in recs:
            self.append(rec)

    @property
    def data(self):
        return self._data[:self.length]
    
    #@property
    #def length(self):
    #    return self.legnth
    @property
    def amount_in_network(self):
        __tempArray = self.data
      #  print(__tempArray)
        __test = 'I'
        __test = __test.encode()
        __tempArray1 =np.core.defchararray.startswith(__tempArray['In.Out.Of.ork'],__test, start=0, end=None)
        return(sum(__tempArray[__tempArray1]['Provider.Payment.Amount']))
     
    @property
    def payment_to_providers(self):
        __tempArray1 = self.data
      #  print(__tempArray)
        #__tempArray1 =np.core.defchararray.startswith(__tempArray['In.Out.Of.ork'],__test, start=0, end=None)
        return(sum(__tempArray1['Provider.Payment.Amount']))
        



class ProviderArray(CodeArray):
    def __init__(self,dtype,code):
        super().__init__(dtype,code)
      
    @property    
    def CountofPayments(self):
        __temp = self.data
        #t =  __temp['Provider.Payment.Amount']
        __zeroPayments = np.count_nonzero(__temp['Provider.Payment.Amount'])
        __allPayments=len(__temp['Provider.Payment.Amount'])
        __nonPayments=__allPayments - __zeroPayments
        return([__zeroPayments,__nonPayments])
        #return t
        
        
        
    
    
        
    
        
                            
        
    