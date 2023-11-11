import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import re
import openpyxl
import math
from matplotlib import pyplot as plt
from scipy.stats import skewnorm
from scipy.stats import norm
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.optimize import minimize
import seaborn as sns


class Future():
    def __init__(self, name, begin, expiration) -> None:
        '''
        give sofr future name, begin date and expiration
        '''
        self.name = name
        self.expiration = expiration
        self.begin = datetime.strptime(begin, "%Y-%m-%d")

    def obs_data(self, obsdate, midprice):
        '''
        future contract observation data
        '''
        self.obs_date = obsdate
        self.price = midprice
        self.days = (self.obs_date - self.begin).days



class Option():
    def __init__(self, baseasset_name, option_exp, type, K) -> None:
        # attri_ls = descrip_str.split(' ')
        self.baseassetname = baseasset_name  # attri_ls[0]
        if self.baseassetname == 'SR3':
            self.baseassetfullname = 'SOFR3'
            self.contract_days = 90
        else:
            self.baseassetfullname = 'SOFR1'
            self.contract_days = 30
        self.expiration = pd.to_datetime(option_exp)  # attri_ls[1]
        self.type = type   # attri_ls[3]
        self.K = K   # attri_ls[4]
        

    def obs_data(self, obsdate, bidprice, askprice):
        self.obsdate = obsdate
        self.datestr = datetime.strftime(self.obsdate, "%Y%m%d")
        self.bidprice = bidprice
        self.askprice = askprice
        self.datetomature = (self.expiration - self.obsdate).days


    def get_future(self, futures_df):
        if self.baseassetname == "SR3":
            self.expir_month = math.ceil(self.expiration.month/3) * 3
        else:
            self.expir_month = self.expiration.month

        baseassetdata = futures_df[(futures_df['Product'] == self.baseassetfullname) & (futures_df['month'] == self.expir_month) & (futures_df['year'] == self.expiration.year)].copy()
        baseassetdata = baseassetdata[baseassetdata['Timestamp'] == self.obsdate].iloc[0]
        self.baseasset = Future(baseassetdata.Product, baseassetdata.RefDate, 
                                baseassetdata.Expiration)
        self.baseasset.obs_data(baseassetdata.Timestamp, baseassetdata.MID_PRICE)
        return self.baseasset

    def get_discountrate(self, ratedict, method='simple'):
        
        bucketrate = ratedict[self.datestr]
        self.datetomature = (self.expiration - self.obsdate).days
        
        idx = bucketrate[bucketrate['CumLength'] > self.datetomature].index[0]
        rates = bucketrate.iloc[:idx+1, ].copy()
        # ratesls = rates['DailyRate']
        # daysls = rates['BucketLength']
        if len(rates) > 1:
            rates.iloc[-1, 2] = self.datetomature - rates['CumLength'].iloc[-2]
        else:
            rates.iloc[-1, 2] = self.datetomature
        # print(rates)

        # SIMPLE RATE
        if method == 'simple':
            rate = (rates['DailyRate'] * rates['BucketLength']).sum()/self.datetomature
        # COMPOUNDED RATE
        elif method == 'compounded':
            rate = 1
            if len(rates) == 1:
                rate = pow((1+rates['DailyRate'][0])**rates['BucketLength'][0], 1/self.datetomature) - 1
            else:
                for i in range(len(rates)):
                    rate *= (1+rates['DailyRate'][i])**rates['BucketLength'][i]
                rate = pow(rate, 1/self.datetomature) - 1
        # print(self.datetomature)
        return rate
    
    
    def cal_effectiverate(self, index):
        # realized rates: mean value
        self.realizedrate = index[(index['Effective Date'] <= self.obsdate) 
      & (index['Effective Date'] >= self.baseasset.begin)]['SOFR Index'].mean()
        self.todayrate = index[index['Effective Date'] <= self.obsdate].iloc[-1, ]['SOFR Index']
        
        self.realizedrate = (self.realizedrate - 1) * 100
        self.todayrate = (self.todayrate - 1) * 100


        self.effectiverate = ((100-self.baseasset.price) * self.contract_days - self.baseasset.days * self.realizedrate)/(self.contract_days-self.baseasset.days)

        ##### assume future rate == today rate in all the contract days. 
        self.ft = (self.realizedrate * self.baseasset.days + self.todayrate * (self.contract_days - self.baseasset.days)) / self.contract_days
        return self.todayrate, self.realizedrate, self.effectiverate    # , self.effectiverate
    

    # def cal_FT(self, noFOMC=False):
    #     if noFOMC:
    #         self.ft = (self.realizedrate * self.baseasset.days + self.todayrate * (self.contract_days - self.baseasset.days)) / self.contract_days
        

    def get_ratedistribution(self, feddict):
        
        self.fedrateexp = feddict[self.datestr].copy()

        # need to use future contract expiration date to define the interval
        self.fedrateexp = self.fedrateexp[self.fedrateexp['Date'] <= self.baseasset.expiration]
        self.ft = (self.realizedrate * self.baseasset.days + self.todayrate * (self.contract_days - self.baseasset.days)) / self.contract_days
        
        if len(self.fedrateexp) > 0:
            situationnum = 0
            probs = [1]
            ratechanges = [0]
            exp_date = self.baseasset.expiration
            today = self.obsdate
            exprate = []
            days_intr = []
            for idx, row in self.fedrateexp.iterrows():
                rateprob = row[(row != 0)]
                del rateprob['Date']
                # FOMCdict[row['Date']] = rateprob
                prob = rateprob.values  
                probs = [i * j for i in probs for j in prob]
                # print(probs)
                ratechange = rateprob.index.values
                
                days_intr.append((exp_date - row['Date']).days)  # today
                today = row['Date']
                
                ## calculating the date length
                ## adjust rate change values
                ratechanges = [i + days_intr[-1]/self.contract_days*float(j) for i in ratechanges for j in ratechange]
                # print(ratechanges)
                exprate.append(sum([i*j for i in probs for j in ratechanges]))
            days_intr.append((self.baseasset.expiration - today).days)
            self.ratechanges = ratechanges; self.probs = probs;
            return probs, ratechanges
        else:
            print('no meeting.')
            return None
        
    def cal_sigma(self):
        self.sigma = 0.00025/90*252 * self.datetomature 
        return self.sigma
    
    def cal_payoff(self, method, vol=None, N=100000):
        '''
        assumption: normal distribution
        '''
        if method == 'normal':
            fx = np.zeros(N)
            x = np.linspace(-self.baseasset.price, 100-self.baseasset.price, N)
            len_of_itvl = 100/N
            ST = self.baseasset.price - x
            payoff = [st - self.K if st - self.K> 0 else 0 for st in ST]
            if not vol:
                vol = self.cal_sigma()
            for i in range(len(self.ratechanges)):
                fx += self.probs[i]*norm.pdf(x, loc=self.ratechanges[i]*100, scale=vol)
            payoffexp = sum(payoff*fx*len_of_itvl)
            return payoffexp


    
