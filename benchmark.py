# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import numpy as np
import pandas
import datetime

from matplotlib import pyplot as plt
from matplotlib import cm as cm
    

dataset = pandas.read_csv("file:///D:/Documents/Master/Data Mining Techniques/GitHub/dataset_mood_smartphone_converter.csv")
b = np.mean(dataset, axis=0)

# start date: 17/02/2014
# end date: 09/06/2014
firstDate = dataset['Day'].min()

def correlation_matrix(df):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    #ax1.grid(True)
    plt.title('Dataset Feature Correlation')
    labels=['mood','arousal','valence','activity','screen','call','sms','builtin','comm', 'lol', 'money', 'game', 'work', 'other', 'social','travel', 'unknown', 'util', 'weather',]
    #ax1.set_xticklabels(labels,fontsize=6)
    #ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.,.2,.4,.6,.8,1.])
    plt.axis('off')
    plt.savefig('dataset_corr.png')
    
#correlation_matrix(dataset)

#xyz = dataset.corr()
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

#print("Top Absolute Correlations")
#print(get_top_abs_correlations(dataset, 30))

#Given a date/timestamp t get the latest non-zero mood
# returns the mood and the date/timestamp the mood came from
def getMood(t, ids=-1):
    
    #Break recursion, because there will be no valid value before t=0
    if t < 0:
        return -1,-1
    
    #If no ids is given, return the first occurance
    if ids == -1:
        mood = avg_dataset.loc[avg_dataset['Day']==getDateStringFromNumber(t)]['mood'].iloc[0]
    else:
        mood = dataset.loc[(dataset['Day']==getDateStringFromNumber(t)) 
                        & (dataset['Ids']==ids)]['mood'].iloc[0]
        
    if(mood == 0):
        return getMood(t-1, ids)
    return mood, t

def getNumberFromDateString(date):
    date_dt = datetime.datetime.strptime(date, "%Y-%m-%d")
    firstDate_dt = datetime.datetime.strptime(firstDate, "%Y-%m-%d")
    return (date_dt-firstDate_dt).days
    
def getDateStringFromNumber(t):
    return (datetime.datetime.strptime(firstDate, "%Y-%m-%d")+datetime.timedelta(days=t)).strftime("%Y-%m-%d")

def averageDataFramePerDay(df):
    #Average all values per day
    avg = df.groupby('Day', as_index = False).mean()
    
    #Average all values where mood was actually scored per day
    clean_avg = df[df['mood'] != 0].groupby('Day', as_index = False).mean()
    
    i = 0
    for j in range(avg.shape[0]):
        if avg['mood'].ix[j] != 0:
            avg['mood'].ix[j] = clean_avg['mood'].ix[i]
            avg['circumplex.arousal'].ix[j] = clean_avg['circumplex.arousal'].ix[i]
            avg['circumplex.valence'].ix[j] = clean_avg['circumplex.valence'].ix[i]
            i+=1
        
    return avg

avg_dataset = averageDataFramePerDay(dataset)

def testBaseline(targetId=-1, testId=-1, debug=False):
    total = 0
    correct = 0
    correctRound = 0
    missing = 0
    total_dist = 0
    MSE_dist = 0
    MSE = 0
    
    for t in range(1, avg_dataset.shape[0]):
        #Check if at index t there is actually data available
        if t == getMood(t, targetId)[1]:
            
            if targetId == -1:
                mood1,_ = getMood(t)
            else:
                mood1,_ = getMood(t, targetId)
                
            if testId == -1:
                mood2,_ = getMood(t-1)
            else:
                mood2,_ = getMood(t-1, testId)
            
            total+=1
            
            if mood1 == -1 or mood2 == -1:
                missing +=1
            else: 
                if mood1 == mood2:
                    correct +=1
                if round(mood1) == round(mood2):
                    correctRound +=1
                
                total_dist += abs(mood1-mood2)
                MSE += (mood1-mood2)**2
    
    MSE_dist += MSE
    MSE = MSE/(total-missing)
    if debug:
        print('\n-- Baseline targetId: ', targetId, 'testId: ', testId, ' --')
        #print('Total: ', total, '\nCorrect: ', correct, '\nCorrect (round): ', correctRound, '\nMissing: ', missing, '\nTotal distance: ', total_dist, '\nError: ', (total_dist/(total-missing)), 'MSE: ', MSE)        
        print(' ME: ', (total_dist/(total-missing)), '\nMSE: ', MSE)
    return total, correct, correctRound, missing, total_dist, (total_dist/(total-missing)), MSE, MSE_dist

def testBaselineAll(debug=False):
    print('--- Running Baseline on all ---')
    
    total_correct = 0
    total_correctRound = 0
    total_dist = 0
    total_MSE = 0
    total_MSE_dist = 0
    total_count = 0
    total_missing = 0
    
    MSEs = []
    MSEs2 = []
    idx = 0
    
    #Average model for specific target Id
    for targetId in dataset['Ids'].unique():
        res = testBaseline(targetId, -1, debug)
        total_correct += res[1]
        total_correctRound += res[2]
        total_dist += res[4]
        total_MSE += res[6]
        MSEs.append(res[6])
        total_count += (res[0]-res[3])
        total_missing += res[3]
        total_MSE_dist += res[7]
    
    
    print('\n- Average model for individual target Ids -')
    print('Correct: ', total_correct, '\nCorrect round: ', total_correctRound,
              '\nTotal distance: ', total_dist, '\nCount: ', total_count, '\nAverage distance error: ', 
              (total_dist/(total_count)), '\nAverage MSE: ', 
              (total_MSE / dataset['Ids'].unique().shape[0]))
    print('total_MSE_dist: ', total_MSE_dist)
        
    total_correct = 0
    total_correctRound = 0
    total_dist = 0
    total_MSE = 0
    total_MSE_dist = 0
    total_count = 0
    total_missing = 0    
        
    idx = 0
    
    #Target Id model for specific target Id
    for targetId in dataset['Ids'].unique():
        res = testBaseline(targetId, targetId, debug)
        total_correct += res[1]
        total_correctRound += res[2]
        total_dist += res[4]
        total_MSE += res[6]
        MSEs2.append(res[6])
        total_count += (res[0]-res[3])
        total_missing += res[3]
        total_MSE_dist += res[7]

    print('\n- target Id model for individual target Ids -')
    print('Correct: ', total_correct, '\nCorrect round: ', total_correctRound,
              '\nTotal distance: ', total_dist, '\nCount: ', total_count, '\nAverage distance error: ', 
              (total_dist/(total_count)), '\nAverage MSE: ', 
              (total_MSE / dataset['Ids'].unique().shape[0]))
    print('total_MSE_dist: ', total_MSE_dist)
    
    return MSEs,MSEs2

def plotHist(n_bins):    
    clean_moods = round(dataset[dataset['mood'] != 0])
    
    n, bins, patches = plt.hist(clean_moods['mood'], n_bins, range=(0,10), width=0.95, align='left', normed=True)
    
    plt.text(2, .25, r'$\mu=6.99,\ \sigma=0.74$')
    
    plt.xticks(range(1,11))
    plt.axis([1,10,0,0.6])    
    plt.gca().yaxis.grid()
    
    plt.xlabel('Mood')
    plt.ylabel('Occurance rate')
    
    plt.savefig('dataset_mood_freq_text.png')
    
def plotIndividualMoods():
    clean_moods = (dataset[dataset['mood'] != 0])
    
    mins = clean_moods.groupby('Ids', as_index=False).min()['mood']
    means = clean_moods.groupby('Ids', as_index=False).mean()['mood']
    maxs = clean_moods.groupby('Ids', as_index=False).max()['mood']
    std = clean_moods.groupby('Ids', as_index=True).std()['mood']
    
    plt.errorbar(np.arange(27), means, std, fmt='ok', lw=3)
    plt.errorbar(np.arange(27), means, [means - mins, maxs - means],
             fmt='.k', ecolor='gray', lw=1)
    
    plt.xlabel('Individual #')
    plt.ylabel('Mood')
    
    plt.savefig('dataset_mood_error.png')