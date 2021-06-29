#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# In[2]:


mcdf = pd.read_csv('nutrition_values.csv', sep=';')
mcdf = mcdf[mcdf['Chain'] == 'Mc Donalds']
mcdf['Item'] = mcdf['Item'].astype(str)
tbdf = pd.read_csv('TacoBell Nutrition - Sheet1 (1).csv')
tbdf['Item'] = tbdf['Item'].map(lambda x: re.sub(r'\W+', ' ', x))
#These are drinks that both chains possess, so I remove them from the analysis
drop_bevs = ['Sprite', 'Coke', 'Cola', 'Hi-C', 'Powerrade', 'Water', 'Sweetener', 'Packet']
for drink in drop_bevs:
    mcdf = mcdf[~mcdf.Item.str.contains(drink)]
mcdf = mcdf[mcdf['Type'] != 'Breakfast']


# In[3]:


conditions = [
    (mcdf['Type'] == 'Sandwiches'),
    (mcdf['Type'] == 'McCafe Smoothies') | (mcdf['Type'] == 'Beverages') | 
    (mcdf['Type'] == 'McCafe Coffees - Nonfat Milk') | (mcdf['Type'] == 'McCafe Coffees - Whole Milk') 
    | (mcdf['Type'] == 'McCafe Frappes'), 
    (mcdf['Type'] == 'French Fries') | (mcdf['Type'] == 'Salads'),
    (mcdf['Type'] == 'Desserts/Shakes')]
choices = ['Entree', 'Drink', 'Side', 'Dessert']
mcdf['Type2'] = np.select(conditions, choices, default='Entree')
for index, row in mcdf.iterrows():
    if 'Sauce' in row['Item'] or 'Honey' in row['Item']:
        mcdf.at[index, 'Type2']= 'Sauce'
    if 'Dressing' in row['Item'] or 'Vinaigrette' in row['Item']:
        mcdf.at[index, 'Type2'] = 'Dressing'
    
mcdf['Type'] = mcdf['Type2']
mcdf = mcdf.drop(['Type2', 'Serving Size (g)', 'Dietary Fiber (g)', 'Chain'], axis=1)
mcdf.columns=tbdf.columns.values
mcdf = mcdf.replace(' -   ',0)
mcdf['Total Fat (g)'] = mcdf['Total Fat (g)'].str.replace(",",".").astype(float)
mcdf['Sat Fat (g)'] = mcdf['Sat Fat (g)'].str.replace(",",".").astype(float)
mcdf['Trans Fat (g)'] = mcdf['Trans Fat (g)'].str.replace(",",".").astype(float)
mcdf = mcdf[~mcdf.Item.str.contains('Butter Garlic Croutons')]
mcdf = mcdf[~mcdf.Item.str.contains('Coffee Cream')]


# In[4]:


tbdf['Cholesterol (mg)'] = tbdf['Cholesterol (mg)'].str.replace('<5', '2.5').astype(float)
tbdf['Cholesterol (mg)'] = tbdf['Cholesterol (mg)'].astype('float64')
tbdf['Sodium (mg)'] = tbdf['Sodium (mg)'].str.replace(',', '').astype(float)
tbdf['Sodium (mg)'] = tbdf['Sodium (mg)'].astype('float64')
tbdf['Sugar(g)'] = tbdf['Sugar(g)'].astype('float64')
mcdf = mcdf.fillna(0)
tbdf = tbdf.fillna(0)
mcdf['Item'].astype(str)
tbdf['Item'].astype(str)


# In[8]:


#Creating different masks for sampling
mcdf_e = mcdf[mcdf['Type'] == 'Entree']
mcdf_s = mcdf[mcdf['Type'] == 'Side']
mcdf_des = mcdf[mcdf['Type'] == 'Dessert']
mcdf_dri = mcdf[mcdf['Type'] == 'Drink']
mcdf_sauce = mcdf[mcdf['Type'] == 'Sauce']
mcdf_dress = mcdf[mcdf['Type'] == 'Dressing']

tbdf_e = tbdf[tbdf['Type'] == 'Entree']
tbdf_s = tbdf[tbdf['Type'] == 'Side']
tbdf_des = tbdf[tbdf['Type'] == 'Dessert']
tbdf_dri = tbdf[tbdf['Type'] == 'Drink']
tbdf_sauce = tbdf[tbdf['Type'] == 'Sauce']


# In[9]:


def getSamples(entreedf, sidedf, dessertdf, drinkdf, df):
    i = 0
    i2 = 0
    #One entree
    entree1 = entreedf.sample()
    if 'McNuggets' in str(entree1['Item']) or 'Strips' in str(entree1['Item']):
            sauce1 = mcdf_sauce.sample()
            sample1 = pd.concat([entree1, sauce1])
    else:
        sample1 = entree1
    if df == 'tbdf':
        sauce1 = tbdf_sauce.sample(2)
        sample1 = pd.concat([entree1, sauce1])
    
    #One entree, one drink
    entree2 = entreedf.sample()
    drink2 = drinkdf.sample()
    if 'McNuggets' in str(entree2['Item']) or 'Strips' in str(entree2['Item']):
            sauce2 = mcdf_sauce.sample()
            sample2 = pd.concat([entree2, drink2, sauce2])
    else:
        sample2 = pd.concat([entree2, drink2])
    if df == 'tbdf':
        sauce2 = tbdf_sauce.sample(2)
        sample2 = pd.concat([entree2, drink2, sauce2])


    #One entree, one drink, one side
    entree3 = entreedf.sample()
    drink3 = drinkdf.sample()
    side3 = sidedf.sample()
    if 'Salad' in str(side3['Item']) or 'salad' in str(side3['Item']):
        dress3 = mcdf_dress.sample()
    if 'McNuggets' in str(entree3['Item']) or 'Strips' in str(entree3['Item']):
            sauce3 = mcdf_sauce.sample()
            if 'dress3' in locals():
                sample3 = pd.concat([entree3, drink3, side3, dress3, sauce3])
            else:
                sample3 = pd.concat([entree3, drink3, side3, sauce3])
    elif 'dress3' in locals():
        sample3 = pd.concat([entree3, drink3, side3, dress3])
    else:
        sample3 = pd.concat([entree3, drink3, side3])
    if df == 'tbdf':
        sauce3 = tbdf_sauce.sample(2)
        sample3 = pd.concat([entree3, drink3, side3, sauce3])
        

    #One entree, one drink, one side, one dessert
    entree4 = entreedf.sample()
    drink4 = drinkdf.sample()
    side4 = sidedf.sample()
    dessert4 = dessertdf.sample()
    
    if 'Salad' in str(side4['Item']) or 'salad' in str(side4['Item']):
        dress4 = mcdf_dress.sample()
    if 'McNuggets' in str(entree4['Item']) or 'Strips' in str(entree4['Item']):
            sauce4 = mcdf_sauce.sample()
            if 'dress4' in locals():
                sample4 = pd.concat([entree4, drink4, side4, dress4, sauce4, dessert4])
            else:
                sample4 = pd.concat([entree4, drink4, side4, sauce4, dessert4])
    elif 'dress4' in locals():
        sample4 = pd.concat([entree4, drink4, side4, dress4, dessert4])
    else:
        sample4 = pd.concat([entree4, drink4, side4, dessert4])
    if df == 'tbdf':
        sauce4 = tbdf_sauce.sample(2)
        sample4 = pd.concat([entree4, drink4, side4, sauce4, dessert4])
    return sample1, sample2, sample3, sample4


# In[135]:


def SampleMcdf(mcdf, n_size: int):
    tempdf = pd.DataFrame(columns = mcdf.columns)
    s1df = pd.DataFrame(columns = mcdf.columns)
    s2df = pd.DataFrame(columns = mcdf.columns)
    s3df = pd.DataFrame(columns = mcdf.columns)
    s4df = pd.DataFrame(columns = mcdf.columns)
    
    calsums = []
    calfatsums = []
    fatsums = []
    satfatsums = []
    transfatsums = []
    cholsums = []
    
    outputdf = tempdf.copy()
    
    for i in range(n_size):
        s1, s2, s3, s4 = getSamples(mcdf_e, mcdf_s, mcdf_des, mcdf_dri, 'mcdf')
        s4_sum = s4.sum()
        
        calsum = s4_sum[2]
        calfatsum = s4_sum[3]
        fatsum = s4_sum[4]
        satfatsum = s4_sum[5]
        transfatsum = s4_sum[6]
        cholsum = s4_sum[7]
        
        calsums.append(calsum)
        calfatsums.append(calfatsum)
        fatsums.append(fatsum)
        satfatsums.append(satfatsum)
        transfatsums.append(transfatsum)
        cholsums.append(cholsum)
        
        tempdf = pd.concat([s1,s2,s3,s4])
        s1df = s1df.append(s1, ignore_index=True)
        s2df = s2df.append(s2, ignore_index=True)
        s3df = s3df.append(s3, ignore_index=True)
        s4df = s4df.append(s4, ignore_index=True)
        outputdf = outputdf.append(tempdf, ignore_index=True)
    return s1df, s2df, s3df, s4df, calsums, calfatsums, fatsums, satfatsums, transfatsums, cholsums, outputdf
        
def SampleTbdf(tbdf, n_size: int):
    
    tempdf = pd.DataFrame(columns = tbdf.columns)
    s1df = pd.DataFrame(columns = tbdf.columns)
    s2df = pd.DataFrame(columns = tbdf.columns)
    s3df = pd.DataFrame(columns = tbdf.columns)
    s4df = pd.DataFrame(columns = tbdf.columns)
    
    calsums = []
    calfatsums = []
    fatsums = []
    satfatsums = []
    transfatsums = []
    cholsums = []
    
    outputdf = tempdf.copy()
    for i in range(n_size):
        s1, s2, s3, s4 = getSamples(tbdf_e, tbdf_s, tbdf_des, tbdf_dri, 'tbdf')
        s4_sum = s4.sum()
        
        calsum = s4_sum[2]
        calfatsum = s4_sum[3]
        fatsum = s4_sum[4]
        satfatsum = s4_sum[5]
        transfatsum = s4_sum[6]
        cholsum = s4_sum[7]
        
        calsums.append(calsum)
        calfatsums.append(calfatsum)
        fatsums.append(fatsum)
        satfatsums.append(satfatsum)
        transfatsums.append(transfatsum)
        cholsums.append(cholsum)
        tempdf = pd.concat([s1,s2,s3,s4])
        s1df = s1df.append(s1, ignore_index=True)
        s2df = s2df.append(s2, ignore_index=True)
        s3df = s3df.append(s3, ignore_index=True)
        s4df = s4df.append(s4, ignore_index=True)
        outputdf = outputdf.append(tempdf, ignore_index=True)
    return s1df, s2df, s3df, s4df, calsums, calfatsums, fatsums, satfatsums, transfatsums, cholsums, outputdf

def plotReady(mc, tb):
    mc['Label'] = 'McDonalds'
    tb['Label'] = 'Taco Bell'
    s = pd.concat([mc, tb])
    s_melt = s.melt(id_vars = 'Label',
                      value_vars = ['Calories',
                             'Calories from Fat',
                             'Sat Fat (g)',
                             'Trans Fat (g)',
                            'Cholesterol (mg)',
                            'Sodium (mg)'],
                      var_name = 'Columns')
    return s_melt

def getMeans(sample):
    x = ['Calories', 'Calories from Fat', 'Sat Fat (g)', 'Trans Fat (g)', 'Cholesterol (mg)', 'Sodium (mg)']
    n = 10000
    means = []
    for i in x:
        temp = (sample[i].sum()) / n
        means.append(temp)
    return means


# In[139]:


mcs1,mcs2,mcs3,mcs4,m_calsums, m_calfatsums, m_fatsums, m_satfatsums, m_transfatsums, m_cholsums, m_mcsample = SampleMcdf(mcdf, 10000)
tbs1,tbs2,tbs3,tbs4,t_calsums, t_calfatsums, t_fatsums, t_satfatsums, t_transfatsums, t_cholsums,tbsample = SampleTbdf(tbdf, 10000)


# In[450]:


fig, axs = plt.subplots(2, 2, figsize=(20,15))
sns.set(font_scale = 1.5)
fig.suptitle('Simulation 4: Meal-by-Meal Snapshot')
axs[0, 0].plot(m_calfatsums[:500],'r', t_calfatsums[:500], 'b')
axs[0, 0].set_title('Calories from Fat per Meal')
axs[0, 1].plot(m_calfatsums[:500],'r', t_calfatsums[:500], 'b')
axs[0, 1].set_title('Cholesterol (mg) per Meal')
axs[1, 0].plot(m_transfatsums[:500],'ro', t_transfatsums[:500], 'bo')
axs[1, 0].set_title('Trans Fats (g) per Meal')
axs[1, 1].plot(m_satfatsums[:500],'r', t_satfatsums[:500], 'b')
axs[1, 1].set_title('Saturated Fats (g) per Meal')
axs[1, 1].set_xlabel('Meal Index')
axs[1, 0].set_xlabel('Meal Index')
axs[0, 0].set_ylabel('Calories')
axs[0, 1].set_ylabel('Cholesterol (mg)')
axs[1, 0].set_ylabel('Trans Fat (g)')
axs[1, 1].set_ylabel('Sat. Fat (g)')
axs[0,0].legend(labels=['McDonalds', 'Taco Bell'])


# In[288]:


s3_melt = plotReady(mcs3, tbs3)


# In[360]:


mentrees = mcs1[mcs1['Type'] == 'Entree']
tentrees = tbs1[tbs1['Type'] == 'Entree']
s1_melt = plotReady(mentrees, tentrees)


# In[364]:


#The average calories, calories from fat, and sodium for a McDonald's entree is far higher than that of Taco Bell's
#Based on the entree-only sample alone, it appears that Taco Bell is the healthier option
plt.figure(figsize=(15,8))
sns.set(color_codes=True)
sns.set(font_scale = 1.2)
b = sns.boxplot(data = s1_melt,
                hue = 'Label', # different colors for different 'cls'
                x = 'Columns',
                y = 'value',
                order = ['Calories', # custom order of boxplots
                         'Calories from Fat',
                        'Cholesterol (mg)',
                        'Sodium (mg)'], palette="Set1")
plt.title('McDonalds vs Taco Bell: Simulation 1', fontsize=14) # You can change the title here
plt.show()


# In[365]:


sns.boxplot(data = s1_melt,
                hue = 'Label', # different colors for different 'cls'
                x = 'Columns',
                y = 'value',
                order = [# custom order of boxplots
                        'Sat Fat (g)', 'Trans Fat (g)'], palette="Set1")


# In[366]:


num = ['Calories', 'Calories from Fat', 'Trans Fat (g)', 'Cholesterol (mg)', 'Sodium (mg)']
mcs1[num] = mcs1[num].apply(pd.to_numeric)
mcs2[num] = mcs2[num].apply(pd.to_numeric)
mcs3[num] = mcs3[num].apply(pd.to_numeric)
mcs4[num] = mcs4[num].apply(pd.to_numeric)

tbs1[num] = tbs1[num].apply(pd.to_numeric)
tbs2[num] = tbs2[num].apply(pd.to_numeric)
tbs3[num] = tbs3[num].apply(pd.to_numeric)
tbs4[num] = tbs4[num].apply(pd.to_numeric)


# In[367]:


mcs1means = getMeans(mcs1)
mcs2means = getMeans(mcs2)
mcs3means = getMeans(mcs3)
mcs4means = getMeans(mcs4)

tbs1means = getMeans(tbs1)
tbs2means = getMeans(tbs2)
tbs3means = getMeans(tbs3)
tbs4means = getMeans(tbs4)


# In[368]:


m_means = pd.DataFrame(mcs1means, index=x)
m_means.rename({0: 'Sim. 1'}, axis=1, inplace=True) 
m_means['Sim. 2'] = mcs2means
m_means['Sim. 3'] = mcs3means
m_means['Sim. 4'] = mcs4means

t_means = pd.DataFrame(tbs1means, index=x)
t_means.rename({0: 'Sim. 1'}, axis=1, inplace=True) 
t_means['Sim. 2'] = tbs2means
t_means['Sim. 3'] = tbs3means
t_means['Sim. 4'] = tbs4means


# In[370]:


fig, axs = plt.subplots(3, 2, figsize=(15,12), sharey='row')
m_means.iloc[[0,1,5]].plot(ax=axs[0,0], kind='bar', rot=0)
axs[0,0].set_title('McDonalds')
t_means.iloc[[0,1,5]].plot(ax=axs[0,1], kind='bar', rot=0)
axs[0,1].set_title('Taco Bell')

m_means.iloc[[2,4]].plot(ax=axs[1,0], kind='bar', rot=0)
t_means.iloc[[2,4]].plot(ax=axs[1,1], kind='bar', rot=0)

m_means.iloc[[3]].plot(ax=axs[2,0], kind='bar', rot=0)
t_means.iloc[[3]].plot(ax=axs[2,1], kind='bar', rot=0)


# In[512]:


fig, axs = plt.subplots(1, 2, figsize=(20,8))
fig.suptitle('Empirical Cumulative Distribution Function (ECDF) Plots')
sns.set(font_scale = 1.3)
sns.ecdfplot(data=m_calfatsums, ax=axs[0], label = 'McDonalds')
sns.ecdfplot(data=t_calfatsums, ax=axs[0], label = 'Taco Bell')
sns.ecdfplot(data=m_cholsums, ax=axs[1])
sns.ecdfplot(data=t_cholsums, ax=axs[1])
fig.legend(loc = 'center right')
axs[0].set_xlabel('Calories from Fat Per Meal')
axs[1].set_xlabel('Cholesterol Per Meal (mg)')


# In[394]:


from scipy.stats import mannwhitneyu
cols = ['Calories', 'Calories from Fat', 'Cholesterol (mg)', 'Sodium (mg)', 'Sat Fat (g)', 'Trans Fat (g)']
for i in cols:
    stat, p = mannwhitneyu(mcs1[i], tbs1[i])
    if p>0.05:
        print('Not significant')


# In[395]:


for i in cols:
    stat, p = mannwhitneyu(mcs2[i], tbs2[i])
    if p>0.05:
        print('Not significant')
for i in cols:
    stat, p = mannwhitneyu(mcs3[i], tbs3[i])
    if p>0.05:
        print('Not significant')
for i in cols:
    stat, p = mannwhitneyu(mcs4[i], tbs4[i])
    if p>0.05:
        print('Not significant')

