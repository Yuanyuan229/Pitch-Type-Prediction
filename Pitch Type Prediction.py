#!/usr/bin/env python
# coding: utf-8

# # Pitch Prediction


# **Import Packages**

# In[1]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import collections

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import export_graphviz

#for some statistics
from scipy import stats
from scipy.stats import norm, skew 

# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import  Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import tree

#clear interface
import warnings
warnings.filterwarnings('ignore')

#Limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))


# **Import data**

# In[2]:


batters_df = pd.read_csv('Batter_info.csv')
pitches_df = pd.read_csv('Pitches.csv')
pitcher_info_df = pd.read_csv('Pitcher_info.csv')
pitching_stats_df = pd.read_csv('Pitching_stats.csv')
teams_df = pd.read_csv('Teams.csv')


# **preview the data**

# In[3]:


print('Batters.csv',batters_df.shape)
print('Pitches.csv',pitches_df.shape)
print('Pitching_stats.csv',pitching_stats_df.shape)
print('teams.csv',teams_df.shape)


# In[10]:


batters_df.drop_duplicates(subset='batter_name',keep = 'last', inplace = True)


# In[11]:


batters_df.head()


# In[12]:


pitches_df.drop_duplicates()


# In[13]:


pitches_df.head()


# In[14]:


pitching_stats_df.head()


# In[15]:


teams_df.head()


# **join the tables**

# In[47]:


p_match = pd.merge(pitches_df,pitching_stats_df, how='inner', on='pitcher_id',
         left_index=False, right_index=False, sort=True,
         suffixes=('_', '_1'), copy=True, indicator=False,
         validate=None)


# In[48]:


p_t_match = pd.merge(p_match, teams_df, how='inner', on='team_id',
         left_index=False, right_index=False, sort=True,
         suffixes=('_', '_1'), copy=True, indicator=False,
         validate=None)


# In[49]:


all_data = pd.merge(p_t_match, batters_df, how='inner', on='batter_id',
         left_index=False, right_index=False, sort=True,
         suffixes=('_pitcher', '_batter'), copy=True, indicator=False,
         validate=None)


# In[50]:


all_data = pd.merge(all_data, teams_df, how='inner', left_on='team_id_batter',right_on='team_id',
         left_index=False, right_index=False, sort=True,
         suffixes=('_pitcher', '_batter'), copy=True, indicator=False,
         validate=None)


# In[16]:


all_data.shape


# **clean data**

# In[17]:


# missing value check
missing_ratio = all_data.isnull().sum()/len(all_data)*100
show=missing_ratio[missing_ratio!=0].sort_values(ascending=False)
print(len(show))
show


# In[20]:


# drop rows with na
all_data = all_data.dropna()
all_data.shape


# In[72]:


#drop id and dulplicate columns
all_data = all_data.drop(['pitcher_id', 'batter_id', 'g_id', 'Stats_id',
       'pitcher_name', 'team_id_pitcher', 'era', 'year_pitcher',
       'team_id_batter', 'year_batter', 'team_id'],axis=1)


# **take a look at the descriptive info**

# In[22]:


all_data.describe().T


# In[23]:


all_data.describe(include=['O']).T


# In[75]:


all_data['pitch_type'].value_counts(dropna=False)[:]


# **Run visualizations of the target variable**

# In[91]:


#plot: pitch type counts
c = collections.Counter(all_data['pitch_type'])
c = sorted(c.items())
pitch_type = [i[0] for i in c]
freq = [i[1] for i in c]

f, ax = plt.subplots()

plt.bar(pitch_type, freq)
plt.title("Count per pitch type")
plt.xlabel("pitch type")
plt.ylabel("Frequency")
ax.set_xticks(range(1, 20))
ax.set_xticklabels(pitch_type)

plt.show()


# **change the target variable 'pitch type' into 'pitch class'** (leverage domain knowledge and make the dataset more balanced)

# fastballs
# 'FF' : four-seam fastball
# 'FT' : two-seam fastball
# 'FC' : fastball (cutter)
# ’FA’ :	Four-Seam Fastball (FA)
# 'FS' :	Splitter (FS)
# 
# Breaking Balls:
# CU:Curveball (CU)
# SL:Slider (SL)
# SC:Screwball (SC)
# KC:Knuckle-curve (KC)
# 
# Change Ups:
# CH:changeup
# 
# Knuckleball:
# KB

# In[51]:


all_data['pitch_class']='other'
#merging levels of pitch type to make fewer class that will hold more meaning. 
#Fastballs
all_data['pitch_class']=np.where(all_data['pitch_type'] =='FF','Fastballs', all_data['pitch_class'])
all_data['pitch_class']=np.where(all_data['pitch_type'] =='FT','Fastballs', all_data['pitch_class'])
all_data['pitch_class']=np.where(all_data['pitch_type'] =='FC','Fastballs', all_data['pitch_class'])
all_data['pitch_class']=np.where(all_data['pitch_type'] =='FS','Fastballs', all_data['pitch_class'])
all_data['pitch_class']=np.where(all_data['pitch_type'] =='FA','Fastballs', all_data['pitch_class'])


# In[52]:


#Breaking Balls
all_data['pitch_class']=np.where(all_data['pitch_type'] =='CU', 'Breaking Balls',all_data['pitch_class'])
all_data['pitch_class']=np.where(all_data['pitch_type'] =='SL', 'Breaking Balls',all_data['pitch_class'])
all_data['pitch_class']=np.where(all_data['pitch_type'] =='SC', 'Breaking Balls',all_data['pitch_class'])
all_data['pitch_class']=np.where(all_data['pitch_type'] =='KC', 'Breaking Balls',all_data['pitch_class'])


# In[53]:


#Change Ups
all_data['pitch_class']=np.where(all_data['pitch_type'] =='CH', 'Change Ups',all_data['pitch_class'])
#Knuckleball
all_data['pitch_class']=np.where(all_data['pitch_type'] =='KB', 'Change Ups',all_data['pitch_class'])


# In[54]:


#drop the other rows with pitch types since they are not real 'pitch types'
all_data.drop(all_data[all_data['pitch_class']=='other'].index, axis=0,inplace=True)


# In[73]:


all_data


# In[59]:


#plot: pitch class counts
c = collections.Counter(all_data['pitch_class'])
c = sorted(c.items())
pitch_class = [i[0] for i in c]
freq = [i[1] for i in c]

f, ax = plt.subplots()

plt.bar(pitch_class, freq)
plt.title("Count per pitch class")
plt.xlabel("pitch class")
plt.ylabel("Frequency")
ax.set_xticks(range(0, 3))
ax.set_xticklabels(pitch_class)

plt.show()


# **Model Development**

# **Model 1-All batters against all pitchers**

# In[75]:


y = all_data['pitch_class']
features = all_data.drop(['pitch_type','pitch_class'],axis=1)


# In[77]:


features = features.drop(['team_abv_pitcher', 'team_name_pitcher', 'league_pitcher',
       'region_pitcher', 'batter_name', 'team_abv_batter', 'team_name_batter',
       'league_batter', 'region_batter'],axis=1)
#get dummies
features = pd.get_dummies(features)
print(features.shape)


# In[78]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size = 0.25, random_state = 42)


# In[79]:


rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, y_train)
rfc.pred = rfc.predict(X_test)


# In[80]:


print(classification_report(y_test,rfc.pred))


# In[81]:


class_names = y.unique()

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(rfc, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)


# **Model 2-National League pitchers v. American League batters**

# In[82]:


#filter the dataset
data2 = all_data[all_data['league_pitcher']=='National']
data2 = data2[data2['league_batter']=='American']


# In[83]:


y2 = data2['pitch_class']
features2 = data2.drop(['pitch_type','pitch_class','team_abv_pitcher', 'team_name_pitcher', 'league_pitcher',
       'region_pitcher', 'batter_name', 'team_abv_batter', 'team_name_batter',
       'league_batter', 'region_batter'],axis=1)
features2 = pd.get_dummies(features2)
print(features2.shape)


# In[84]:


# Split the data into training and testing sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(features2, y2, test_size = 0.25, random_state = 42)


# In[85]:


rfc2 = RandomForestClassifier(n_estimators = 100)
rfc2.fit(X_train2, y_train2)
rfc2.pred = rfc2.predict(X_test2)


# In[86]:


print(classification_report(y_test2,rfc2.pred))


# In[145]:


class_names = y2.unique()

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(rfc2, X_test2, y_test2,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)


# **Model 3-National League batters v. American League pitchers**

# In[87]:


#filter the dataset
data3 = all_data[all_data['league_pitcher']== 'American']
data3 = data3[data3['league_batter']=='National']


# In[88]:


y3 = data3['pitch_class']
features3 = data3.drop(['pitch_type','pitch_class','team_abv_pitcher', 'team_name_pitcher', 'league_pitcher',
       'region_pitcher', 'batter_name', 'team_abv_batter', 'team_name_batter',
       'league_batter', 'region_batter'],axis=1)
features3 = pd.get_dummies(features3)
print(features3.shape)


# In[89]:


# Split the data into training and testing sets
X_train3, X_test3, y_train3, y_test3 = train_test_split(features3, y3, test_size = 0.25, random_state = 42)


# In[90]:


rfc3 = RandomForestClassifier(n_estimators = 100)
rfc3.fit(X_train3, y_train3)
rfc3.pred = rfc3.predict(X_test3)


# In[91]:


print(classification_report(y_test3,rfc3.pred))


# In[146]:


class_names = y3.unique()

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(rfc3, X_test3, y_test3,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)


# **Model 4-All Pitchers v.  Astro Batters**

# In[92]:


#filter the dataset
data4 = all_data[all_data['team_abv_batter']=='HOU']


# In[93]:


y4 = data4['pitch_class']
features4 = data4.drop(['pitch_type','pitch_class','team_abv_pitcher', 'team_name_pitcher', 'league_pitcher',
       'region_pitcher', 'batter_name', 'team_abv_batter', 'team_name_batter',
       'league_batter', 'region_batter'],axis=1)
features4 = pd.get_dummies(features4)
print(features4.shape)


# In[94]:


# Split the data into training and testing sets
X_train4, X_test4, y_train4, y_test4 = train_test_split(features4, y4, test_size = 0.25, random_state = 42)


# In[95]:


rfc4 = RandomForestClassifier(n_estimators = 100)
rfc4.fit(X_train4, y_train4)
rfc4.pred = rfc4.predict(X_test4)


# In[96]:


print(classification_report(y_test4,rfc4.pred))


# In[147]:


class_names = y4.unique()

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(rfc4, X_test4, y_test4,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)


# **Model 5-Dodgers Pitchers v. Astros Batters**

# In[97]:


#filter the dataset
data5 = all_data[ all_data['team_abv_pitcher']== 'LAD']
data5 = data5[data5['team_abv_batter']=='HOU']


# In[98]:


y5 = data5['pitch_class']
features5 = data5.drop(['pitch_type','pitch_class','team_abv_pitcher', 'team_name_pitcher', 'league_pitcher',
       'region_pitcher', 'batter_name', 'team_abv_batter', 'team_name_batter',
       'league_batter', 'region_batter'],axis=1)
features5 = pd.get_dummies(features5)
print(features5.shape)


# In[99]:


# Split the data into training and testing sets
X_train5, X_test5, y_train5, y_test5 = train_test_split(features5, y5, test_size = 0.25, random_state = 42)


# In[100]:


rfc5 = RandomForestClassifier(n_estimators = 100)
rfc5.fit(X_train5, y_train5)
rfc5.pred = rfc5.predict(X_test5)


# In[101]:


print(classification_report(y_test5,rfc5.pred))


# In[148]:


class_names = y5.unique()

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(rfc5, X_test5, y_test5,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)


# **Model 6-Astros Pitchers v. Dodger Batters**

# In[177]:


#filter the dataset
data6 = all_data[ all_data['team_abv_pitcher']== 'HOU']
data6 = data6[data6['team_abv_batter']=='LAD']


# In[178]:


y6 = data6['pitch_class']
features6 = data6.drop(['pitch_type','pitch_class','team_abv_pitcher', 'team_name_pitcher', 'league_pitcher',
       'region_pitcher', 'batter_name', 'team_abv_batter', 'team_name_batter',
       'league_batter', 'region_batter'],axis=1)
features6 = pd.get_dummies(features6)
print(features6.shape)


# In[179]:


# Split the data into training and testing sets
X_train6, X_test6, y_train6, y_test6 = train_test_split(features6, y6, test_size = 0.25, random_state = 42)


# In[180]:


rfc6 = RandomForestClassifier(n_estimators = 100)
rfc6.fit(X_train6, y_train6)
rfc6.pred = rfc6.predict(X_test6)


# In[181]:


print(classification_report(y_test6,rfc6.pred))


# In[182]:


class_names = y6.unique()

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(rfc6, X_test6, y_test6,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)


# **Draw a decision tree for all pitchers & batters to interprete**

# In[193]:


#choose a small sample to build a decision tree
BB = all_data[all_data['pitch_class']=='Breaking Balls'][:1000]
CU = all_data[all_data['pitch_class']=='Change Ups'][:1000]
FB = all_data[all_data['pitch_class']=='Fastballs'][:1000]
df = pd.concat([BB, CU], ignore_index=True)
df = pd.concat([FB,df], ignore_index=True)


# In[195]:


X222=df.drop(['pitch_type','pitch_class','team_abv_pitcher', 'team_name_pitcher', 'league_pitcher',
       'region_pitcher', 'batter_name', 'team_abv_batter', 'team_name_batter',
       'league_batter', 'region_batter','top_bottom', 'end_speed',
       'spin_rate', 'outcome_code', 'wins', 'loses', 'g_played', 'g_started',
       'g_finished', 'shutouts', 'saves', 'innings_pitched', 'hits', 'runs',
       'errors', 'hr', 'bb', 'int_bb', 'strike_outs', 'hit_by_pitch', 'balks',
       'wild'],axis=1)
X222 = pd.get_dummies(X222)
y222 = df['pitch_class']


# In[201]:


clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X222, y222)
# Draw Tree diagram and Export as dot file
export_graphviz(clf, out_file='tree.dot', 
                feature_names = X222.columns,
                class_names = y222.unique(),
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')


# **tree for model 2**

# In[210]:


#choose a small sample to build a decision tree
BB = data2[data2['pitch_class']=='Breaking Balls'][:1000]
CU = data2[data2['pitch_class']=='Change Ups'][:1000]
FB = data2[data2['pitch_class']=='Fastballs'][:1000]
df2 = pd.concat([BB, CU], ignore_index=True)
df2 = pd.concat([FB,df], ignore_index=True)
X2=df2.drop(['pitch_type','pitch_class','team_abv_pitcher', 'team_name_pitcher', 'league_pitcher',
       'region_pitcher', 'batter_name', 'team_abv_batter', 'team_name_batter',
       'league_batter', 'region_batter','top_bottom', 'end_speed',
       'spin_rate', 'outcome_code', 'wins', 'loses', 'g_played', 'g_started',
       'g_finished', 'shutouts', 'saves', 'innings_pitched', 'hits', 'runs',
       'errors', 'hr', 'bb', 'int_bb', 'strike_outs', 'hit_by_pitch', 'balks',
       'wild'],axis=1)
X2 = pd.get_dummies(X2)
y2 = df2['pitch_class']
clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X2, y2)
# Draw Tree diagram and Export as dot file
export_graphviz(clf, out_file='tree.dot', 
                feature_names = X2.columns,
                class_names = y2.unique(),
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')


# **tree for model 3**

# In[211]:


#choose a small sample to build a decision tree
BB = data3[data3['pitch_class']=='Breaking Balls'][:1000]
CU = data3[data3['pitch_class']=='Change Ups'][:1000]
FB = data3[data3['pitch_class']=='Fastballs'][:1000]
df3 = pd.concat([BB, CU], ignore_index=True)
df3 = pd.concat([FB,df], ignore_index=True)
X3=df3.drop(['pitch_type','pitch_class','team_abv_pitcher', 'team_name_pitcher', 'league_pitcher',
       'region_pitcher', 'batter_name', 'team_abv_batter', 'team_name_batter',
       'league_batter', 'region_batter','top_bottom', 'end_speed',
       'spin_rate', 'outcome_code', 'wins', 'loses', 'g_played', 'g_started',
       'g_finished', 'shutouts', 'saves', 'innings_pitched', 'hits', 'runs',
       'errors', 'hr', 'bb', 'int_bb', 'strike_outs', 'hit_by_pitch', 'balks',
       'wild'],axis=1)
X3 = pd.get_dummies(X3)
y3 = df3['pitch_class']
clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X3[:3000], y3[:3000])
# Draw Tree diagram and Export as dot file
export_graphviz(clf, out_file='tree.dot', 
                feature_names = X3.columns,
                class_names = y3.unique(),
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')


# **tree for model 4**

# In[212]:


BB = data4[data4['pitch_class']=='Breaking Balls'][:1000]
CU = data4[data4['pitch_class']=='Change Ups'][:1000]
FB = data4[data4['pitch_class']=='Fastballs'][:1000]
df4 = pd.concat([BB, CU], ignore_index=True)
df4 = pd.concat([FB,df], ignore_index=True)
X4=df4.drop(['pitch_type','pitch_class','team_abv_pitcher', 'team_name_pitcher', 'league_pitcher',
       'region_pitcher', 'batter_name', 'team_abv_batter', 'team_name_batter',
       'league_batter', 'region_batter','top_bottom', 'end_speed',
       'spin_rate', 'outcome_code', 'wins', 'loses', 'g_played', 'g_started',
       'g_finished', 'shutouts', 'saves', 'innings_pitched', 'hits', 'runs',
       'errors', 'hr', 'bb', 'int_bb', 'strike_outs', 'hit_by_pitch', 'balks',
       'wild'],axis=1)
X4 = pd.get_dummies(X4)
y4 = df4['pitch_class']
clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X4[:3000], y4[:3000])
# Draw Tree diagram and Export as dot file
export_graphviz(clf, out_file='tree.dot', 
                feature_names = X3.columns,
                class_names = y3.unique(),
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')


# **tree for model 5**

# In[131]:


# using Dodgers Pitchers v. Astros Batters data
X111 = data6.drop(['pitch_type','pitch_class','team_abv_pitcher', 'team_name_pitcher', 'league_pitcher',
       'region_pitcher', 'batter_name', 'team_abv_batter', 'team_name_batter',
       'league_batter', 'region_batter','top_bottom', 'end_speed',
       'spin_rate', 'outcome_code', 'wins', 'loses', 'g_played', 'g_started',
       'g_finished', 'shutouts', 'saves', 'innings_pitched', 'hits', 'runs',
       'errors', 'hr', 'bb', 'int_bb', 'strike_outs', 'hit_by_pitch', 'balks',
       'wild'],axis=1)
X111 = pd.get_dummies(X111)
y111 = y6


# In[140]:


clf5 = RandomForestClassifier(n_estimators=10, random_state=123,max_depth=3)
X_train7, X_test7, y_train7, y_test7 = train_test_split(X111, y111, test_size = 0.25, random_state = 42)


# In[141]:


clf5.fit(X_train7, y_train7)
clf5.pred = clf5.predict(X_test7)
print(classification_report(y_test7,clf5.pred))


# In[143]:


estimator = clf5.estimators_[1]
# Draw Tree diagram and Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = X111.columns,
                class_names = y111.unique(),
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')


# **tree for model 6**

# In[214]:


BB = data6[data6['pitch_class']=='Breaking Balls'][:1000]
CU = data6[data6['pitch_class']=='Change Ups'][:1000]
FB = data6[data6['pitch_class']=='Fastballs'][:1000]
df6 = pd.concat([BB, CU], ignore_index=True)
df6 = pd.concat([FB,df], ignore_index=True)

X6=df6.drop(['pitch_type','pitch_class','team_abv_pitcher', 'team_name_pitcher', 'league_pitcher',
       'region_pitcher', 'batter_name', 'team_abv_batter', 'team_name_batter',
       'league_batter', 'region_batter','top_bottom', 'end_speed',
       'spin_rate', 'outcome_code', 'wins', 'loses', 'g_played', 'g_started',
       'g_finished', 'shutouts', 'saves', 'innings_pitched', 'hits', 'runs',
       'errors', 'hr', 'bb', 'int_bb', 'strike_outs', 'hit_by_pitch', 'balks',
       'wild'],axis=1)
X6 = pd.get_dummies(X6)
y6 = df6['pitch_class']
clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X6, y6)
# Draw Tree diagram and Export as dot file
export_graphviz(clf, out_file='tree.dot', 
                feature_names = X3.columns,
                class_names = y3.unique(),
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')


# **Probability Analysis**

# **Dodger Pitchers v. Astros Hitters**

# In[127]:


clf1 = LogisticRegression(max_iter=1000, random_state=123)
clf2 = GaussianNB()
clf3 = RandomForestClassifier(n_estimators=100, random_state=123)

X = features5
y = y5

eclf = VotingClassifier(estimators=[('lr', clf1), ('gnb', clf2), ('rf', clf3)],
                        voting='soft',
                        weights=[1, 1, 5])

# predict class probabilities for all classifiers
probas = [c.fit(X, y).predict_proba(X) for c in (clf1, clf2, clf3, eclf)]


# In[110]:


print("Logistic Regression score:",clf1.score(X, y))
print("GaussianNB score:",clf2.score(X, y))
print("Random Forest Classifier score:",clf3.score(X, y))
print("Voting Classifier score:",eclf.score(X, y))


# In[136]:


# get class probabilities for the first sample in the dataset
class1_1 = [pr[0, 0] for pr in probas] #Fastballs
class2_1 = [pr[0, 1] for pr in probas] #Breaking Balls
class3_1 = [pr[0, 2] for pr in probas] #Change Ups

# plotting

N = 4  # number of groups
ind = np.arange(N)  # group positions
width = 0.25  # bar width

fig, ax = plt.subplots()

# bars for classifier 1-3
p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width,
            color='green', edgecolor='k')
p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width,
            color='lightgreen', edgecolor='k')
p3 = ax.bar(ind + 2*width, np.hstack(([class3_1[:-1], [0]])), width,
            color='seagreen', edgecolor='k')


# bars for VotingClassifier
p4 = ax.bar(ind, [0, 0, 0, class1_1[-1]], width,
            color='blue', edgecolor='k')
p5 = ax.bar(ind + width, [0, 0, 0, class2_1[-1]], width,
            color='steelblue', edgecolor='k')
p6 = ax.bar(ind + 2*width, [0, 0, 0, class3_1[-1]], width,
            color='lightblue', edgecolor='k')


# plot annotations
plt.axvline(2.8, color='k', linestyle='dashed')
ax.set_xticks(ind + width)
ax.set_xticklabels(['LogisticRegression\nweight 1',
                    'GaussianNB\nweight 1',
                    'RandomForestClassifier\nweight 5',
                    'VotingClassifier\n(average probabilities)'],
                   rotation=40,
                   ha='right')
plt.ylim([0, 1])
plt.title('Class probabilities for sample 1 by different classifiers')
plt.legend([p1[0], p2[0], p3[0]], ['Fastballs', 'Breaking Balls','Change Ups'], loc='upper left')
plt.tight_layout()
plt.show()


# **Astros Pitchers v. Dodgers Hitters**

# In[113]:


clf1 = LogisticRegression(max_iter=1000, random_state=123)
clf2 = GaussianNB()
clf3 = RandomForestClassifier(n_estimators=100, random_state=123)

X = features6
y = y6

eclf = VotingClassifier(estimators=[('lr', clf1), ('gnb', clf2), ('rf', clf3)],
                        voting='soft',
                        weights=[1, 1, 5])

# predict class probabilities for all classifiers
probas = [c.fit(X, y).predict_proba(X) for c in (clf1, clf2, clf3, eclf)]


# In[114]:


print("Logistic Regression score:",clf1.score(X, y))
print("GaussianNB score:",clf2.score(X, y))
print("Random Forest Classifier score:",clf3.score(X, y))
print("Voting Classifier score:",eclf.score(X, y))


# In[123]:


class1_1


# In[126]:


# get class probabilities for the first sample in the dataset
class1_1 = [pr[0, 0] for pr in probas] #Fastballs
class2_1 = [pr[0, 1] for pr in probas] #Breaking Balls
class3_1 = [pr[0, 2] for pr in probas] #Change Ups

# plotting

N = 4  # number of groups
ind = np.arange(N)  # group positions
width = 0.35  # bar width

fig, ax = plt.subplots()

# bars for classifier 1-3
p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width,
            color='green', edgecolor='k')
p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width,
            color='lightgreen', edgecolor='k')
p3 = ax.bar(ind + 2*width, np.hstack(([class3_1[:-1], [0]])), width,
            color='black', edgecolor='k')


# bars for VotingClassifier
p4 = ax.bar(ind, [0, 0, 0, class1_1[-1]], width,
            color='blue', edgecolor='k')
p5 = ax.bar(ind + width, [0, 0, 0, class2_1[-1]], width,
            color='steelblue', edgecolor='k')
p6 = ax.bar(ind + 2*width, [0, 0, 0, class3_1[-1]], width,
            color='lightblue', edgecolor='k')


# plot annotations
plt.axvline(2.8, color='k', linestyle='dashed')
ax.set_xticks(ind + width)
ax.set_xticklabels(['LogisticRegression\nweight 1',
                    'GaussianNB\nweight 1',
                    'RandomForestClassifier\nweight 5',
                    'VotingClassifier\n(average probabilities)'],
                   rotation=40,
                   ha='right')
plt.ylim([0, 1])
plt.title('Class probabilities for sample 1 by different classifiers')
plt.legend([p1[0], p2[0], p3[0]], ['Fastballs', 'Breaking Balls','Change Ups'], loc='upper left')
plt.tight_layout()
plt.show()

