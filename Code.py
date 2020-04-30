#!/usr/bin/env python
# coding: utf-8

# Credits to Vikashraj Luhaniwal from TowardDataScience.com for his article on
# Feature Selection Using Wrapper Methods for the feature selection code I use in this 
# notebook. Also credits to Dr. Tirthajyoti Sarkar from TowardDataScience.com for 
# his article on How to Check the Quality of a Regression Model with Python and the code
# I use from his github to do the residual analysis in this notebook. Links to these articles
# are posted here:

# https://towardsdatascience.com/feature-selection-using-wrapper-methods-in-python-f0d352b346f

# https://towardsdatascience.com/how-do-you-check-the-quality-of-your-regression-model-in-python-fa61759ff685

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
import scipy.stats 
from sklearn.linear_model import LinearRegression


# In[6]:


raw_data = pd.read_csv('Infections.csv')
raw_data.head()


# In[7]:


# Renaming of columns and cleanup
raw_data.dropna(0, how='any', inplace=True)

raw_data.rename(index=str, columns={'The SAS System': 'id num',
                              'Unnamed: 1': 'length of stay',
                              'Unnamed: 2': 'age',
                              'Unnamed: 3': 'infection risk',
                              'Unnamed: 4': 'routine culturing',
                              'Unnamed: 5': 'routine chest x-ray',
                              'Unnamed: 6': 'num of beds',
                              'Unnamed: 7': 'med school affiliation',
                              'Unnamed: 8': 'avg daily census',
                              'Unnamed: 9': 'num of nurses',
                              'Unnamed: 10': 'available facilities & services'}, inplace=True)

raw_data.drop('1', inplace=True)

raw_data.head()


# # An analysis of 108 obs, ID's 6-113

# In[8]:


data = raw_data.copy()
data.drop(['3','4','5','6','7'], axis=0, inplace=True)
raw_data.head()


# In[9]:


data


# # Exploratory Analysis
# **Target Variable: infection risk**

# In[10]:


# Descriptive Statistics
data = data.astype(float)

print(data.shape) # 108 rows, 11 columns
data.describe()


# In[11]:


# Pairwise scatter plots and correlation heatmap to check for 
# multicollinearity

sns.pairplot(data)


# In[12]:


# Table of Pearson correlations between each feature
data.corr()


# In[13]:


# Heatmap to see correlations easier
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True, linewidth=.5, ax=ax, yticklabels=False)
plt.show()


# With length of stay, routine culturing, routine chest x-ray, and available facilities & services having the highest correlation with infection risk, we shall further analyze these variables with scatter plots.

# In[14]:


# Creation of feature matrix
x = data.drop('infection risk', 1)

# Creation of response var
y = data['infection risk']


# In[15]:


# Scatterplots of highest correlating variables
cols = ['length of stay',
       'routine culturing',
       'routine chest x-ray',
       'available facilities & services']

for i in cols:
    plt.scatter(data[i], y)
    plt.xlabel(i, fontsize=14)
    plt.ylabel('Infection Risk', fontsize=14)
    plt.show()


# These graphs gives us an idea of which features could end up in our model for predicting infection risk. We shall now use wrapper methods (forward selction, backwards selection, & stepwise selection) to pick the best features.

# # 1. Forward Selection
# 
# **Steps for Forward Selection**
# 
# 1. Pick an alpha (ours will be alpha = 0.05).
# 
# 2. Fit each feature one at a time to and keep the one with the lowest p-value.
# 
# 3. Fit all possible models with one extra feature added to the previously selected feature(s).
# 
# 4. Again, select the feature with miniumum p-value. If p_value < alpha, continue, otherwise stop
# 
# The code to achieve this follows:

# In[16]:


def forward_selection(data, response, alpha=0.05):
    ini_feats = data.columns.tolist()
    best_feats = []
    
    while len(ini_feats) > 0:
        remaining_feats = list(set(ini_feats) - set(best_feats))
        new_pval = pd.Series(index=remaining_feats, dtype=float)
        
        for new_col in remaining_feats:
            model = sm.OLS(response, sm.add_constant(data[best_feats+[new_col]])).fit()
            new_pval[new_col] = model.pvalues[new_col]
        min_p_val = new_pval.min()
        
        if min_p_val < alpha:
            best_feats.append(new_pval.idxmin())
        else:
            break
    return best_feats, model.summary()


# In[17]:


forward_selection(x,y)


# # Backward Elimination 
# 
# **Steps for Backward Elimination**
# 
# 1. Pick our alpha (our alpha is alpha = 0.05).
# 
# 2. Fit full model with all features.
# 
# 3. Consider feature with highest p-value. If < alpha, go to next step, otherwise stop.
# 
# 4. Remove the feature under consideration.
# 
# 5. Fit a model without this feature. Repeat process from step 3.
# 
# The code follows:

# In[18]:


def backward_elimination(data, response, alpha=0.05):
    feats = data.columns.tolist()
    while len(feats) > 0:
        
        feats_with_constant = sm.add_constant(data[feats])
        p_vals = sm.OLS(response, feats_with_constant).fit().pvalues[1:]
        model = sm.OLS(response, feats_with_constant).fit()
        max_p_val = p_vals.max()
        
        if max_p_val >= alpha:
            excluded_feat = p_vals.idxmax()
            feats.remove(excluded_feat)
        else:
            break
    return feats, model.summary()


# In[19]:


backward_elimination(x,y)


# # 3. Stepwise Selection
# 
# **Steps for Stepwise Selection**
# A combination of forward selection and backwards elimination, this is how we'll do stepwise:
# 
# 1. Pick our alpha (alpha = 0.05).
# 
# 2. Perform next step of forward selection (newly added feat must have p-value < alpha).
# 
# 3. Perform all steps of backward elimination (any previous feat must have p-value > alpha).
# 
# 4. Repeat step 2 & 3 until final best set of feats.
# 
# The code is as follows:

# In[20]:


def stepwise_selection(data, response, alpha_in=0.05, alpha_out=0.05):
    ini_feats = data.columns.tolist()
    best_feats = []
    while len(ini_feats) > 0:
        
        remaining_feats = list(set(ini_feats) - set(best_feats))
        new_pval = pd.Series(index=remaining_feats)
        
        for new_col in remaining_feats:
            model = sm.OLS(response, sm.add_const(data[best_feats+[new_col]])).fit()
            new_pval[new_col] = model.pvalues[new_col]
        min_p_val = new_pval.min()
        
        if min_p_val < alpha_in:
            best_feats.append(new_pval.idxmin())
            
            while len(best_feats) > 0:
                best_feats_with_constant = sm.add_constant(data[best_feats])
                p_vals = sm.OLS(response, best_feats_with_constant).fit().pvalues[1:]
                max_p_val = p_vals.max()
                
                if max_p_val >= alpha_out:
                    excluded_feat = p_vals.idxmax()
                    best_feats.remove(excluded_feat)
                else:
                    break
        else:
            break
    return best_feats, model.summary()


# In[21]:


backward_elimination(x,y)


# **Conclusion** 
# 
# From our Feature Selection procedures, we can see that the best predictors of infection risk are the variables "length of stay", "routine culturing", and "available facilities & services". In that case, we shall create the variables containing our best features and our model:

# In[22]:


# Save the model that the regression methods created into variables

# Best variables recommended by wrapper methods
best_feats = data[['length of stay', 
                   'routine culturing', 
                   'available facilities & services']]

# Actual model
best_model = sm.OLS(y, best_feats).fit()


# # Residual Analysis
# 
# **Residuals vs. Prediciting Variable plots**
# 
# To confirm the independence assumption, we shall plot the residuals versus each of the best variables our model selection procedures (wrapper methods) picked out for us. **If the residuals are distributed uniformly randomly around the zero x-axes and do not form specific clusters, then the assumption holds true.**

# In[23]:


for c in best_feats:
    plt.figure(figsize = (8,5))
    plt.title('{} vs. \nModel Residuals'.format(c),
              fontsize=16)
    
    plt.scatter(x = data[c], y = best_model.resid, 
                color = 'green',
                edgecolor = 'k')
    
    plt.grid(True)
    xmin = min(data[c])
    xmax = max(data[c])
    
    plt.hlines(y=0, 
               xmin = xmin * 0.9,
               xmax = xmax * 1.1,color = 'yellow',
               linestyle = '--',
               lw = 3)
    
    plt.xlabel(c,fontsize = 14)
    
    plt.ylabel('Residuals',
               fontsize = 14)
    plt.show()


# Residual plots of best variables show some clustering but overall **assumptions of linearity and independence hold up** since the distribution is random around the 0 axis.

# **Fitted vs. Residuals**
# 
# By plotting fitted response values vs. residuals, we are checking for **constant variance of the residuals as the response variable increases.** If this isn't the case, it implies that a variable transformation may be needed to improve our model quality.

# In[24]:


plt.figure(figsize = (8,5))

p=plt.scatter(x = best_model.fittedvalues,
              y = best_model.resid,
              edgecolor='k')

xmin = min(best_model.fittedvalues)
xmax = max(best_model.fittedvalues)
plt.hlines(y = 0,
           xmin = xmin * 0.9,
          xmax = xmax*1.1,
          color = 'red',
          linestyle = '--',
          lw = 3)
plt.xlabel('Fitted values',
          fontsize = 15)
plt.ylabel('Residuals', 
           fontsize = 15)
plt.title('Fitted vs. Residuals Plot', 
          fontsize = 18)
plt.grid(True)
plt.show()


# Based on this plot, **homoscedasticity assumption is met.**

# **Histogram and Q-Q plot of normalized residuals**
# 
# To check the normality assumption, we'll generate a histogram and q-q plot of the normalized residuals.

# In[25]:


plt.figure(figsize = (8,5))
plt.hist(best_model.resid_pearson,
        bins = 20,
        edgecolor='k')
plt.ylabel('Count',
          fontsize = 15)
plt.xlabel('Normalized Residuals',
          fontsize = 15)
plt.title('Histogram of Normalized Residuals',
         fontsize = 18)
plt.show()


# In[26]:


from statsmodels.graphics.gofplots import qqplot


# In[27]:


plt.figure(figsize = (8,5))
fig = qqplot(best_model.resid_pearson,
            line = '45',
            fit = 'True')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Theoretical Quantiles',
          fontsize = 15)
plt.ylabel('Sample Quantiles',
          fontsize = 15)
plt.title('Q-Q plot of Normalized Residuals',
         fontsize = 18)
plt.grid(True)
plt.show()


# Based on the histogram and q-q plot, **the normality assumption is satisfied.**

# **Normality: Shapiro-Wilk Test of Residuals**

# In[28]:


_, p = scipy.stats.shapiro(best_model.resid)

if p < 0.05:
    print('The Residuals pass this test.')
else:
    print ('Normality NOT confirmed.')


# According to the Shapiro-Wilk test, **normality isn't confirmed since the p-values of the residuals aren't all less than alpha = 0.05.** While our residuals fail this normality test, they stilled passed the histogram and q-q plot tests, so we can still assume normality.

# **Cook's Distance (To check for outliers in residuals)**
# 
# Cook's distance measures how much effect deleting an observation has on the model. A large Cook's distance for a point can be a potential outlier.

# In[29]:


from statsmodels.stats.outliers_influence import OLSInfluence as inf


# In[30]:


inf = inf(best_model)

(c, p) = inf.cooks_distance
plt.figure(figsize = (8,5))
plt.title("Cook's Distance Plot of Residuals",
         fontsize = 16)
plt.stem(np.arange(len(c)),
        c,
        markerfmt = ',',
        use_line_collection = True)
plt.grid(True)
plt.show()


# Based on the Cook's Distance plot, **there are few data points with residuals possibly being outliers.**

# **Variance Inflation Factor (VIF)**
# 
# The VIF of eacb predictor allows us to check which factors to a degree cause multicollinearity in our model by dividing the ratio of variance in our multi-linear model by the variance of a simple-linear model.

# In[31]:


from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


# In[32]:


for i in range(len(best_feats.columns)):
    v = vif(np.matrix(best_feats), i)
    print('Variance Inflation Factor for {}: {}'.format(best_feats.columns[i],
                                                       round(v,2)))


# It seems that two factors in our model, **length of stay** and **available facilities & services**, have VIFs > 10. This means **there is multicollinearity in our model.**

# # Prediction & their Intervals
# 
# To test our model on patients with IDs 1-5, we shall create a new dataframe with just those rows, and get predictions from our model

# In[33]:


# Create test data from patient ids 1-5 with best features
test_data = pd.DataFrame(raw_data[:5], columns=['length of stay',
                                                'routine culturing',
                                                'available facilities & services'])
test_data = test_data.astype(float)
test_data


# In[34]:


from statsmodels.sandbox.regression.predstd import wls_prediction_std


# In[35]:


best_feats.shape


# In[36]:


# Confidence and Prediction Intervals
predictions = best_model.get_prediction(test_data)
predictions.summary_frame(alpha = 0.05)


# In[38]:


# Predicted Observations
predict = best_model.predict(test_data)
predict

