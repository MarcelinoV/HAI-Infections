# HAI-Infections
Linear Regression Analysis on Nosocomial Infections using a dataset from an extract from the Study on the Efficacy of Nosocomial Infection Control (SENIC)

This was an assignment in my Stastical Computer Packages course at the George Washington University, and I learned a lot about feature selection using wrapper methods and machine learning techniques in Regression because of this project.

Credits to **Vikashraj Luhaniwal from TowardDataScience.com** for his article on Feature Selection Using Wrapper Methods for the feature selection code I use in this  notebook. Also credits to **Dr. Tirthajyoti Sarkar from TowardDataScience.com** for  his article on How to Check the Quality of a Regression Model with Python and the code I use from his github to do the residual analysis in this notebook. Links to these articles are posted below:

## Code and Resources Used

**Python Version**: 3.6

**Packages**: pandas, numpy, matplotlib, seaborn, statsmodels

**Feature Selection Using Wrapper Methods by Vikashraj Luhaniwal**: https://towardsdatascience.com/feature-selection-using-wrapper-methods-in-python-f0d352b346f

**How to Check the Quality of a Regression Model with Python by Dr. Tirthajyoti Sarkar**: https://towardsdatascience.com/how-do-you-check-the-quality-of-your-regression-model-in-python-fa61759ff685

## Data

The is from an extract from the Study on the Efficacy of Nosocomial Infection Control (SENIC). The variables are the following:
- length of stay
- age
- infection risk
- routine culturing 
- routine chest x-ray
- num of beds
- med school affiliation 
- avg daily census 
- num of nurses
- available facilities & services

## Data Cleaning
With our test data being patient ID's 1-5, these rows are dropped from the original dataset.

## EDA
With Infection Risk as our Target Variable, I develop an understanding of the data with the following methods:
- View descriptive statistics of data (means, standard deviations, etc.)
- Visualize distributions and linearity via pairplot.
- Visualize correlations between variables.
- Further analyze variables with high correlations to target variable.

**Descriptive Statistics**
![alt text](https://github.com/MarcelinoV/Twitter-Covid-NLP-KMeans/blob/master/Images/Desc Stats.jpg "Descriptive Stats of Features")

**Correlations**
![alt text](https://github.com/MarcelinoV/Twitter-Covid-NLP-KMeans/blob/master/Images/Heatmap.jpg "Heatmap of Pearson Correlations between Variables")

Seeing that length of stay, routine culturing, routine chest x-ray, and available facilities & services have the highest correlation with infection risk, I then plotted their respective scatter plots against infection risk.

**Sample Scatter Plots**
![alt text](https://github.com/MarcelinoV/Twitter-Covid-NLP-KMeans/blob/master/Images/l_s_scatter.jpg "length of stay vs. infection risk")
![alt text](https://github.com/MarcelinoV/Twitter-Covid-NLP-KMeans/blob/master/Images/af&s_scatter.jpg "available facilities & services vs. infection risk")

## Model Building
To build a Linear Regression Model with the test data, I used these three feature selection methods:
- Forward Selection
- Backward Elimination
- Stepwise Selection

For the most part, all three methods selected **"length of stay", "routine culturing", and "available facilities & services"** as the top three features.

**Sample Feature Selection Output**
![alt text](https://github.com/MarcelinoV/Twitter-Covid-NLP-KMeans/blob/master/Images/Stepwise.jpg "Output with selected features (First Line) and OLS Regression Results")

## Model Analysis: Residual Analysis & Verifying Assumptions
To ensure we can trust our model, I had to verify the following assumptions about Linear Models:
- Independence of predictors
- Linearity with Target Variable
- Homoscedasticity
- Normally Distributed
- No Multicollinearity 

More of this is explored in the Jupyter Notebook, but overall, with the exception of No Multicollinearity (**there is multicollinearity**), all assumptions are satisfied.

**Linearity & Independence**
![alt text](https://github.com/MarcelinoV/Twitter-Covid-NLP-KMeans/blob/master/Images/r_c_res.jpg "Routine Culturing Residual Plot")

**Homoscedasticity**
![alt text](https://github.com/MarcelinoV/Twitter-Covid-NLP-KMeans/blob/master/Images/homo.jpg "Fitted vs Residuals Plot: Homoscedasticity")

**Multicollinearity**
![alt text](https://github.com/MarcelinoV/Twitter-Covid-NLP-KMeans/blob/master/Images/vif.jpg "Variance Inflation Factors")
- Note that there is Multicollinearity since length of stay and available facilities & services **have VIFs > 10**.

## Prediction Results

Our model, without further optmization, predicts the following:

**Test Data**
![alt text](https://github.com/MarcelinoV/Twitter-Covid-NLP-KMeans/blob/master/Images/test.jpg "Variance Inflation Factors")

**Predictions of Test Data**
![alt text](https://github.com/MarcelinoV/Twitter-Covid-NLP-KMeans/blob/master/Images/predictions.jpg "Variance Inflation Factors")

**Confidence and Prediction Intervals**
![alt text](https://github.com/MarcelinoV/Twitter-Covid-NLP-KMeans/blob/master/Images/conf_pred_inf.jpg "Variance Inflation Factors")

## Optimization Ideas
Since the model has an adjusted R-squared of .471, it is obvious that the model does need more optimazation to become more accurate and useful. I recommend the following:
- Sampling data which includes a factor that scores for the quality of sanitation a healthcare facility has, using criteria such as hand washing, presence of rodents, preventive measures against germ spread, use of gloves, etc. 
- Record or engineer with existing data the average of how many patients per room in a healthcare facility.
- Delete outliers, of which there are few, from dataset.
