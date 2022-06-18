#!/usr/bin/env python
# coding: utf-8

# #### Introduction
# This assignment aims to give you an idea of applying EDA in a real business scenario. In this assignment, apart from applying the techniques that you have learnt in the EDA module, you will also develop a basic understanding of risk analytics in banking and financial services and understand how data is used to minimise the risk of losing money while lending to customers.

# #### Business Understanding

#  The loan providing companies find it hard to give loans to the people due to their insufficient or non-existent credit history. Because of that, some consumers use it as their advantage by becoming a defaulter. Suppose you work for a consumer finance company which specialises in lending various types of loans to urban customers. You have to use EDA to analyse the patterns present in the data. This will ensure that the applicants capable of repaying the loan are not rejected.

# 
# When the company receives a loan application, the company has to decide for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision:

# - If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company
# - If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company.

# When a client applies for a loan, there are four types of decisions that could be taken by the client/company):

# - Approved: The Company has approved loan Application
# 
# - Cancelled: The client cancelled the application sometime during approval. Either the client changed her/his mind about the loan or in some cases due to a higher risk of the client he received worse pricing which he did not want.
# 
# - Refused: The company had rejected the loan (because the client does not meet their requirements etc.).
# 
# - Unused offer:  Loan has been cancelled by the client but on different stages of the process.

# #### Business Objectives

# This case study aims to identify patterns which indicate if a client has difficulty paying their installments which may be used for taking actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc. This will ensure that the consumers capable of repaying the loan are not rejected. Identification of such applicants using EDA is the aim of this case study.

# - In other words, the company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default.  The company can utilise this knowledge for its portfolio and risk assessment.
# - To develop your understanding of the domain, you are advised to independently research a little about risk analytics - understanding the types of variables and their significance should be enough).

# #### Importing Library

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',999) 
pd.set_option('display.max_rows',200)
pd.set_option('float_format', '{:f}'.format)
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Loading Data

# In[3]:


data=pd.read_csv(r"C:\Users\Lenovo\Downloads\application_data.csv")


# In[4]:


data.head(80)


# #### Cleaning Data

# In[5]:


# Finding the percentage of missing values in all columns in Application data
n1=data.isnull().sum()/data.shape[0]*100
n1


# As we can see the data contains high number of null values, there are more attributes which are having null values around 50% and then it slightly decreases to 31%, so we can drop the coulmns which are above 50%.

# In[6]:


# Removing all the columns from Application Data with more than 35% nulls values.
column=n1[n1>=50].index
data.drop(column,axis = 1,inplace=True)


# In[7]:


null1=data.isnull().sum()/data.shape[0]*100


# In[8]:


null1.sort_values(ascending=False)


# In[9]:


data.shape


# In[10]:


data.info()


# #### Data Imputation

# In[11]:


#Selecting columns with less or equal to than 50% null vallues
list(data.columns[(data.isnull().mean()<=0.50) & (data.isnull().mean()>0)])


# ####  Imputation in AMT_ANNUITY

# In[12]:


data['AMT_ANNUITY'].describe()


# In[13]:


# Since AMT_ANNUITY is a continuous variable. So checking for outliers
sns.boxplot(data['AMT_ANNUITY'])
plt.show()


# In[14]:


data['AMT_ANNUITY'].median()


# In[15]:


data['AMT_ANNUITY'].fillna(24903.0,inplace=True)


# Since AMT_ANNUITY has outliers, the column can be imputed using the median of the coumn i.e. 24903.0

# #### Imputation in AMT_GOODS_PRICE

# In[16]:


data['AMT_GOODS_PRICE'].describe()


# In[17]:


# Since AMT_ANNUITY is a continuous variable. So checking for outliers
sns.boxplot(data['AMT_GOODS_PRICE'])
plt.show()


# In[18]:


# Checking for median in AMT_GOODS_PRICE
data['AMT_GOODS_PRICE'].median()


# In[19]:


data['AMT_GOODS_PRICE'].fillna(450000.0,inplace=True)


# Since AMT_GOODS_PRICE has outliers, the column can be imputed using the median of the coumn i.e. 450000.0

# ####  Imputation in NAME_TYPE_SUITE

# In[20]:


# Checking for mode in Name type suite
data.NAME_TYPE_SUITE.mode()


# Clearly the column NAME_TYPE_SUITE is a categorical column. So this column can be imputed using the mode of the column i.e Unaccompanied

# In[21]:


data.NAME_TYPE_SUITE.fillna('Unaccompanied',inplace=True)


# #### Imputation in OCCUPATION_TYPE

# In[22]:


# Checking for most common value in Occupation Type
data.OCCUPATION_TYPE.value_counts()


# In[23]:


# checking for mode in occupation type
data.OCCUPATION_TYPE.mode()


# But,in the occupation columns we cann't fill NULL value with mode . so , we can fill null value with 'other' variable .

# In[24]:


# filling null value
data['OCCUPATION_TYPE'].fillna('Other', inplace=True)


# #### Imputation in CNT_FAM_MEMBERS

# In[25]:


# Plotting box plot to check for outliers
sns.boxplot(data['CNT_FAM_MEMBERS'])
plt.show()


# In[26]:


# Check for median in CNT_FAM_MEMBERS
round(data['CNT_FAM_MEMBERS'].median(),2)


# In[27]:


#Clearly CNT_FAM_MEMBERS is a continuous variable and we can impute the mean/median i.e. 2.0
data['CNT_FAM_MEMBERS'].fillna(2.0,inplace=True)


# In[28]:


data['CNT_FAM_MEMBERS'].unique()


# #### Imputation in EXT_SOURCE_2

# In[29]:


# Plotting box plot to check for outliers
sns.boxplot(data['EXT_SOURCE_2'])
plt.show()


# In[30]:


# Check for mean in EXT_SOURCE_2
round(data['EXT_SOURCE_2'].mean(),2)


# In[31]:


# Clearly EXT_SOURCE_2 has no outliers so we can impute the mean i.e. 0.51
data['EXT_SOURCE_2'].fillna(0.51,inplace=True)


# ####  Imputation in AMT_REQ_CREDIT_BUREAU

# In[32]:


round(data['AMT_REQ_CREDIT_BUREAU_HOUR'].median(),2)


# In[33]:


data['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0.0,inplace=True)


# In[34]:


round(data['AMT_REQ_CREDIT_BUREAU_DAY'].median(),2)


# In[35]:


data['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(0.0,inplace=True)


# In[36]:


round(data['AMT_REQ_CREDIT_BUREAU_WEEK'].median(),2)


# In[37]:


data['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(0.0,inplace=True)


# In[38]:


round(data['AMT_REQ_CREDIT_BUREAU_MON'].median(),2)


# In[39]:


data['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0.0,inplace=True)


# In[40]:


round(data['AMT_REQ_CREDIT_BUREAU_QRT'].median(),2)


# In[41]:


data['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0.0,inplace=True)


# In 'AMT_REQ_CREDIT_BUREAU_HOUR' ,'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', we can impute the median i.e. 0.0

# #### Imputation in ORGANIZATION_TYPE

# In[42]:


data.ORGANIZATION_TYPE.value_counts()


# In[43]:


# Replacing XNA with NaN 
data.replace('XNA',np.NaN)


# Imputing the value'NaN' which means not available for the column 'ORGANIZATION_TYPE'

# #### Testing Data Types

# In[44]:


# The columns with days.
days= [x for x in data if x.startswith('DAYS')]
days


# In[45]:


# Checking for the unique values present in all columns starting with 'DAYS'
print(data['DAYS_BIRTH'].unique())
print(data['DAYS_EMPLOYED'].unique())
print(data['DAYS_REGISTRATION'].unique())
print(data['DAYS_ID_PUBLISH'].unique())
print(data['DAYS_LAST_PHONE_CHANGE'].unique())


# As we can see these columns contains -ve values and dates cant be in negative, so we need to convert it in +ve

# In[46]:


# Applying abs() function to columns starting with 'DAYS' to convert the negative values to positive
data['DAYS_EMPLOYED']=abs(data['DAYS_EMPLOYED'])
data['DAYS_REGISTRATION']=abs(data['DAYS_REGISTRATION'])
data['DAYS_ID_PUBLISH']=abs(data['DAYS_ID_PUBLISH'])
data['DAYS_LAST_PHONE_CHANGE']=abs(data['DAYS_LAST_PHONE_CHANGE'])


# In[47]:


#Creating a column AGE using DAYS_BIRTH
data['AGE']=abs(data['DAYS_BIRTH'])//365.25


# In[48]:


data['AGE'].describe()


# In[49]:


# Value Count of Gender Column
data.CODE_GENDER.value_counts()


# In[50]:


# Checking for the unique values in Gender Column.
print(data['CODE_GENDER'].unique())


# We can see there are some Nan values, so lets impute them.

# In[51]:


# Replacing XNA value with F as number of F is more
data.loc[data.CODE_GENDER == 'XNA','CODE_GENDER'] = 'F'
data.CODE_GENDER.value_counts()


#  #### Finding Outliers

# In[52]:


# Finding outlier in days birth
sns.boxplot(data=data,x='DAYS_BIRTH')
plt.show()


# We dont see any outlier in Days Birth

# In[53]:


# Finding Outlier in Amount Annuity
sns.boxplot(data=data,x='AMT_ANNUITY')
plt.show()


# AMT Annuity shows outliers after 250000

# In[54]:


# Finding outliers in CNT FAM Members
sns.boxplot(data['CNT_FAM_MEMBERS'])
plt.show()


# CNT FAM members shows outliers around 20

# In[55]:


# Finding outliers in Goods Price
sns.boxplot(data['AMT_GOODS_PRICE'])
plt.show()


# AMT Goods Price shows outliers

# #### Binning Continious Variables

# In[56]:


# Bining 'AGE'
data['AGE_GROUP']=pd.cut(data['AGE'], bins=[18,25,35,60,100], labels=['Children','Young', 'Middle_Age', 'Senior_Citizen'])


# In[57]:


# Verifying age
data['AGE_GROUP'].value_counts()


# In[58]:


#Binning AMT_INCOME_TOTAL
data['INCOME_GROUP']=pd.qcut(data['AMT_INCOME_TOTAL'],q=[0,0.1,0.2,0.6,0.8,1],labels=['VeryLow','Low','Medium','High','VeryHigh'])


# In[59]:


data['INCOME_GROUP']


# #### Checking Data Imbalance

# In[60]:


# Counting percentage of defaulters and non defaulters
data['TARGET'].value_counts(normalize=True)*100


# In[61]:


# Plotting pieplot on Defaulter and non defaulter
plt.pie(data['TARGET'].value_counts(normalize=True)*100,labels=['NON-DEFAULTER','DEFAULTER'],explode=(0,0.05),autopct='%1.f%%')
plt.title('DEFAULTER Vs NON-DEFAULTER')
plt.show()


# As it shows Application data has high imbalance with Defaulted population of 8% as compared to Non-defualted population of 92%.

#  #### Splitting Data

# In[62]:


# Dividing the original dataset into two different datasets depending upon the target value
target_0 = data.loc[data['TARGET'] == 0]
target_1 = data.loc[data['TARGET'] == 1]


# ### Analysis

#  Univariate Analysis on Categorical Variable

#  Unorderd Catagorical

# #### Housing Type

# In[63]:


#Plotting bar plot on the Grouped data of Housing Type and Non- Defaulters
target_0['NAME_HOUSING_TYPE'].value_counts().plot.bar()
plt.title('Housing Type of Non-Defaulters')
plt.show()


# In[64]:


#Plotting bar plot on the Grouped data of Housing Type and Defaulters
target_1['NAME_HOUSING_TYPE'].value_counts().plot.bar()
plt.title('Housing Type of Defaulters')
plt.show()


# On comparing Housing Type we can see that people living with parents tend to default increases when compared with others.

# #### Gender

# In[65]:


#Plotting bar plot on the Grouped data of Gender and Non Defaulters
target_0['CODE_GENDER'].value_counts().plot.bar()
plt.title('Gender Distribution of Non- Defaulters')
plt.show()


# In[66]:


#Plotting bar plot on the Grouped data of Gender and Defaulters
target_1['CODE_GENDER'].value_counts().plot.bar()
plt.title('Gender Distribution of Defaulters')
plt.show()


# On comaparing the Defaulters and Non Defaulters on the basis of Gender, we observe that Females are the majority in both the cases.

# #### Income Type

# In[67]:


#Plotting bar plot on the Grouped data of Income Type and Non- Defaulters
target_0['NAME_INCOME_TYPE'].value_counts().plot.bar()
plt.title('Income Type of Non-Defaulters')
plt.show()


# In[68]:


#Plotting bar plot on the Grouped data of Income Type and Defaulters
target_1['NAME_INCOME_TYPE'].value_counts().plot.bar()
plt.title('Income Type of Defaulters')
plt.show()


# On comparing Income Type We can see that the students & BusinessMen don't default. The reason could be they are not required to pay during the time they are students. Most of the loans are distributed to working class people

# #### Organization Type

# In[69]:


#Plotting bar plot on the Grouped data of Organization Type and Non- Defaulters
plt.figure(figsize=[12,6])
target_0['ORGANIZATION_TYPE'].value_counts().plot.bar()
plt.title('Organization Type of Non-Defaulters')
plt.show()


# In[70]:


#Plotting bar plot on the Grouped data of Organization Type and Non- Defaulters
plt.figure(figsize=[12,6])
target_1['ORGANIZATION_TYPE'].value_counts().plot.bar()
plt.title('Organization of Non-Defaulters')
plt.show()


# On comparing organization type Bussiness entity type 3 tends to apply for more loans and there is also similarity in defaulting it

# #### Suit Type

# In[71]:


#Plotting bar plot on the Grouped data of Suite Type and Non- Defaulters
target_0['NAME_TYPE_SUITE'].value_counts().plot.bar()
plt.title('Suite Type of Non-Defaulters')
plt.show()


# In[72]:


#Plotting bar plot on the Grouped data of Suite Type and Defaulters
target_1['NAME_TYPE_SUITE'].value_counts().plot.bar()
plt.title('Suite Type of Defaulters')
plt.show()


# On comparing between defaulter and Non defaulter about type suite we see that unaccompained clients defaults more and also apply for loan more as compared to others

# #### Contract Type

# In[73]:


#Plotting bar plot on the Grouped data of Contract Type and Non- Defaulters
target_0['NAME_CONTRACT_TYPE'].value_counts().plot.bar()
plt.title('Contract Type of Non-Defaulters')
plt.show()


# In[74]:


#Plotting bar plot on the Grouped data of Contract Type and Defaulters
target_1['NAME_CONTRACT_TYPE'].value_counts().plot.bar()
plt.title('Contract Type of Defaulters')
plt.show()


# Ratio decreases of Revolving loan in defaulter section

# ### Orderd Catagorical

# #### Age Group

# In[75]:


#Plotting bar plot on the Grouped data of Age Grou and Non- Defaulters
target_0['AGE_GROUP'].value_counts().plot.bar()
plt.title('Age Group of Non-Defaulters')
plt.show()


# In[76]:


#Plotting bar plot on the Grouped data of Age Grou and Defaulters
target_1['AGE_GROUP'].value_counts().plot.bar()
plt.title('Age Group of Defaulters')
plt.show()


# On comparing age group we can see that distribution of loan is more in Middle age but also as compared to senior citizen childrens are defaulting more

# #### Income Group

# In[77]:


#Plotting bar plot on the Grouped data of Income and Non-Defaulters
target_0['INCOME_GROUP'].value_counts().plot.bar()
plt.title('Income Range of Non-Defaulters')
plt.show()


# In[78]:


#Plotting bar plot on the Grouped data of Income and Defaulters
target_1['INCOME_GROUP'].value_counts().plot.bar()
plt.title('Income Range of Defaulters')
plt.show()


# On comparing income group we observe that there is increase in Loan Defaulters whose income is low.

# #### Education Type

# In[79]:


#Plotting bar plot on the Grouped data of Education and Non-Defaulters
target_0['NAME_EDUCATION_TYPE'].value_counts().plot.bar()
plt.title('Education Type of Non-Defaulters')
plt.show()


# In[80]:


#Plotting bar plot on the Grouped data of Education and Defaulters
target_1['NAME_EDUCATION_TYPE'].value_counts().plot.bar()
plt.title('Education Type of Defaulters')
plt.show()


# By comparing education type we can observe an increase in percentage of Loan Defaulters whose educational qualifications are secondary/secondary and a decrease in the percentage of Loan Default who have completed higher education.

# #### Univariate Analysis on Continuous Variable

# In[81]:


# function to plot bar for continuous variables
def unidist(var):

    plt.style.use('ggplot')
    sns.despine
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,6))
    sns.distplot(a=target_0[var],ax=ax1)
    ax1.set_title(f' {var} for Non-Defaulters',fontsize=10)
    sns.distplot(a=target_1[var],ax=ax2)
    ax2.set_title(f'{var} for Defaulters',fontsize=10)    
    plt.show()


# In[82]:


unidist('AMT_ANNUITY')


# We can observe that the high default annuity amount lies between 20 to 40 thousand.

# In[83]:


unidist('AMT_CREDIT')


# In[84]:


unidist('AMT_GOODS_PRICE')


# ### Bivariate Analysis

# #### Catagorical Vs Numerical

# In[85]:


# Plotting bargraph for income group and credit amount for non defaulter
sns.barplot(data=target_0, x="INCOME_GROUP", y="AMT_CREDIT")
plt.show()


# In[86]:


# Plotting bargraph for income group and credit amount for defaulter
sns.barplot(data=target_1, x="INCOME_GROUP", y="AMT_CREDIT")
plt.show()


# On comparing income group with credit amount we observe that similarilty lies between both defaulters and non defaulters.

# In[87]:


# Plotting bargraph for education type and credit amount for non defaulter
plt.figure(figsize=[12,6])
sns.barplot(data=target_0, x="NAME_EDUCATION_TYPE", y="AMT_CREDIT")
plt.show()


# In[88]:


# Plotting bargraph for education type and credit amount for defaulter
plt.figure(figsize=[12,6])
sns.barplot(data=target_1, x="NAME_EDUCATION_TYPE", y="AMT_CREDIT")
plt.show()


# In[89]:


# Plotting bargraph for  Age group and credit amount for non defaulter
sns.barplot(data=target_0, x="AGE_GROUP", y="AMT_CREDIT")
plt.show()


# On comparing age group with credit amount we see that young people default more than senior citizen.

# ### Numerical Vs Numerical

# In[90]:


# Plotting pairplot for 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE','CNT_FAM_MEMBERS' for non defaulter
NvN = target_0[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE','CNT_FAM_MEMBERS']].fillna(0)
sns.pairplot(NvN)
plt.show()


# In[91]:


# Plotting pairplot for 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE','CNT_FAM_MEMBERS' for defaulter
NvN = target_1[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE','CNT_FAM_MEMBERS']].fillna(0)
sns.pairplot(NvN)
plt.show()


# #### Top 10 Correlation

# In[92]:


# Finding correlation for defaulter
corr=target_1.corr()
c_df = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool)).unstack().reset_index()
c_df.columns=['Var_1','Var_2','Correlation']
c_df.dropna(subset=['Correlation'],inplace=True)
c_df = c_df.sort_values(by=['Correlation'], ascending=False)
c_df.head(10)


# In[93]:


# Finding correlation for non defaulter
corr=target_0.corr()
c_df = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool)).unstack().reset_index()
c_df.columns=['Var_1','Var_2','Correlation']
c_df.dropna(subset=['Correlation'],inplace=True)
c_df = c_df.sort_values(by=['Correlation'], ascending=False)
c_df.head(10)


# In[ ]:





# ###  Data Analysis on Previous Application Data

# #### Checking Previous Application Data

# In[94]:


pd=pd.read_csv(r'C:\Users\Lenovo\Downloads\previous_application.csv')


# In[95]:


pd.head()


# In[96]:


pd.tail()


# In[97]:


pd.shape


# In[98]:


pd.info(verbose=True, null_counts=True)


# In[99]:


pd.describe()


# #### Data Cleaning

# In[100]:


# Finding the percentage of missing values in all columns in Previous application data
MissingPD=round(pd.isnull().mean()*100,2).sort_values(ascending = False)
MissingPD.head(60)


# There are several columns having null values between 99% to 22% and then it comes down to 0%, so lets drop the columns having null values above 22%.

# In[101]:


# Removing all the columns from Previous Application with more than 22% nulls values.
pd= pd.loc[:,pd.isnull().mean()<=0.22]
pd.shape


# In[102]:


# Replacing XNA and XAP with NaN 
pd=pd.replace('XNA', np.NaN)
pd=pd.replace('XAP', np.NaN)


# ### Univariate Analysis

# #### Catagorical

# In[103]:


# Plotting bar plot on contract status
pd['NAME_CONTRACT_STATUS'].value_counts().plot.bar()
plt.show()


# We can observe that majority of loans are approved and very less percentage of loans are unused offer

# In[104]:


# Plotting bar plot on payment type
pd['NAME_PAYMENT_TYPE'].value_counts().plot.bar()
plt.show()


# Almost all clients opption for Cash through bank

# In[105]:


# Plotting bar plot on reson for loan reject
pd['CODE_REJECT_REASON'].value_counts().plot.bar()
plt.show()


# HC was reason of rejection of most application before.

# In[106]:


# Plotting bar for goods category
pd['NAME_GOODS_CATEGORY'].value_counts().plot.bar()
plt.show()


# Mobile was the most loan applied goods

# In[107]:


# Plotting bar for client type
pd['NAME_CLIENT_TYPE'].value_counts().plot.bar()
plt.show()


# Most clients were repeater

# ###  Numerical

# In[108]:


# Plotting dist plot for credit amount
sns.distplot(pd['AMT_CREDIT'])
plt.show()


# ### Bivariate Analysis

# In[109]:


# plotting bar for credit amt vs contract status
sns.barplot(data=pd, x="NAME_CONTRACT_TYPE", y="AMT_CREDIT")
plt.show()


# In[ ]:





# As we can see Cash Loans has high amount credited

# In[111]:


# Plotting bar for client type vs credit amount
sns.barplot(data=pd, x="NAME_CLIENT_TYPE", y="AMT_CREDIT")
plt.show()


# Comparison of Credited Amount vs Client type shows that Refreshed comes after the Repeater client with very less difference.

# ### Data Merge

# In[ ]:





# In[112]:


# Merging Application data and Previous data
ND = pd.merge(data,on='SK_ID_CURR', how='left' )


# In[113]:


ND.shape


# In[114]:


ND.head()


# In[116]:


ND.info(4)


# In[128]:


# Obtaining pivot table on gender and contract status
pvtb = ND.pivot_table( values=['SK_ID_CURR'], index=['CODE_GENDER'],columns=['NAME_CONTRACT_STATUS'], aggfunc=np.sum)
pvtb


# In[129]:


# plotting var on gender vs contract status
pvtb.plot(kind='bar',stacked=True,figsize=(10,5))
plt.xlabel('CODE_GENDER')
plt.ylabel('CONTRACT_STATUS')
plt.show()


# Female has more number of approved loan as per male.

# In[131]:


# Obtaining pivot table on client type and contract status
pvtb2 = ND.pivot_table( values='SK_ID_CURR', index=['NAME_CLIENT_TYPE'],columns=['NAME_CONTRACT_STATUS'], aggfunc=np.sum)
pvtb2


# In[132]:


# plotting var on client type vs contract status
pvtb2.plot(kind='bar',stacked=True,figsize=(10,5))
plt.xlabel('CLIENT_TYPE')
plt.ylabel('CONTRACT_STATUS')
plt.show()


# Repeater and New has more number of approved loans.

# In[133]:


# Obtaining pivot table on target and contract status
pvtb3 = ND.pivot_table( values='SK_ID_CURR', index=['TARGET'],columns=['NAME_CONTRACT_STATUS'], aggfunc=np.sum)
pvtb3


# In[134]:


# plotting var on target vs contract status
pvtb3.plot(kind='bar',stacked=True,figsize=(10,5))
plt.xlabel('TARGET')
plt.ylabel('CONTRACT_STATUS')
plt.show()


# Few defaulter also get there loan approved but there is vast majority in Non defaulter.

# In[ ]:





# In[ ]:




