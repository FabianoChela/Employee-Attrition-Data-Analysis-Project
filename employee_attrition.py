# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:12:22 2024

@author: ADMIN
"""
# Importing libraries
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATA
# Save filepath to variable for easier access
employee_attrition_file_path = 'C:/Users/ADMIN/Employee_Attrition/employee_attrition.csv'

# Read and store the data in DataFrame titled employee_data
employee_data = pd.read_csv(employee_attrition_file_path)

# PREVIEW DATA
# View top five rows in the dataset
print(employee_data.head())

# View bottom five rows in the dataset
print(employee_data.tail())

# Preview the data # Number of rows and columns
print(employee_data.shape)

# Display a summary of the data in employee data
print(employee_data.describe())

# Display a list of all the columns/variables in the dataset
print(employee_data.columns.values)

# Check the data types of each column
print(employee_data.dtypes)

# Check for any blank cells in dataset
print(employee_data.isnull().sum())

# Check for duplicate rows
print(employee_data.duplicated())

# Count number of duplicated rows, if any.
print(employee_data.duplicated().value_counts())

# CLEAN DATA
# Check for columns with missing values
print(employee_data.isnull().values.any())

# Drop unnecessary columns
print(employee_data.drop('Employee_ID', axis = 1))

# Display the contents of the 'Satisfaction_Level' column
print(employee_data['Satisfaction_Level'])

# Converting values in the 'Satisfaction_Level' column to int64 percentgages
print((employee_data['Satisfaction_Level'] * 100).round().astype(int))

# Change attrition indicators of 0 and 1 to No and Yes respectively.
print(employee_data['Attrition'].replace( {0: 'No', 1: 'Yes'}, inplace = True))

# Display attrition column
print(employee_data['Attrition'].head())
print(employee_data['Attrition'].tail())

# Find outliers in all the numerical columns
# Calculate the z-score
for z_score_cols in employee_data[['Age', 'Years_at_Company', 'Satisfaction_Level',
        'Average_Monthly_Hours', 'Promotion_Last_5Years', 'Salary']]:
    column_series = employee_data[z_score_cols]
    print(np.abs(stats.zscore(employee_data[z_score_cols])))
    z = np.abs(stats.zscore(employee_data[z_score_cols]))

# Removal of Outliers with Z-Score
# Let’s remove rows where Z value is greater than 2.
    threshold_z = 2
    outlier_indices = np.where(z > threshold_z)[0]
    no_outliers = employee_data.drop(outlier_indices)
print("\nOriginal DataFrame Shape:", employee_data.shape)
print("DataFrame Shape after Removing Outliers:", no_outliers.shape)

# Unique values in dataset
print(employee_data.nunique())

# EXPLORATORY DATA ANALYSIS
# Question 1
'''Are certain departments experiencing higher levels of attrition than others? 
If so, why might that be?'''

# Display the attrition stats
attr_count = employee_data['Attrition'].value_counts()
print(attr_count)

# Distribution of attrition using a chart  #  to show pct '%. No of dec. places '1f'
# Pie chart
plt.pie(attr_count, labels = ['No', 'Yes'], autopct = '%.1f%%', explode = (0, 0))
plt.show()

# Bar chart
employee_data['Attrition'].value_counts().plot(kind = 'barh', figsize = (10,4))

plt.xlabel('Number of Employees', fontsize = 11)
plt.ylabel('Attrition', fontsize = 11)
plt.title('Employee Attrition', fontsize = 15)
plt.show()

# Find the unique values in department column
unique_values_dept = employee_data['Department'].unique()
print('\nThe deparments in the employees dataset are:', unique_values_dept)

# Find the employee distribution by departments 
dept_count = employee_data['Department'].value_counts()
dept_count_pct = (dept_count/sum(dept_count)) * 100
print('The departmental breakdown is:', dept_count, dept_count_pct)

# Pie chart showing departmental distribution
dept_labels = ['Sales', 'Finance', 'Engineering', 'HR', 'Marketing']
plt.pie(dept_count_pct, labels = dept_labels, autopct = '%.1f%%')
plt.show()

# Table showing attrition by department
dept_attr_tbl = pd.crosstab(employee_data.Department, columns = employee_data.Attrition)
print(dept_attr_tbl)

# Does the employee salary affect departmental attrition?
# No
sns.catplot(x = 'Department', y = 'Salary', hue = 'Attrition', kind = 'box',
            data = employee_data)
plt.show()

# Does the average monthly hours affect departmental attrition?
# No 
#avg_month_hr = employee_data['Average_Monthly_Hours'].mean()
#print(avg_month_hr)
sns.catplot(x = 'Department', y = 'Average_Monthly_Hours', hue = 'Attrition', 
           kind = 'box', data = employee_data)
plt.show() 

# Does the job title affect departmental attrition?
# No
job_title_count = employee_data['Job_Title'].value_counts()
print('\nThe Job Title breakdown is as follows:', job_title_count)

#sns.catplot(x = 'Department', y = job_title_count, hue = 'Attrition',
#            data = employee_data)
#plt.show()

# Does the age of an employee affect departmental attrition?
# No
sns.catplot(x = 'Department', y = 'Age', hue = 'Attrition', kind = 'box',
            data = employee_data)
plt.show()

# Does the number of years of employment affect departmental attrition?
# No
sns.catplot(x = 'Department', y = 'Years_at_Company', hue = 'Attrition', 
            kind = 'box', data = employee_data)
plt.show()

# Does the satisfaction level affect departmental attrition?
# No
sns.catplot(x = 'Department', y = 'Satisfaction_Level', hue = 'Attrition', 
            kind = 'box', data = employee_data)
plt.show()

# Does a promotion in the last 5 years affect departmental attrition?
# No
sns.catplot(x = 'Department', y = 'Promotion_Last_5Years', hue = 'Attrition', 
            kind = 'box', data = employee_data)
plt.show()

# Question 2					
'''What is the average tenure of employees who leave the company, and how 
does it compare to employees who stay long-term? '''

# Part A: What is the average tenure of employees who leave the company?
print(employee_data['Years_at_Company'].describe())

# Number of years served by employees # bar chart
employee_data['Years_at_Company'].value_counts().plot(kind = 'bar', figsize = (10,4), fontsize = 13)
plt.xlabel('Years_at_Company', fontsize = 11)
plt.ylabel('Number of Employees', fontsize = 11)
plt.title("Employees' Tenure", fontsize = 15)
plt.show()

# Years_at_Company vs Attrition chart
plt.subplots(figsize = (15,8))
sns.countplot(x = 'Years_at_Company', hue = 'Attrition', data = employee_data)
plt.title('Attrition in Relation to Years_at_Company', fontsize = 15)
plt.show()

# Question 3
# How does attrition vary based on demographics (e.g., age, gender, education level, etc.)?

# Part A: Age

# Number of unique ages
print(employee_data['Age'].nunique())

# Age column stats
print(employee_data['Age'].round().describe())

# Employees' Age Groups histogram
# The bins(towers) parameter determines how your data is divided e.g. bins = 6
employee_data.hist(column = 'Age', grid = False, figsize = (6,4), edgecolor = 'black',
                   bins = 7)
plt.xlabel('Age Groups', fontsize = 12)
plt.ylabel('Number of Employees', fontsize = 12)
plt.title("Distribution of Employees' Age Groups")
plt.show()

# Number of Employees' for specific ages
employee_data['Age'].value_counts().plot(kind = 'bar', figsize = (15,4), fontsize = 12)
plt.xlabel('Ages', fontsize = 12)
plt.ylabel('Number of Employees', fontsize = 12)
plt.title('Number of Employees at a Specific Age', fontsize = 15)
plt.show()

# Age vs Attrition table
age_vs_attr_tabl = pd.crosstab(employee_data.Age, columns = employee_data.Attrition)
print(age_vs_attr_tabl)

# Attrition in relation to Age bar chart
plt.subplots(figsize = (20,6))
sns.countplot(x = 'Age', hue = 'Attrition', data = employee_data, palette = 'colorblind')
plt.show()

# Part B: Gender

# Number of males vs females
print(employee_data['Gender'].value_counts())

# Number of males vs females bar chart
employee_data['Gender'].value_counts().plot(kind = 'bar', figsize = (8, 6), fontsize = 15)
plt.xlabel('Gender', fontsize = 15)
plt.ylabel('Number of Employees', fontsize = 15)
plt.title('Number of Employees by Gender', fontsize = 20)
plt.show()

# Attrition in relation to male vs female employees table
mal_vs_fem_tabl = pd.crosstab(employee_data.Gender, columns = employee_data.Attrition)
print(mal_vs_fem_tabl)

# Attrition in relation to male vs female employees bar chart
plt.subplots(figsize = (10, 6))
sns.countplot(x = 'Gender', hue = 'Attrition', data = employee_data, palette = 'colorblind')
plt.show()

# Question 4
'''What are the key indicators or drivers of employee attrition? 
(e.g. job satisfaction,Question 4 '''

# Part A: Job Satisfaction

# Satisfaction level description
print(employee_data['Satisfaction_Level'].describe())

# Satisfaction level histogram
employee_data.hist(column = 'Satisfaction_Level', grid = False, figsize = (15, 6),
                   edgecolor = 'black', bins = 10)
plt.xlabel('Satisfaction_Level', fontsize = 15)
plt.ylabel('Number of Employees', fontsize = 15)
plt.title('Job Satisfaction Scores', fontsize = 20)
plt.show()

# Number of job satisfaction level unique ratings
print(employee_data['Satisfaction_Level'].nunique())

# Set Satisfaction_Level into categories
''' E.g. 0.00 to 0.25 - Poor
         0.26 to 0.50 - Fair
         0.51 to 0.75 - Good
         0.76 to 1.00 - Outstanding '''
         
job_rating = []
for rating in employee_data['Satisfaction_Level']:
    if rating > 0.00 and rating <= 0.25:
        job_rating.append('Poor')
    elif rating >= 0.26 and rating <= 0.50:
        job_rating.append('Fair')
    elif rating >= 0.51 and rating <= 0.75:
        job_rating.append('Good')
    else:
        job_rating.append('Outstanding')
print(job_rating)

# Replace 0.00 to 0.99 job scores with job rating list.
# Example: df['a'].where(~(df.a < 0), other=0, inplace=True)
''' employee_data['Satisfaction_Level'].where(~(employee_data.Satisfaction_Level > 0.00 and 
                            employee_data.Satisfaction_Level <= 0.25), other = 'Poor', inplace = True)
employee_data['Satisfaction_Level'].where(~(employee_data.Satisfaction_Level > 0.26 and 
                            employee_data.Satisfaction_Level <= 0.50), other = 'Fair', inplace = True)
employee_data['Satisfaction_Level'].where(~(employee_data.Satisfaction_Level > 0.51 and 
                            employee_data.Satisfaction_Level <= 0.75), other = 'Good', inplace = True)
employee_data['Satisfaction_Level'].where(~(employee_data.Satisfaction_Level > 0.76 and 
                            employee_data.Satisfaction_Level <= 1.00), other = 'Poor', inplace = True)

print(employee_data['Satisfaction_Level'].head())
print(employee_data['Satisfaction_Level'].tail()) '''

# Breakdown of job satistaction scores
#for job_rate in range(1):
print('Poor:', '\t\t', job_rating.count('Poor'))
print('Fair:', '\t\t', job_rating.count('Fair'))
print('Good:', '\t\t', job_rating.count('Good'))
print('Outstanding:', job_rating.count('Outstanding'))
    
# Satisfaction level pie chart
rating_labels = ['Poor', 'Fair', 'Good', 'Outstanding']
rating_count = [247, 245, 234, 274]
plt.pie(rating_count, labels = rating_labels, autopct = '%.1f%%')
plt.show()

# Satisfaction level vs attrition table
job_rating_tab = pd.crosstab(job_rating, columns = employee_data.Attrition)
print(job_rating_tab)

# Part B: Average Monthly Hours

# Average Monthly Hours description
print(employee_data['Average_Monthly_Hours'].describe())

# Unique Average Monthly Hours values
print(employee_data['Average_Monthly_Hours'].nunique())

# Average_Monthly_Hours displot chart
employee_data.hist(column = 'Average_Monthly_Hours', grid = False, figsize = (15, 6),
                   edgecolor = 'black', bins = 10)
plt.xlabel('Average_Monthly_Hours', fontsize = 15)
plt.ylabel('Number of Employees', fontsize = 15)
plt.title("Employees' Average Monthly Hours", fontsize = 20)
plt.show()     

# Average_Monthly_Hours vs Attrition catplot (categorical plot)
sns.catplot(x = 'Attrition', y = 'Average_Monthly_Hours', kind = 'box', data = employee_data)
plt.title('Attrition in Relation to Average_Monthly_Hours', fontsize = 20)
plt.show()

# Part C: Salary

# Salary description
print(employee_data['Salary'].describe())

# unique salary values
print(employee_data['Salary'].nunique())

# Salary histplot chart
sns.histplot(employee_data['Salary']) 
plt.show()

# Salary vs Attrition catplot
sns.catplot(x = 'Attrition', y = 'Salary', kind = 'box', data = employee_data)
plt.title('Attrition in Relation to Employee Salary')
plt.show()

# Part D: Promotion in the last Five (5) Years

# Replace 0 and 1 with No and Yes respectively
print(employee_data['Promotion_Last_5Years'].replace({0: 'No', 1: 'Yes'}, inplace = True))
print(employee_data['Promotion_Last_5Years'].head())
print(employee_data['Promotion_Last_5Years'].tail())

# Promotion description
print(employee_data['Promotion_Last_5Years'].describe())

# Promotion bar chart
employee_data['Promotion_Last_5Years'].value_counts().plot(kind = 'bar', figsize = (8, 6), fontsize = 15)
plt.xlabel('Promotion_Last_5Years', fontsize = 15)
plt.ylabel('Number of Employees', fontsize = 15)
plt.title('Number of Employees by Promotion_Last_5Years', fontsize = 20)
plt.show()


# Promotion count
print(employee_data['Promotion_Last_5Years'].value_counts())

# Promotion table
promo_tab = pd.crosstab(employee_data.Promotion_Last_5Years, columns = employee_data.Attrition)
print(promo_tab)

# Part E: Job Title

# Job title description
print(employee_data['Job_Title'].describe())

# Job title count
print(employee_data['Job_Title'].value_counts())

# Job title pie chart
employee_data['Job_Title'].value_counts().plot(kind = 'bar', figsize = (12, 6), fontsize = 15)
plt.xlabel('Job_Title', fontsize = 15)
plt.ylabel('Number of Employees', fontsize = 15)
plt.title('Number of Employees by Job_Title', fontsize = 20)
plt.show()

# Job title vs attrition table
job_tab = pd.crosstab(employee_data.Job_Title, columns = employee_data.Attrition)
print(job_tab)

# Job title vs attrition countplot
plt.subplots(figsize = (15, 8))
#sns.set(font_scale = 10)
sns.countplot(x = 'Job_Title', hue = 'Attrition', data = employee_data)
plt.title('Attritions in realtion to Job Title', fontsize = 20)
plt.show()

# Job title and Age vs Attrition boxplot
sns.catplot(x = 'Job_Title', y = 'Age', hue = 'Attrition', kind = 'box', data = employee_data)
plt.show()

# Job title and Salary vs Attrition boxplot
sns.catplot(x = 'Job_Title', y = 'Salary', hue = 'Attrition', kind = 'box', data = employee_data)
plt.show()

# Job title and Years_at_Company vs Attrition boxplot
sns.catplot(x = 'Job_Title', y = 'Years_at_Company', hue = 'Attrition', kind = 'box', data = employee_data)
plt.show()

# Job title and Average_Monthly_Hours vs Attrition boxplot
sns.catplot(x = 'Job_Title', y = 'Average_Monthly_Hours', hue = 'Attrition', kind = 'box', data = employee_data)
plt.show()
