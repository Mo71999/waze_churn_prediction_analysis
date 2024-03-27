# Waze Churn Analysis

# Overview

# Waze is subsidiary company of Google that provides satellite navigation on devices that supports GPS.
# The leadership at Waze has tasked the data team with developing a machine learning model for anticipating user attrition.
# This model relies on information gathered from individuals using the Waze application.

# Objective

# I have been assigned the responsibility to develop a machine learning model by the Waze leadership.
# The objective is to predict user churn using data obtained from users of the Waze app.
# To develop a ML model for predicting user churn using data obtained from users of the Waze app

# Stage 1: Inspect and Analyze Data
# Purpose
# To investigate and understand the data by constructing dataframe with python and perform cursory inspection of the dataset

# Key Activities

# Prepae the data by reviewing the data dictionary, and conducting an initial exploration of the dataset
# to pinpoint essential variables for the stakeholder's needs

# Generate a pandas dataframe to facilitate data learning, future exploratory data analysis (EDA), and statistical tasks

# Aggregate comprehensive summary details about the dataset to guide subsequent actions

#################################################################################################################
# Import packages for data manipulation
import pandas as pd
import numpy as np
import seaborn as sns


# Load dataset into dataframe
# Load dataset into dataframe
from matplotlib import pyplot as plt

df = pd.read_csv('waze_dataset.csv')

print(df.head(10))
print(df.info())

# Key Takeaways:
#
# First 10 observations have no missing value, in-depth analysis for missing value still needed.
# The dataset consists of 14,999 rows and 13 columns. The variable types are as follows:
# label and device are categorical variables of type object.
# total_sessions, driven_km_drives, and duration_minutes_drives are numerical variables of type float64.
# The remaining variables are numerical variables of type int64.
# Within the dataset, the label column contains 700 missing values.

########################################################################################################################
########################################################################################################################

      # PART 1

########################################################################################################################

                         ########################################################

#    a)  Null values and summary statistics

# Contrast the summary statistics of the 700 rows that are missing labels with
# the summary statistics of the rows containing complete data.

null_df = df[df['label'].isnull()]
print(null_df.describe())
# Isolate rows without null values
not_null_df = df[~df['label'].isnull()]
# Display summary stats of rows without null values
not_null_df.describe()

# Key takeaway:

# The means and standard deviations are fairly consistent between the two groups having missing data and complete data

                    #########################################################
#     b)        Device Count by Null Values

# Get count of null values by device
null_df['device'].value_counts()

# Calculate % of iPhone nulls and Android nulls
null_df['device'].value_counts(normalize=True)

# Calculate % of iPhone users and Android users in full dataset
df['device'].value_counts(normalize=True)

# Key takeaway:
#
# Out of 700 rows with missing values, 447 were iPhone users and 253 were Android users.

                 #################################################################

#      c)       Device Ratio Comparison

#Comparing device ratio within the Null Values in dataset

# Calculate % of iPhone nulls and Android nulls
null_df['device'].value_counts(normalize=True)

#Comparing to the device ratio within Complete dataset

# Calculate % of iPhone users and Android users in full dataset
df['device'].value_counts(normalize=True)

# Key takeaway:

#The percentage of missing values by each device is consistent with their representation in the data overall.
#There is nothing to suggest a non-random cause of the missing data


                ################################################3########################

#      d)       Churned Vs Retained Users

# Calculate counts of churned vs. retained

print(df['label'].value_counts())
print()
print(df['label'].value_counts(normalize=True))

# KeyTakeaway:

      # This dataset contains 82% retained users and 18% churned users.


# Median comparison of retained and churned Users, we use median to minimize outliers unduly affect the analysis

# Calculate median values of all columns for churned and retained users
df.groupby('label').median(numeric_only=True)


# KeyTakeaway:
###############
# Churned users had about ~3 more drives on average in the past month than retained users.
# Meanwhile, retained users used the app for more than twice the number of days compared to churned users in the same period.
# The median churned user drove ~200 more kilometers and 2.5 more hours during the last month than the median retained user.
# This pattern indicates that churned users engaged in a higher frequency of drives within a shorter span,
# with farther trips and longer durations. This could potentially hint at distinctive user profiles


         ###########################################################################


#      e)        Comparison - Average Kilometers/Drive

# # Group data by `label` and calculate the medians
medians_by_label = df.groupby('label').median(numeric_only=True)
print('Median kilometers per drive:')
# Divide the median distance by median number of drives
print("Median kilometers per drive: \n")
kmPer_drive = medians_by_label['driven_km_drives'] / medians_by_label['drives']


# KeyTakeaway:

    # The median user from both groups drove ~73 km/drive. How many kilometers per driving day was this?

# Median kilometers per driving day

# Divide the median distance by median number of driving days
print('Median kilometers per driving day:')
kmPer_DrivingDay = medians_by_label['driven_km_drives'] / medians_by_label['driving_days']


# Median number of drives per driving day for each group

# Divide the median number of drives by median number of driving days
print('Median drives per driving day:')
Num_drives_perDrivingDay = medians_by_label['drives'] / medians_by_label['driving_days']


# KeyTakeaway:
################

#Median churned user covered 608 kilometers per driving day last month,
# nearly 250% more than retained users. Similarly, the median churned user had
# significantly more drives per drive day compared to retained users
#These figures clearly show that the users in the dataset, regardless of churn status,
# are avid drivers. It's likely that the data doesn't represent typical drivers.
# The sample of churned users might consist largely of long-haul truckers
#Given the extensive driving habits of these users, it's advisable for Waze to consider
# collecting additional data on these highly active drivers. The underlying reason for their
# high mileage could potentially shed light on why the Waze app might not align with their
# unique requirements. These needs might differ significantly from those of a more usual driver, like a daily commuter

##########

# Counts of each device type for each group - churned and retained

df.groupby(['label', 'device']).size()


# Now, within each group, churned and retained, calculate what percent was Android and what percent was iPhone.

# For each label, calculate the percentage of Android users and iPhone users
df.groupby('label')['device'].value_counts(normalize=True)

df.groupby('label')['device'].value_counts(normalize = True)

# Key Takeaway:
#######################

# The ratio of iPhone users and Android users is consistent between the churned group and the retained group, and those ratios are both consistent with the ratio found in the overall dataset.

# Conclusion - Inspect and Analyze Data ( PART 1 )
###################################################

# The dataset has 700 missing values in the label column. There was no obvious pattern to the missing values
# Mean is subject to the influence of outliers, while the median represents the middle value of the distribution
# regardless of any outlying values
# investigation gave rise to further questions, the median user who churned drove 608 kilometers each day they
# drove last month, which is almost 250% the per-drive-day distance of retained users. It would be helpful
# to know how this data was collected and if it represents a non-random sample of users
# Android users comprised approximately 36% of the sample, while iPhone users made up about 64%
# Generally, users who churned drove farther and longer in fewer days than retained users.
# They also used the app about half as many times as retained users over the same period
# The churn rate for both iPhone and Android users was within one percentage point of each other.
# There is nothing suggestive of churn being correlated with device


###########################################################################################################################
############################################################################################################################



#                                                    PART 2



#######################################################################################################


#                     Stage 2: Exploratory Data Analysis - EDA
#Purpose

#To conduct exploratory data analysis (EDA) on a provided dataset

#Goal
#To continue examination by further exploring data and adding relevant visualizations that help in communicating the data story

#Key Activities
######################

#Import and Load data
#Data Exploration & Data Cleaning
#Creating Visualizations
#Evaluating and sharing results


# Identifying Outliers
###########################################

# i will use numpy functions mean() and median(), then use boxplot to visualize the distribution of data
#
# Handling outliers involves three options: deleting, reassigning, or leaving them as is. Your choice depends on dataset specifics and modeling goals.
#
# Delete: Remove outliers if they're errors or the dataset is for modeling. This option is used less frequently.
#
# Reassign: Create new values to replace outliers in small datasets or when preparing data for modeling.
#
# Leave: Keep outliers for exploratory analysis or when the model is robust to outliers (we will follow this one as we want further exploration)
#

##############################################

# Activity 2c - Visualizations

#############################################

# We will construct viz of following events

# Session
# Drives
# Total Sessions
# No of Days after on boarding
# Driven Km Drives
# Duration in minutes during complete month
# App Activity during complete month
# No of days user drives during complete month
# Device Distribution
# Labels(Churned, Retained)
# Driving Days, Activity Days
# Retention by Device
# Retention by kilometers driven per driving day
# Churn rate per number of driving days
# Proportion of sessions that occurred in the last month


###### Sessions
# The frequency of a user accessing the app within the month

# Box plot
# Create a centered box plot
plt.figure(figsize=(8, 1.5))
sns.boxplot(x=df['sessions'], fliersize=2)
plt.title('Sessions Box Plot')
plt.show()  # Display the plot


# Histogram
plt.figure(figsize=(8, 3))
sns.histplot(x=df['sessions'])
median = df['sessions'].median()
plt.axvline(median, color='red', linestyle='--')
plt.text(75,1200, 'median=56.0', color='red')
plt.title('sessions box plot');


# Key Takeaways
##########
#The sessions variable exhibits a right-skewed distribution, with approximately 50% of data points having 56 sessions or fewer
#As depicted in the boxplot, there are instances where certain users have accumulated more than 700 sessions


#Drives

#############*An instance of traveling a distance of at least 1 kilometer within the span of a month

# Box plot
plt.figure(figsize=(8,1.5))
sns.boxplot(x=df['drives'], fliersize=1)
plt.title('drives box plot');


def histogrammer(column_str, median_text=True, **kwargs):
    """
    Helper function to plot histograms based on the format of the `sessions` histogram.

    Parameters:
    - column_str (str): The name of the column in the DataFrame to be plotted.
    - median_text (bool, optional): Whether to display the median value as text on the plot (default is True).
    - **kwargs: Additional keyword arguments to be passed to the sns.histplot() function.

    Returns:
    - None: This function displays the histogram plot with an optional median line and text.
    """
    median = round(df[column_str].median(), 1)
    plt.figure(figsize=(8, 2.5))
    ax = sns.histplot(x=df[column_str], **kwargs)  # Plot the histogram
    plt.axvline(median, color='red', linestyle='--')  # Plot the median line
    if median_text == True:  # Add median text unless set to False
        ax.text(0.25, 0.85, f'median={median}', color='red',
                ha='left', va='top', transform=ax.transAxes)
    else:
        print('Median:', median)
    plt.title(f'{column_str} histogram')

# Histogram
histogrammer('drives')

###  Key Takeaways
########################
#The distribution of the drives data closely resembles that of the sessions variable
#It exhibits a right-skewed pattern that approximates a log-normal distribution, with a median of 48
#There are instances where certain drivers recorded more than 400 drives within the past month


#Total Sessions
#Calculation of the cumulative count of sessions that have occurred since a user's initial onboarding.

# Box plot
plt.figure(figsize=(8, 1.5))
sns.boxplot(x=df['total_sessions'], fliersize=1)
plt.title('total_sessions box plot')


#The distribution of total_sessions is skewed to the right
#The median value for the total number of sessions is approximately 159.6
#This piece of information is intriguing because, when comparing it to the median number of sessions in the last month (which was 48), it suggests that a substantial portion of a user's overall sessions may have occurred within that last month
#This presents an interesting aspect for further investigation in the future


# No of Days after Onboarding
# The duration in days since a user registered for the app.

# Box plot
plt.figure(figsize=(8,1.5))
sns.boxplot(x=df['n_days_after_onboarding'], fliersize=1)
plt.title('No of Days after Onboarding Box Plot')

# Histogram
histogrammer('n_days_after_onboarding', median_text=False)

#Key Takeaway
#The complete user tenure, which represents the number of days since onboarding, follows a uniform distribution
#Values ranging from almost 0 to approximately 3,500 days, ~ 9.5 years

#Driven Km Drives
#Total kilometers driven during the month

# Box plot
plt.figure(figsize=(8,1.5))
sns.boxplot(x=df['driven_km_drives'], fliersize=1)
plt.title('Driven Km Drives Box Plot')

