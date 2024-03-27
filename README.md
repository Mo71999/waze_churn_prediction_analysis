Predicting The Churn of Waze Users
(If a user is going to stop using waze or not)  
Author(s):
Mohamed Mahmoud

1. Introduction

The goal of the project is to predict if a user is going to be churned or retained. The data is from Waze, it contains about 15,000 records and it has about 13 attributes:

-	Id
-	label                     
-	sessions                   
-	drives                   
-	total_sessions           
-	n_days_after_onboarding  
-	total_navigations_fav1    
-	total_navigations_fav2     
-	driven_km_drives         
-	duration_minutes_drives  
-	activity_days              
-	driving_days               
-	device       

I’m going to use Recurrent Neural Network to predict the value of the attribute Label.

2. Dataset Analysis

The Waze dataset contains user data and activity data. Key metrics include sessions, drives, total sessions, days after onboarding, favorite navigations, driven kilometers, drive duration, and device type. The dataset's primary objective is to understand user retention patterns, as indicated by the target variable 'label', which categorizes users based on their engagement level. 
•	Features and Data Types:
•	Sessions (Numerical): Number of app sessions.
•	Drives (Numerical): Number of drives recorded.
•	Total Sessions (Numerical): Aggregate of sessions over a period.
•	Days After Onboarding (Numerical): Time since the user started using the app.
•	Favorite Navigations (Numerical): Frequency of using favorite routes.
•	Driven Kilometers (Numerical): Total distance driven.
•	Duration Minutes Drives (Numerical): Total duration of drives.
•	Device (Categorical): Type of device used (Android or iPhone).
•	Label (Categorical): User retention category.
•	Preprocessing:
•	Numerical features were standardized to have a mean of 0 and a standard deviation of 1.
•	Categorical features, such as 'device', were encoded into numeric formats.
•	The target variable 'label' was encoded for classification.


Findings:

-	The label attribute, which is also the target value has about 700 missing values out of 15,000.

-	Out of 700 rows with missing values, 447 were iPhone users and 253 were Android users.

-	11,763 users are retained (82%)

-	2,536 users are churned (18%)

-	Churned users had about 3 more drives on average in the past month than retained users. Meanwhile, retained users used the app for more than twice the number of days compared to churned users in the same period.

-	churned users drove 200 more kilometers and 2.5 more hours during the last month than the median retained user.
-	This pattern indicates that churned users engaged in a higher frequency of drives within a shorter span than retained users.

3. Machine Learning Algorithm(s) Description

•	Model Choice: I chose Recurrent Neural Network (RNN) for its proficiency in handling sequential data, ideal for the time-dependent features in the dataset such as session.
•	Model Details: The RNN consisted of a simple RNN layer with 50 units, which is the number of hidden neurons. This was followed by a dense layer for classification, with an output size matching the number of unique labels.
•	Training Details: The model was compiled using the 'adam' optimizer and 'categorical_crossentropy' as the loss function, appropriate for multi-class classification. It was trained over 10 epochs with a batch size of 32.


4. Results Analysis

•	Experimental Process: The model was trained on a subset of the data, with 70% used for training and 30% for testing. The training involved feeding the network with batches of data, where it learned to predict the 'label' based on the input features.
•	Performance Metrics: The RNN's performance was evaluated based on its accuracy and loss on the test dataset. These metrics provided insights into how well the model generalized to unseen data.
•	Outcomes: Test accuracy is  0.7919999957084656


4. Discussion and Conclusion
Summarize the main insights drawn from your analysis and experiments. You can get a good project grade with mostly negative results, as long as you show evidence of extensive exploration, thoughtfully analyze the causes of your negative results, and discuss potential solutions.

The result of using the RNN model:

Epoch 1/10
329/329 [==============================] - 3s 4ms/step - loss: 0.6247 - accuracy: 0.7756 - val_loss: 0.5757 - val_accuracy: 0.7862
Epoch 2/10
329/329 [==============================] - 1s 3ms/step - loss: 0.5944 - accuracy: 0.7802 - val_loss: 0.5725 - val_accuracy: 0.7911
Epoch 3/10
329/329 [==============================] - 1s 3ms/step - loss: 0.5885 - accuracy: 0.7828 - val_loss: 0.5712 - val_accuracy: 0.7916
Epoch 4/10
329/329 [==============================] - 1s 3ms/step - loss: 0.5858 - accuracy: 0.7836 - val_loss: 0.5701 - val_accuracy: 0.7920
Epoch 5/10
329/329 [==============================] - 1s 3ms/step - loss: 0.5834 - accuracy: 0.7843 - val_loss: 0.5715 - val_accuracy: 0.7920
Epoch 6/10
329/329 [==============================] - 1s 3ms/step - loss: 0.5820 - accuracy: 0.7857 - val_loss: 0.5698 - val_accuracy: 0.7924
Epoch 7/10
329/329 [==============================] - 1s 4ms/step - loss: 0.5801 - accuracy: 0.7835 - val_loss: 0.5687 - val_accuracy: 0.7916
Epoch 8/10
329/329 [==============================] - 1s 3ms/step - loss: 0.5800 - accuracy: 0.7848 - val_loss: 0.5688 - val_accuracy: 0.7936
Epoch 9/10
329/329 [==============================] - 1s 3ms/step - loss: 0.5786 - accuracy: 0.7843 - val_loss: 0.5697 - val_accuracy: 0.7916
Epoch 10/10
329/329 [==============================] - 1s 3ms/step - loss: 0.5781 - accuracy: 0.7846 - val_loss: 0.5707 - val_accuracy: 0.7920
141/141 [==============================] - 0s 2ms/step - loss: 0.5707 - accuracy: 0.7920

Test accuracy:  0.7919999957084656
