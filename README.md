# Alphabet Soup Success Prediction Analysis Report

## Overview of the Analysis

In this analysis, I evaluate a model that can help select the applicants for funding with the best 
chance of success in their ventures. I used a dataset (csv) of historical and current organizations 
(more than 34,000 ) that have received funding from the non-profit foundation Alphabet Soup.
<br>
To determine chance of success, I looked at the other features in the current foundation dataset 
to train the model to predict an applicant's potential success rate using actual clients' current
success statuses (successful vs unsuccessful) based on the target column "IS_SUCCESSFUL".
<br>
The variables ("features") used to train the model are listed below:
* APPLICATION_TYPE—Alphabet Soup application type 
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* INCOME_AMT—Income classification
* ASK_AMT—Funding amount requested
<br>
<br>
Below is a brief overview of the stages of the machine learning process I went through as part of this 
analysis: [EDIT BELOW]
<br>
1. Loading dataset (csv) into analysis tool of choice (in this case python using libraries like pandas
and scikit learn). 
<br>
2. Prprocess Data (to be easier for the model to train with/"digest"):
<br>a. Drop the non-beneficial columns (ex. ID columns "EIN" and "NAME"; classification columns 
"SPECIAL_CONSIDERATIONS" and "STATUS" due to vast majority of data being one of each) to reduce noise for
model.
<br>b. Check number of unique values for each column/feature, determine the number of data points for each 
unique value for columns with more than 10 unqiue values, and group rare occurances for each feature into
"Other" bin to reduce data noise for the model. I used  number of data points for each unique value to pick 
a cutoff point to combine "rare" categorical variables. This was done for features "APPLICATION_TYPE" and 
"CLASSIFICATION".
<br>c. Encode categorical variables (using pd.get_dummies()) into 0/1 (AKA true/false) so that the model 
can read them.
<br>
3. Split the Data into Training and Testing Sets - this allows you to use a part of the real data
to train the model and the other to test whether the model is predicting chance of success in a satisfactory 
manner. I used train_test_split from scikit-learn for this step, amongst other functions.
<br>
4. Scale the data (using StandardScaler()) for better model readability. This is done to normalize features
so that each feature contributes equally to the calculations in the model (AKA so difference in scale does not
overcontribute/overpredict only due to scale). 
<br>
5. Define neural net model (outline how many layers, neurons in each layer), outline that model should
also tell us accuracy and loss (amount of the data that the model could not classify) for each training 
round (epoch) with the training dataset, and train the model with the training dataset we created from 
original data (we set how many epochs we want to use to train the model here as well).
<br>
6. Evaluate the model’s performance using the testing data set (created from the original dataset) by 
looking at the accuracy and loss from this test.
<br>
<br>

## Results

* **Selected Neural Network Model (Model 5):**
    * Test Acc: 0.7268 (72.68%)
    * Test Loss: 0.5548 (55.48%)

After creating the original base model, I optimized the model by experimenting with various factors, 
such as removing more potential noisy columns, attempting to bin the ASK_AMT column to reduce noise 
(ultimately undid this as this created more features for the model once scaled and reduced performance), 
increasing and decreasing epochs run, and increasing and decreasing layers and neurons for each layer. 
Utlimately, I selected a model that was comprised of the following factors:
<br> 
* Input layer with 10 neurons and a tanh activation function.
* 1 hidden layer with 5 neurons and tanh activation function.
* Output layer with 1 neuron and sigmoid activation function.
* 20 epochs.
<br>
This model was selected as it used less computational resources (less epochs, layers, and neurons) than some 
of the others I tried, had slightly better accuracy, and similar loss. It was not able to reach the target of 
75% accuracy.

## Summary
In summary, I found that **Model 5** was the best one for this use. This model kept a slightly higher 
accuracy score for predicted success seen by the other models but also used less computational resources. I 
would still reccommend human supervison with making the final decision due to the high loss rate for the model
and room for improvmeent in the accuracy of the predictions. Ideally work to better optimize this model (or
creation of a different model) could be done to improve both the accuracy and loss of the chosen model. 
Potential work could be continued in preprocessing the data to reduce what features the model has to work with
(ex. PCA or an analagous analysis for neural networks) to see what features may be most impactful 
in predicting success for a venture in order to get more accurate prediction and less loss. We could also do
further work in optimizing the model by utilizing keras-tuner to auto-optimize the model.
