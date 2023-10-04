# Healthcare-Analytics
Projects in healthcare data

## Diabetes Prediction project
In response to the escalating global diabetes epidemic, this project focuses on developing a predictive model using patient data. By analyzing treatment histories and physical features, our goal is to enhance diabetes risk assessment, early detection, and personalized treatment strategies. 
In this project, I am going to use the Diabetes dataset from the Vanderbilt Biostatistics Datasets to build Predictive models to classify the diagnosis of diabetes. Logistic Regression, Random Forest, and Gradient Boosting will be used to build the predictive tools based on Binary Classification . The dataset contains the following features/values.

<img src="nhgh_labels.png?" width="400" height="300"/>

### Setup
After going through the dataset, I found that several columns contains 1.3 - 14.2% null values so I imputed the colummns by means, zero, and 'missing'.
```
#impute missing values with mean
#I will replace 'albumin', 'bun','Scr' with 0
#And 'income' with 'Missing'
columns_to_impute = ['income', 'leg', 'arml', 'armc','waist','tri','sub']
df[columns_to_impute] = df[columns_to_impute].fillna(df[columns_to_impute].mean())
columns_to_zero = ['albumin', 'bun','SCr']
df[columns_to_zero] = df[columns_to_zero].fillna(0)
df['income'] = df['income'].fillna('Missing')
print("\nDataFrame with Missing Values Imputed:")
print(df)
```
Since we are going to create a **Binary Classification** tool, we are going to create a predictive value named `diabetes`, where 0 means no diabetes and 1 means having diabetes.
According to CDC (https://www.cdc.gov/diabetes/managing/managing-blood-sugar/a1c.html), Glycohemoglobin level below 5.7 % is classified as 'Normal', between 5.7% and 6.4% is 'Prediabetes', and 6.5% or above is 'Diabetes'. Here, I am going to cerate a column 'diabetes' and classify gh 6.5 amd above as 1, below 6.4 as 0.

```
def classify_diabetes(gh_level):
    if gh_level >= 6.5:
        return 1
    else:
        return 0
    
df['diabetes'] = df['gh'].apply(classify_diabetes)
df[:5]
```
After creating the column, I checked the distribution of 0 and 1 of `diabetes` data and found a class imbalance between them, where no diabetes - 90.8% and with diabetes - 9.2%, as follows:
```
n_diabetes = df['diabetes'].value_counts()
percent_diabetes = (n_diabetes/n_diabetes.sum())*100
print(n_diabetes, percent_diabetes)
```

This issue will be addressed by oversampling in later section.

### Exploratory Data Analysis


### Building models
#### Feature Engineering
I performed the following feature engineering:
- Binary encoding for sex
- One-hot encoding for re and income

#### Checking the Feature Importance
I am checking the feature importances by Random Forest

#### Oversampling
As I mentioned in the first section, the data has a significant class imbalance where 90.8% have no diabetes while only 9.2% have diabetes. I am going to use oversampling to address this issue.
```
#Plotting after oversampling
# Resampling method
resampling = ADASYN(random_state=42, sampling_strategy=1.0)
y = y.astype('int32')
# Create the resampled feature set
X_resampled, y_resampled = resampling.fit_resample(X, y)
```

#### Building Models
I am going to use three models to predict the diabetes: Logistic regression, Random Forest, and XG Boost. I'm going to use Brier Score for the assessment of the model performance. Brier score is a metric used to assess the accuracy of probabilistic predictions made by a model. It typically ranges from 0 to 1, with lower values indicating better performance. A perfect model that makes accurate probabilistic predictions would have a Brier score of 0, whereas a completely random or uninformative model would have a Brier score of 0.25 for binary classification tasks (where there are two classes).

The formula:
<img src="brier_score.png?" width="200" height="150"/>
