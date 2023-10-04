# Healthcare-Analytics
Projects in healthcare data

## Diabetes Prediction project
In response to the escalating global diabetes epidemic, this project focuses on developing a predictive model using patient data. By analyzing treatment histories and physical features, our goal is to enhance diabetes risk assessment, early detection, and personalized treatment strategies. 
In this project, I am going to use the Diabetes dataset from the Vanderbilt Biostatistics Datasets to build Predictive models to classify the diagnosis of diabetes. Logistic Regression, Random Forest, and Gradient Boosting will be used to build the Binary Classification predictive tools. The dataset contains the following features/values.

<img src="nhgh_labels.png?" width="400" height="300"/>

After going through the dataset, I found that several columns contains 1.3 - 14.2% null values so I imputed the colummns by means, zero, and 'missing'.
'''
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
'''
