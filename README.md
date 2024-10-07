# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
### Date:
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program.

Step 2: import pandas module and import the required data set.

Step 3: Find the null values and count them.

Step 4: Count number of left values.

Step 5: From sklearn import LabelEncoder to convert string values to numerical values.

Step 6: From sklearn.model_selection import train_test_split.

Step 7: Assign the train dataset and test dataset.

Step 8: From sklearn.tree import DecisionTreeClassifier.

Step 9: Use criteria as entropy.

Step 10: From sklearn import metrics.

Step 11: Find the accuracy of our model and predict the require values.

Step 12: Stop the program.
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Naveenaa A K 
RegisterNumber:  212222230094
```
```
import pandas as pd
data=pd.read_csv("Exp_8_Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])


```

## Output:
![image](https://github.com/user-attachments/assets/bf70d585-1811-4761-8df9-cb0243e89b18)
### info :
![image](https://github.com/user-attachments/assets/30532818-50ed-4dd5-9f5c-e919cd50789c)
### checking  for null values :
![image](https://github.com/user-attachments/assets/e7556717-184a-4778-9f1c-56de6929b877)

![image](https://github.com/user-attachments/assets/eed04dbd-d048-4536-a65e-4e2d2766f5ea)
### Accuracy:
![image](https://github.com/user-attachments/assets/2a490fc8-8ac1-4e52-9edf-22c2916b23e6)

### Predict:
![image](https://github.com/user-attachments/assets/9cffb877-80f7-4854-a834-b98d779623da)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
