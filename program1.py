import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn import preprocessing
df=pd.read_csv("train.csv")
#print(df.head())
#print(df.describe())
df.fillna(0,inplace=True)
df['LoanAmount'].fillna(0,inplace=True)
z=[]
for i in df['ApplicantIncome']:
    z.append(i)
#print(z)
plt.hist(z,bins=50,color="R")
plt.grid(True)
plt.xlabel("Income in Thousands")
plt.ylabel("People")
plt.show()
f=[]
for i in df['LoanAmount']:
        f.append(i)
#print(f)
plt.hist(f,bins=50)
plt.grid(True)
plt.xlabel("Loan Amount in Thousands")
plt.ylabel("People")
plt.show()

count,count2,k=0,0,0
for i in df["Loan_Status"]:
    if i=='Y':
        count+=1
    elif i=='N':
        count2+=1
    k+=1
y=[count,count2]
x=[0.25,0.5]
index=[0.25,0.5]
label=["Yes","No"]
#print(x)
#print(y)
width = 0.2
plt.bar(x,y,width,color="blue")
plt.xticks(index,label,fontsize=8, rotation=30)
plt.xlabel("Loan Status")
plt.ylabel("Total Applicants Count")
plt.show()

s=0
cd1,cd2,cd3,cd4=0,0,0,0
for i in df["Loan_Status"]:
    if i=='Y' and df['Credit_History'][s]==1:
        cd1+=1
    elif i=='Y' and df['Credit_History'][s]==0:
        cd2+=1
    elif i=='N' and df['Credit_History'][s]==1:
        cd3+=1
    else:
        cd4+=1
    s+=1
index=[0.25,0.5]
label=["Yes","No"]
plt.bar(x,[cd1,cd3],width)
plt.xticks(index,label,fontsize=8, rotation=30)
plt.xlabel("Loan Status")
plt.ylabel("Total Applicants Count")
plt.title("With Having Credit History")
plt.show()


index=[0.25,0.5]
label=["Yes","No"]
plt.bar(x,[cd2,cd4],width)
plt.xticks(index,label,fontsize=8, rotation=30)
plt.xlabel("Loan Status")
plt.ylabel("Total Applicants Count")
plt.title("Without Having Credit History")
plt.show()

k=[]
s=0
counter,counter2=0,0
counter1,counter3=0,0
for i in df["Education"]:
    if i=="Graduate" and df["Loan_Status"][s]=='Y':
        counter+=1
    elif i=="Graduate" and df["Loan_Status"][s]=='N':
        counter1+=1
    elif i=="Not Graduate" and df["Loan_Status"][s]=='Y':
        counter2+=1
    else:
        counter3+=1
    s+=1
        
index=[0.25,0.5]
label=["Yes","No"]
plt.bar(x,[counter,counter1],width)
plt.xticks(index,label,fontsize=8, rotation=30)
plt.xlabel("Loan Status")
plt.ylabel("Total Applicants Count")
plt.title("With Having Graduation")
plt.show()


index=[0.25,0.5]
label=["Yes","No"]
plt.bar(x,[counter2,counter3],width)
plt.xticks(index,label,fontsize=8, rotation=30)
plt.xlabel("Loan Status")
plt.ylabel("Total Applicants Count")
plt.title("Without Having Graduation")
plt.show()    


count1,count2=0,0
count3,count4=0,0
s=0
for i in df["Self_Employed"]:
    if i=="No" and df["Loan_Status"][s]=='Y':
        count1+=1
    elif i=='No' and df["Loan_Status"][s]=='N':
        count2+=1
    elif i=="Yes" and df["Loan_Status"][s]=="Y":
        count3+=1
    else:
        count4+=1
    s+=1
    
index=[0.25,0.5]
label=["Yes","No"]
plt.bar(x,[count1,count2],width)
plt.xticks(index,label,fontsize=8, rotation=30)
plt.xlabel("Loan Status")
plt.ylabel("Total Applicants Count")
plt.title("Self Employed")
plt.show()


index=[0.25,0.5]
label=["Yes","No"]
plt.bar(x,[count3,count4],width)
plt.xticks(index,label,fontsize=8, rotation=30)
plt.xlabel("Loan Status")
plt.ylabel("Total Applicants Count")
plt.title("Not Self Employed")
plt.show()

var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = preprocessing.LabelEncoder()
s=0
df.replace(1, "Graduate")
df.replace(0, "Not Graduate")
        


def classification_model(model, data, predictors, outcome):
  model.fit(data[predictors],data[outcome])
  predictions = model.predict(data[predictors])
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %.3f"%float(accuracy))
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    train_predictors = (data[predictors].iloc[train,:])
    train_target = data[outcome].iloc[train]
    model.fit(train_predictors, train_target)
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
  print("Cross-Validation Score : %.3f"%np.mean(error))
  model.fit(data[predictors],data[outcome]) 

outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, df,predictor_var,outcome_var)
