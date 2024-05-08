import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data=pd.read_csv("Train.csv")
print(data.head())
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
data['Name']=label_encoder.fit_transform(data['Name'])
data['Occupation']=label_encoder.fit_transform(data['Occupation'])
data['Type_of_Loan']=label_encoder.fit_transform(data['Type_of_Loan'])
data['Payment_of_Min_Amount']=label_encoder.fit_transform(data['Payment_of_Min_Amount'])
data['Payment_Behaviour']=label_encoder.fit_transform(data['Payment_Behaviour'])
data['Credit_Mix']=data['Credit_Mix'].map({'Standard':1,'Good':2,'Bad':0})
data=data.drop(['ID','Customer_ID','Month','Name','Age','SSN','Occupation','Type_of_Loan','Changed_Credit_Limit',
                'Num_Credit_Inquiries','Credit_Utilization_Ratio','Payment_of_Min_Amount','Total_EMI_per_month',
                'Amount_invested_monthly','Payment_Behaviour'],axis=1)
x=data.drop('Credit_Score',axis=1)
y=data['Credit_Score']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
rf_cls= RandomForestClassifier()
rf_cls=rf_cls.fit(x_train,y_train)
y_pred_rf=rf_cls.predict(x_test)
print(y_pred_rf)
pickle.dump(rf_cls,open("model.pkl","wb"))