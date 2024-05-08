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
data['Credit_Mix']=label_encoder.fit_transform(data['Credit_Mix'])
x=np.array(data[['Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts',
                   'Num_Credit_Card','Interest_Rate','Num_of_Loan',
                   'Delay_from_due_date','Num_of_Delayed_Payment','Credit_Mix',
                   'Outstanding_Debt','Credit_History_Age','Monthly_Balance']])
y=data['Credit_Score']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
rf_cls= RandomForestClassifier()
rf_cls=rf_cls.fit(x_train,y_train)
y_pred_rf=rf_cls.predict(x_test)
print(y_pred_rf)
pickle.dump(rf_cls,open("model.pkl","wb"))