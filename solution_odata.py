import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data=pd.read_csv('odata.csv')
print(data.describe())
# data=data.replace('unlikely',0)
# data=data.replace('somewhat likely',1)
# data=data.replace('very likely',2)
print(data.corr())
x=data[['pared','gpa']]
y=data['apply']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=False)
lr=LogisticRegression(solver='lbfgs',multi_class='auto')
lr.fit(x_train,y_train)
predi=lr.predict(x_test)
y=lr.score(x_test,y_test)
print('model accuracy is = ',round(y,2))
confusion_matrix = confusion_matrix(y_test,predi)
print ('confusion_matrix\n',confusion_matrix)

