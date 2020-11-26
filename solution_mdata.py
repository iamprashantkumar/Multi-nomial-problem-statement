import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


data=pd.read_csv('mdata.csv')
# print(data.corr())
# data=data.replace('vocation',0)
# data=data.replace('academic',1)
# data=data.replace('general',2)
# print(data)
# x=data[['read','write','math','science']]
# dummies=pd.get_dummies(data.ses)
# print(dummies)
# merged=pd.concat([data,dummies],axis='columns')
# print(merged.corr())
# data=merged.drop(['ses','middle'],axis=1)
print(data.corr())

x=data[['math','science']]
# x=data.drop(['female','schtyp','prog','honors'],axis=1)
# print(x)
y=data.prog
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=False)
lr=LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=500)
lr.fit(x_train,y_train)
predi=lr.predict(x_test)
y=lr.score(x_test,y_test)
print('model accuracy is = ',round(y,2))
confusion_matrix = confusion_matrix(y_test,predi)
print ('confusion_matrix\n',confusion_matrix)


