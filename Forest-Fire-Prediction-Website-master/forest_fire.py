#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv('Alcohol_Sales.csv',index_col='DATE',parse_dates=True)
data.index.freq = 'MS'
data.columns = ['Sales']
data['Sale_LastMonth']=data['Sales'].shift(+1)
data['Sale_2Monthsback']=data['Sales'].shift(+2)
data['Sale_3Monthsback']=data['Sales'].shift(+3)
data=data.dropna()
lin_model=LinearRegression()
model=RandomForestRegressor(n_estimators=100,max_features=3, random_state=1)
x1,x2,x3,y=data['Sale_LastMonth'],data['Sale_2Monthsback'],data['Sale_3Monthsback'],data['Sales']
x1,x2,x3,y=np.array(x1),np.array(x2),np.array(x3),np.array(y)
x1,x2,x3,y=x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1),y.reshape(-1,1)
final_x=np.concatenate((x1,x2,x3),axis=1)
X_train,X_test,y_train,y_test=final_x[:-30],final_x[-30:],y[:-30],y[-30:]
model.fit(X_train,y_train)
lin_model.fit(X_train,y_train)
pred=model.predict(X_test)
plt.rcParams["figure.figsize"] = (12,8)
plt.plot(pred,label='Random_Forest_Predictions')
plt.plot(y_test,label='Actual Sales')
plt.legend(loc="upper left")
plt.savefig("plot.png")

# X = data[1:, 1:-1]
# y = data[1:, -1]
# y = y.astype('int')
# X = X.astype('int')
# # print(X,y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# log_reg = LogisticRegression()


# log_reg.fit(X_train, y_train)

# inputt=[int(x) for x in "45 32 60".split(' ')]
# final=[np.array(inputt)]

# b = log_reg.predict_proba(final)


pickle.dump(lin_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


