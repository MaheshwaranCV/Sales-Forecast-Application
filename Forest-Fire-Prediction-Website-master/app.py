from flask import Flask,request, url_for, redirect, render_template, session, flash
from flask_uploads import IMAGES, UploadSet, configure_uploads
import secrets
import os
import pickle
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from fileinput import filename
import warnings
warnings.filterwarnings("ignore")

def set_destination(app):
        return os.path.join(app.instance_path, "uploads")

app = Flask(__name__)
photos = UploadSet(name="photo", default_dest=set_destination)
app.config["UPLOADED_PHOTOS_DEST"] = "static/img"
app.config["SECRET_KEY"] = str(secrets.SystemRandom().getrandbits(128))
configure_uploads(app, photos)

@app.route('/')
def hello_world():
    return render_template("forest_fire.html")


@app.route('/upload',methods=['POST','GET'])
def upload():
    filename=photos.save(request.files["Recordset"])
    # FName={"fname":filename}
    pickle.dump(filename,open('upload.pkl','wb'))
    return render_template("forest_fire.html",pred=("Records uploaded successfully you can now predict the sales"))


@app.route('/predict',methods=['POST','GET'])
def predict():
    fname=pickle.load(open('upload.pkl','rb'))
    plotname=str("plot_"+(str(fname[:-4])+".png"))
    data = pd.read_csv(fname,index_col='DATE',parse_dates=True)
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
    plt.show()
    return render_template('forest_fire.html',pred=("Prediction successful"))

@app.route('/plot',methods=['POST','GET'])
def plot():
    g=pickle.load(open('predict.pkl','rb'))

@app.errorhandler(FileNotFoundError)
def handle_exception(e):
    return render_template('forest_fire.html',pred="File Not Found Error")

if __name__ == '__main__':
    app.run(debug=True)
