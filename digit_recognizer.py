import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#imported classifier from sklearn

from sklearn.neighbors import KNeighborsClassifier

#taking data set from location

dataTrain = pd.read_csv("D:\ml_dataset/train.csv").to_numpy()
dataTest=pd.read_csv("D:\ml_dataset/test.csv").to_numpy()
clf1=KNeighborsClassifier()

#creating a trainig dataset

xTrain=dataTrain[0:,1:]
train_label=dataTrain[0:,0]
clf1.fit(xTrain,train_label)

#creating a test data set

xTest=dataTest[0:,0:]

#predicting the result using classifier

pr=clf1.predict(xTest)

m=28000
dict_={"ImageId": np.array(range(1,m+1)),"label":pr}
Submission=pd.DataFrame(dict_)
Submission.to_csv("D:\ml_dataset/Submission.csv",index=False)



