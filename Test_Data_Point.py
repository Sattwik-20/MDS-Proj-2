import numpy as np
import pandas as pd
import numpy as np
import scipy as sp
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv("/content/data_set.csv", names = headers)
df

df.replace("?", np.nan, inplace = True)
df.head()

missing_data = df.isnull().sum()
missing_data.sort_values(inplace=True, ascending=False)
missing_data

avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses: ", avg_norm_loss)

avg_bore = df['bore'].astype('float').mean(axis=0)
print("Average of bore: ", avg_bore)

avg_stroke = df["stroke"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_stroke)

avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)

avg_peakrpm = df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)

avg_price = df['price'].astype('float').mean(axis=0)
print("Average price:", avg_price)

df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
df["stroke"].replace(np.nan, avg_stroke, inplace = True)
df["bore"].replace(np.nan, avg_bore, inplace=True)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

df['num-of-doors'].value_counts()
df['num-of-doors'].value_counts().idxmax()
df["num-of-doors"].replace(np.nan, "four", inplace=True)
df['price'].replace(np.nan, avg_price, inplace=True)
df.dtypes

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df['length'] = df['length'] / df['length'].max()
df['width'] = df['width'] / df['width'].max()
df['height'] = df['height'] / df['height'].max()

le = preprocessing.LabelEncoder()
le.fit(df['make'])
en=le.transform(df['make'])
df['make']=en

head = ["fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","engine-type",
         "num-of-cylinders","fuel-system"]
for i in head:
    le = preprocessing.LabelEncoder()
    le.fit(df[i])
    en=le.transform(df[i])
    df[i]=en
df.to_csv('cleaned_final.csv')
df = pd.read_csv("cleaned_final.csv")
df = df.drop(['Unnamed: 0'],axis=1)
df.head()
#Obtaining the to_predict variable
target = df.iloc[:,0]
print("Shape of Target",target.shape)
#Now we extracted the features from our dataframe
features = df.iloc[:,1:]
print("Shape of Features",features.shape)

#Normalizing the given data
sclr=StandardScaler()
features = sclr.fit_transform(features)

#Converting the target variable into an encoded vector
y_en=tf.keras.utils.to_categorical(target, num_classes=6)
print("Y_en.shape",y_en.shape)

#Split data into train and test to check for losses separately
X_train, X_test, y_train, y_test = train_test_split(features,y_en,test_size=0.20,random_state=42)

#Model Definition
model=Sequential()
model.add(Dense(20,input_shape=(X_train.shape[1],), activation="relu" ))
model.add(Dense(6,activation="sigmoid"))
model.summary()

#Using categorical_cross-entropy loss function
model.compile(loss="categorical_crossentropy",optimizer="RMSProp",metrics=["accuracy"])
history = model.fit(X_train,y_train,validation_data=(X_test,y_test), batch_size=51, epochs= 1500, verbose=1)


#Finding the accuracy from the model
accuracy=history.history['accuracy']
validation_accuracy=history.history['val_accuracy']

#Finding the losses from the model
loss=history.history['loss']
validation_loss=history.history['val_loss']
epochs_range=range(1500)


#Plotting training and validation losses and accuracy
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(epochs_range,accuracy,label='Training Accuracy')
plt.plot(epochs_range,validation_accuracy,label='Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range,loss,label='Training Loss')
plt.plot(epochs_range,validation_loss,label='Validation Loss')
plt.legend()
plt.show()





index=[]
trans=[]
head = ["make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","engine-type",
         "num-of-cylinders","fuel-system"]
for i in head:
    index.append(df.columns.get_loc(i))
    le = preprocessing.LabelEncoder()
    le.fit(df1[i])
    trans.append(le)

test_data_pt = np.array([122,'volkswagen','gas','std','two','sedan','fwd','front',97.30,171.70,65.50,55.70,2209,'ohc','four',109,'mpfi',3.19,3.40,9.00,85,5250,27,34,7975])
test_data_pt1 = np.array([122,'volkswagen','gas','std','two','sedan','fwd','front',97.30,171.70,65.50,55.70,2209,'ohc','four',109,'mpfi',3.19,3.40,9.00,85,5250,27,34,7975])
test_data_pt==test_data_pt1

for i in range(len(trans)):
    a=index[i]
    a=a-1
    test_data_pt[a]=trans[i].transform([test_data_pt[a]])[0]

test_dt = test_data_pt
test_dt = test_dt[np.newaxis,:]
test_dt = scaler.transform(test_dt)
mp=model.predict(test_dt)
print(mp)
print('symboling value of the test point is',np.argmax(mp)-2)
