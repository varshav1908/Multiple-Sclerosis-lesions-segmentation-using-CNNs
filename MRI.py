
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import imutils
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn import metrics
import ftplib
from tkinter import ttk

main = tkinter.Tk()
main.title(" Multiple Sclerosis Lesions Segmentation Using Attention-Based CNNs in FLAIR Images") #designing main screen
main.geometry("1300x1200")

global filename
global accuracy
X = []
Y = []
global classifier
disease = ['No Tumor Detected','Tumor Detected']

def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");



def generateModel():
    global X
    global Y
    X.clear()
    Y.clear()
    if os.path.exists('Model/myimg_data.txt.npy'):
        X = np.load('Model/myimg_data.txt.npy')
        Y = np.load('Model/myimg_label.txt.npy')
    else:
        for root, dirs, directory in os.walk(filename+"/no"):
            for i in range(len(directory)):
                name = directory[i]
                img = cv2.imread(filename+"/no/"+name,0)
                ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                img = cv2.resize(img, (128,128))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(128,128,1)
                X.append(im2arr)
                Y.append(0)
                print(filename+"/no/"+name)

        for root, dirs, directory in os.walk(filename+"/yes"):
            for i in range(len(directory)):
                name = directory[i]
                img = cv2.imread(filename+"/yes/"+name,0)
                ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                img = cv2.resize(img, (128,128))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(128,128,1)
                X.append(im2arr)
                Y.append(1)
                print(filename+"/yes/"+name)
                
        X = np.asarray(X)
        Y = np.asarray(Y)            
        np.save("Model/myimg_data.txt",X)
        np.save("Model/myimg_label.txt",Y)
    print(X.shape)
    print(Y.shape)
    print(Y)
    cv2.imshow('ss',X[20])
    cv2.waitKey(0)
    text.insert(END,"Total number of images found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total number of classes : "+str(len(set(Y)))+"\n\n")
           
        
 
def CNN():
    global accuracy
    global classifier
    
    YY = to_categorical(Y)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    x_train = X[indices]
    y_train = YY[indices]

    if os.path.exists('Model/model.json'):
        with open('Model/model.json', "r") as json_file:
           loaded_model_json = json_file.read()
           classifier = model_from_json(loaded_model_json)

        classifier.load_weights("Model/model_weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('Model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,'\n\nCNN Model Generated. See black console to view layers of CNN\n\n')
        text.insert(END,"CNN Prediction Accuracy on Test Images : "+str(accuracy)+"\n")
    else:
        X_trains, X_tests, y_trains, y_tests = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
        classifier = Sequential() #alexnet transfer learning code here
        classifier.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 1), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 128, activation = 'relu'))
        classifier.add(Dense(output_dim = 2, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(x_train, y_train, batch_size=16, epochs=10,validation_split=0.2, shuffle=True, verbose=2)
        classifier.save_weights('Model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("Model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('Model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('Model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,'\n\nCNN Model Generated. See black console to view layers of CNN\n\n')
        text.insert(END,"CNN Prediction Accuracy on Test Images : "+str(accuracy)+"\n")
       


def predict():
    '''ftp = ftplib.FTP_TLS("ftp.drivehq.com")
    ftp.login("cloudfilestorageacademic", "Offenburg965#")
    ftp.prot_p()
    name = imagelist.get()
    with open('drivehq.jpg', 'wb' ) as file :
        ftp.retrbinary('RETR %s' % name, file.write)
    file.close()'''

    
    img = cv2.imread('drivehq.jpg',0)
    img = cv2.resize(img, (128,128))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,128,128,1)
    XX = np.asarray(im2arr)
        
    predicts = classifier.predict(XX)
    print(predicts)
    cls = np.argmax(predicts)
    print(cls)
    img = cv2.imread('drivehq.jpg')
    img = cv2.resize(img, (800,500))
    cv2.putText(img, 'Disease Identified as : '+disease[cls], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow('Disease Identified as : '+disease[cls], img)
    cv2.waitKey(0)

def getImages():
   ''' ftp = ftplib.FTP_TLS("ftp.drivehq.com")
    ftp.login("cloudfilestorageacademic", "Offenburg965#")
    ftp.prot_p()
    filenames = ftp.nlst()
    value.clear()
    for filename in filenames:
        value.append(filename)'''

font = ('times', 16, 'bold')
title = Label(main, text=' Multiple Sclerosis Lesions Segmentation Using Attention-Based CNNs in FLAIR Images')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload MRI Images Dataset", command=upload)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

modelButton = Button(main, text="Generate Images Train & Test Model (OSTU Features)", command=generateModel)
modelButton.place(x=290,y=550)
modelButton.config(font=font1) 

cnnButton = Button(main, text="Generate Deep Learning CNN Model", command=CNN)
cnnButton.place(x=710,y=550)
cnnButton.config(font=font1) 

imageButton = Button(main, text="Get DriveHQ Images", command=getImages)
imageButton.place(x=50,y=600)
imageButton.config(font=font1)

value = ["DriveHQ Images"]
imagelist = ttk.Combobox(main,values=value,postcommand=lambda: imagelist.configure(values=value)) 
imagelist.place(x=240,y=600)
imagelist.current(0)
imagelist.config(font=font1)

predictButton = Button(main, text="Predict Tumor", command=predict)
predictButton.place(x=440,y=600)
predictButton.config(font=font1)



main.config(bg='turquoise')
main.mainloop()
