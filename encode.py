from os.path import splitext
import cv2
import face_recognition
import pickle
import os


imgfolder='images'
imgpath=os.listdir(imgfolder)
print(imgpath)
studname=[]
imgmo=[]
for path2 in imgpath:
    imgmo.append(cv2.imread(os.path.join(imgfolder,path2)))
    studname.append(splitext(path2)[0])
print(len(imgmo))
print(studname)

def findencode(imageslist):
    encodeList=[]
    for img in imageslist:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

print("encoding started >>>>")
encodelistknown=findencode(imgmo)
encodelistknownname=[encodelistknown,studname]
print("encoding completed")

file =open("encodefile.p",'wb')
pickle.dump(encodelistknownname,file)
file.close()
