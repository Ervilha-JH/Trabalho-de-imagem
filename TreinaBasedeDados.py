import cv2
import os
import numpy as np
import faceRecognition as fr


#Foto carregada para reconhecimento final
test_img=cv2.imread('Fotos de identificacao/Matheus.png')#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
#Descomente daqui para baixo para refazer o treino
    #Pega a pasta onde está as fotos de treinamento e cria os Dados de treino na YML
    #faces,faceID=fr.labels_for_training_data('Imagens do Treino')
    #face_recognizer=fr.train_classifier(faces,faceID)
    #face_recognizer.write('trainingData.yml')


#Comente aqui para não fazer o load de treino.
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#Pega a base de dados pronta
#Comente até aqui caso queira fazer outro treino

name={0:"Henrique",1:"Carlos",2:"Gabriel",3:"Matheus"}#Lista de nomes de quem foi treinado.

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("Detector de faces",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows




