  # -*- coding: cp1252 -*-
import numpy as np
import cv2
import os
#import trabalhoImagens

def carregaNomesASeremLidos(txt):
    listaNomeYoutubers = []
    pFile = open(txt, "r")
    for line in pFile:
        listaNomeYoutubers.append(line.rstrip())
    return listaNomeYoutubers

def criaPastaComNomes(listaNomes):
    for nome in listaNomes:
        try:
            print("criou " + nome)
            os.mkdir(nome)
        except OSError:
            print("Não foi possível criar o diretório ou o mesmo já existe.")

def salvaFacesDetectadas(nome):
    face_cascade = cv2.CascadeClassifier('C:\\Users\\JH\\AppData\\Local\\Programs\\Python\\Python38-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0) #inicia captura da câmera

    counterFrames = 0;
    while(counterFrames < 350): #quando chegar ao milésimo frame, para
        print(counterFrames)
        ret, img = cap.read()

        #frame não pode ser obtido? entao sair
        if(ret == False):
            cap.release()
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #se nenhuma face for achada, continue
        if not np.any(faces):
            continue

        #achou uma face? recorte ela (crop)
        rostoImg = img

        #salva imagem na pasta
        cv2.imwrite(nome + "/" + str(counterFrames)+".png", rostoImg)
        counterFrames += 1
            
    cap.release()
    cv2.waitKey(0)

#função principal da aplicação
def main():
    listaNomeYoutubers = carregaNomesASeremLidos("input.txt")
    criaPastaComNomes(listaNomeYoutubers)

    for nome in listaNomeYoutubers:
        print("Analisando: " + nome)
        salvaFacesDetectadas(nome)


if __name__ == "__main__":
    main()