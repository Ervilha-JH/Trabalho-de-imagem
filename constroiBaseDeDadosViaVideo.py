  # -*- coding: cp1252 -*-
import numpy as np
import cv2
import os

def carregaNomesASeremLidos(txt):
    ListaFotosPessoa = []
    pFile = open(txt, "r")
    for line in pFile:
        ListaFotosPessoa.append(line.rstrip())
    return  ListaFotosPessoa

def criaPastaComNomes(listaNomes):
    for nome in listaNomes:
        try:
            print("criou " + nome)
            os.mkdir(nome)
        except OSError:
            print("Não foi possível criar o diretório ou o mesmo já existe.")

def salvaFacesDetectadas(nome):
    face_cascade = cv2.CascadeClassifier('C:\\Users\\JH\\AppData\\Local\\Programs\\Python\\Python38-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture('D:\Imagens\Sabino.mp4') #inicia captura frame a frame da pessoa

    counterFrames = 0;
    while(counterFrames < 100): #quando chegar ao milésimo frame, para
        print(counterFrames)
        ret, img = cap.read()

        #frame não pode ser obtido? entao sair
        if(ret == False):
            cap.release()
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #salva imagem na pasta
        cv2.imwrite(nome + "/" + str(counterFrames)+".png", img)
        counterFrames += 1
            
    cap.release()
    cv2.waitKey(0)

#função principal da aplicação
def main():
    ListaFotosPessoa = carregaNomesASeremLidos("input.txt")
    criaPastaComNomes(ListaFotosPessoa)

    for nome in ListaFotosPessoa:
        print("Analisando: " + nome)
        salvaFacesDetectadas(nome)


if __name__ == "__main__":
    main()