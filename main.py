import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from functions import Smoothingfilters, findShapes, assemblingImages, morphologyOperations, figureBackgroundColor


#=====================================================================#

def nothing(x):
    pass

#=====================================================================#

cv2.namedWindow("CONTROL", cv2.WINDOW_NORMAL)
cv2.namedWindow("IMAGES", cv2.WINDOW_NORMAL)
# cv2.namedWindow("")


cv2.createTrackbar("Sm.FT", "CONTROL", 0, 4, nothing)
cv2.createTrackbar("Morph.FT", "CONTROL", 0, 7, nothing)
cv2.createTrackbar("Matrix.Num", "CONTROL", 0, 50, nothing)
#cv2.createTrackbar("Brightness", "CONTROL", 0, 255, nothing)
cv2.createTrackbar("Threshold", "CONTROL", 0, 255, nothing)
cv2.createTrackbar("Color", "CONTROL", 0, 2, nothing)
cv2.createTrackbar("Ar.Cross", "CONTROL", 2000, 10000, nothing)
cv2.createTrackbar("Ar.Square", "CONTROL", 2000, 10000, nothing)
cv2.createTrackbar("Ar.Circle", "CONTROL", 2000, 10000, nothing)


#=====================================================================#


while (True):

    #----------------------IMPORTANTO A IMAGEM E CONVERTENDO ELA PARA A ESCALA DE CINZA----------------------#
    
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/inputs/chosen/20220930_104444-heic_jpg.rf.ac9bd43453b950dd3457552c39b85e48.jpg")   # 3 4 11 255 24 1
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/inputs/chosen/20220930_105319_mp4-0_jpg.rf.fd8125ba9ff38a71ba25f2d14576cea3.jpg") # 1 4 19 255 117 1
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/inputs/chosen/20220930_103051-heic_jpg.rf.085f10e2e61b7d3e8e8eb10d099b7850.jpg")   # 1 3 11 255 43 2 // 2 4 2 255 51 2
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/inputs/chosen/20220930_104544-heic_jpg.rf.46ccadd421a3fbb9d7822cf1ab95168a.jpg")   # 3 4 19 255 0 1
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/inputs/chosen/IMG_20231201_133104323_jpg.rf.66b05d13d8eaa073f17c4c6d7ae07117.jpg") # nao é o melhor valor 1 2 2 255 42 1
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/inputs/chosen/20220930_103000-heic_jpg.rf.b44a35e7ecbd299bcb68e3f4989eaf4f.jpg")    # ainda nao tem 
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/inputs/Base_de_Takeoff.png")                                                          # 0 0 0 255 0 2 0 0 0
    sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/inputs/chosen/20220930_100330-heic_jpg.rf.cd762474c8ba772d242008004c2e1333.jpg") # 0 5 5 255 145 1 0 0 0 / 4 4 5 0 2 / 0 5 2 40 1 0 627 0
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/inputs/chosen/20220930_103031-heic_jpg.rf.64a4ed37cb175342d604fb8e89b8c498.jpg") # preciso alterar o valor do filtro ante-reflexo
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/inputs/chosen/20220930_104918-heic_jpg.rf.45c0bd59a930f86991c7d35894fdf8e7.jpg")
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/inputs/meus_inputs/7.new.jpg") # 3 0 0 87 2
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/inputs/meus_inputs/8.jpg") # 3 0 0 87 2 hard
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/novos_inputs/ex01.jpg") # 1 0 0 14 2 2000 2000 2000
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/novos_inputs/ex02.jpg") # 2 0 0 38 2 2000 2000 2000
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/novos_inputs/ex03.jpg") # 0 0 0 64 2 2000 2000 2000
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/novos_inputs/ex07.jpg") # nao dei
    #sorce_imagem = cv2.imread("C:/Users/adm/Documents/EDRA-OPENCV/novos_inputs/ex08.jpg") # 3 4 8 0 2 2000 2000 2000
    #sorce_imagem = cv2.resize(sorce_imagem,(640, 640))
    if sorce_imagem is None:
        sys.exit("ERROR: COULD NOT READ THE IMAGE")

    sorce_imagem_copy = sorce_imagem.copy()
    hsv_sorce_image = cv2.cvtColor(sorce_imagem, cv2.COLOR_BGR2HSV)
    #-------------------------------------------------------------------------------------------------------#


    #-----------------------------------FUNCIONAMENTO DOS SLIDERS--------------------------------------------#
    sliders1 = cv2.getTrackbarPos("Sm.FT", "CONTROL")           # responsável por aplicar os filtros de suavização
    sliders2 = cv2.getTrackbarPos("Morph.FT", "CONTROL")            # resposável por aplicar os filtros de morfologia
    sliders3 = cv2.getTrackbarPos("Matrix.Num", "CONTROL")          # resposável por aplicar o numero a dimensão da matriz do elemento estruturante
    #sliders4 = cv2.getTrackbarPos("Brightness", "CONTROL")          # resposável por aplicar brilho a imagem
    sliders5 = cv2.getTrackbarPos("Threshold", "CONTROL")           # resposável por aplicar o limiar na imagem
    sliders6 = cv2.getTrackbarPos("Color", "CONTROL")               # resposável por fazer com que somente a cor amarela ou azul seja identificada na imagem
    sliders7 = cv2.getTrackbarPos("Ar.Cross", "CONTROL")          # resposável por exibir as formas geométricas de acordo com a área definida
    sliders8 = cv2.getTrackbarPos("Ar.Square", "CONTROL")
    sliders9 = cv2.getTrackbarPos("Ar.Circle", "CONTROL")
    #--------------------------------------------------------------------------------------------------------#


    gray_image = figureBackgroundColor(sorce_imagem, hsv_sorce_image, sliders6)

                                                                    
    gray_image = Smoothingfilters(sliders1, gray_image)                      # APLICANDO FILTRO NA IMAGEM JÁ EM TONS DE CINZA, a imagem de indice 0 é imagem mofificada com texto quanto a de indice 1 é a
                                                                      #modificada sem texto de exibição

    # if sliders1  == 2:
    #     sliders4 = cv2.getTrackbarPos("Brightness", "CONTROL")

    gray_image = morphologyOperations(gray_image, sliders2, sliders3)


    #----------------------BINARIZANDO A IMAGEM PARA QUE POSSAMOS UTILIZAR O MÉTODO QUE ENCONTRA OS CONTORNOS----------------------

    _,binarized_image = cv2.threshold(gray_image, sliders5, 255, cv2.THRESH_BINARY) 

    #------------------------------------------------------------------------------------------------------------------------#

    #----------------------ENCONTRANDO OS CONTORNOS DA IMAGEM QUE FOI BINARIZADA LOGO ACIMA----------------------
    contours,hierarquia = cv2.findContours(binarized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(sorce_imagem, contours, -1, (0, 255, 0),2)

    #(contours, sorce_imagem, sliders7, sliders8, sliders9)
    gray_image_copy = gray_image.copy()
    # print(f"numero de dimensoes antes gray image copy--> {gray_image_copy.shape}")
    gray_image_copy = np.stack((gray_image_copy,)*3, axis=-1)
    # print(f"numero de dimensoes depois gray image copy--> {gray_image_copy.shape}")
    findShapes(contours, gray_image_copy, hierarquia, sliders7, sliders8, sliders9)
    
    # print(f"numero de dimensoes antes gray image --> {gray_image.shape}")
    # print(f"numero de dimensoes antes binarized_image --> {binarized_image.shape}")
    #print(f"numero de dimensoes antes gray image --> {gray_image.shape}")

    img_stack = assemblingImages(gray_image, binarized_image, gray_image_copy ,sorce_imagem_copy, sliders1, sliders2)

    # print()

    cv2.imshow("IMAGES", img_stack)
    
    #print("passei")
    cv2.waitKey(1000) #300
    print("oi")
