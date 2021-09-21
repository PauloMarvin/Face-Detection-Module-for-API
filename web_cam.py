import cv2
#carrega o classificador
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)


while True:
#carrega um exemplo de imagem
    _, img = cap.read()

    #coverte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #encontra faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)


    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('webcam',img)

    k =cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()

