import cv2

#carrega o classificador
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#carrega um exemplo de imagem
img = cv2.imread('people-persons-peoples.jpg')

#imagem 600x800 3 canais (RGB)
print(img.shape)

#coverte para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)

#encontra faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

print(faces)




for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('img', img)
cv2.waitKey()

