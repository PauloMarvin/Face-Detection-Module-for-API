import cv2
from face_recognition import Face_recognition
indentificador = Face_recognition


#carrega o classificador
face_cascade = Face_recognition.load_classifier( classifier_path='cascades/haarcascade_frontalface_default.xml')


#carrega um exemplo de imagem
#img = cv2.imread('cascades/WhatsApp Image 2021-09-18 at 22.23.27 (1).jpeg')
img = Face_recognition.load_image(image_path='cascades/people-persons-peoples.jpg')


#imagem 600x800 3 canais (RGB)
print(img.shape)

#coverte para escala de cinza
gray = indentificador.cvt_gray(img)

gray = cv2.resize(gray,(800,600))

#encontra faces
faces = indentificador.face_in_imagens(gray,face_cascade, 1.1, 4)

print(faces)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('img',img)
cv2.waitKey()


