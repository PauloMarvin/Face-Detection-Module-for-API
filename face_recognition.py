import cv2


class Face_recognition:

    def __init__(self):
        self.classifier = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        self.image_path = "cascades/people-persons-peoples.jpg"

    def load_classifier(classifier_path: str):
        new_classifier = cv2.CascadeClassifier(classifier_path)
        return new_classifier


    def load_image(image_path: str):
        new_image = cv2.imread(image_path)

        return new_image

    def cvt_gray(image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image_gray

    def face_in_imagens(image,classifier,scaleFactor = 1.05,minNeighbors = 4):
        faces = classifier.detectMultiScale(image, scaleFactor, minNeighbors)
        return faces


