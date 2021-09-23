from detectors.detection_smile import DetectSmile
from detectors.detection_eyes import DetectEyes
from detectors.detection_face import DetectFace
from typing import Dict, List
from utils.base64 import base64_to_nparray
from fastapi import FastAPI

from pydantic import BaseModel

from utils.decorators import timing

app = FastAPI()

class Request_body(BaseModel):
    img : str


@timing
@app.post(path="/detect_face")
def detect(body:Request_body) -> Dict[str, List[List[int]]]:

    detect_face = DetectFace()

    #carregamento da imagem
    img = body.img
    img = base64_to_nparray(img)

    #pre processamento
    gray_image = detect_face.bgr_to_gray(img)

    response =detect_face.detect(gray_image)

    return {"faces": response}

@app.post(path="/detect_smile")
def detect(body:Request_body) -> Dict[str, List[List[int]]]:

    detect_smile = DetectSmile()

    #carregamento da imagem
    img = base64_to_nparray(body.img)

    #pre processamento
    gray_image = detect_smile.bgr_to_gray(img)

    response =detect_smile.detect(gray_image)

    return {"faces": response}


@app.post(path="/detect_eyes")
def detect(body:Request_body) -> Dict[str, List[List[int]]]:

    detect_eyes = DetectEyes()

    #carregamento da imagem
    img = base64_to_nparray(body.img)

    #pre processamento
    gray_image = detect_eyes.bgr_to_gray(img)

    response =detect_eyes.detect(gray_image)

    return {"eyes": response}