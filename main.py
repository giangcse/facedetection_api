import cv2
from fastapi import responses
import numpy as np
import os
from starlette.requests import Request
from starlette.responses import RedirectResponse
import uvicorn
import shutil

from fastapi import FastAPI, File, UploadFile
from fastapi.params import Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Create an instance of FastAPI
app = FastAPI()
# Create template
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def UploadImage(file: UploadFile = File(...)):
    try:
        if file.content_type == 'image/jpeg':
            with open(f'{file.filename}', 'wb') as image:
                shutil.copyfileobj(file.file, image)
            detection(f'{file.filename}')
            shutil.move(f'{file.filename}',
                        f'{os.getcwd()}/data/{file.filename}')
            return FileResponse(f'{os.getcwd()}/data/{file.filename}')
        else:
            return {'error': 'file is not image'}
    finally:
        # os.remove(f'{os.getcwd()}\\static\\{file.filename}')
        pass


def detection(image):
    # Path to xml file
    face_cascade = cv2.CascadeClassifier(os.path.join(
        os.getcwd(), 'haarcascade_frontalface_default.xml'))
    # Read image and convert to gray
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    # Save image
    cv2.imwrite(f'{image}', img)
    # Return number of people
    # return len(faces)


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=80)
