from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form
from fastapi.responses import HTMLResponse
import face_recognition as fr
from setuptools.sandbox import save_modules
from helper import _dir, _dir_faces, _dir_encoded, compress_img, delete_file, get_pickled_images, list_files, save_pickle, save_file, encode_faces, read_pickle, classify_face
import time
from typing import List
import filetype


app = FastAPI(debug=True)


# @app.get("/")
# def encode_all_images():
#     start = time.perf_counter()

#     server_images = list_files(_dir_faces, '.jpg')

#     total_faces = len(server_images)
#     number_encoded = total_faces
#     for filename in server_images:
#         encoded_faces = encode_faces('/'.join([_dir_faces, filename]))

#         number_faces_detected = len(encoded_faces)
#         if number_faces_detected > 1 or number_faces_detected < 1:
#             number_encoded -= 1
#             continue

#         save_pickle('/'.join([_dir_encoded, filename]), encoded_faces[0])

#     end = time.perf_counter()
#     return {
#         'status': 'success',
#         'message': f'Encoded {number_encoded} of {total_faces} images',
#         'response_time': end-start
#     }


@app.post("/register/", status_code=status.HTTP_201_CREATED)
def register(name: str = Form(...), file: UploadFile = File(...)):
    start = time.perf_counter()

    file.filename = '.'.join([name, 'jpg'])
    save_file(_dir_faces, file.filename, file.file)
    if not filetype.is_image('/'.join([_dir_faces, file.filename])):
        delete_file('/'.join([_dir_faces, file.filename]))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail='This is not an image')

    compress_img('/'.join([_dir_faces, file.filename]), (320, 240), 30)
    encoded_faces = encode_faces('/'.join([_dir_faces, file.filename]))
    # delete_file('/'.join([_dir_encoded, file.filename]))

    number_faces_detected = len(encoded_faces)
    if number_faces_detected == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail='No face detected')
        
    if number_faces_detected > 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail='Detected more than one faces. Only support single face')

    save_pickle(_dir_encoded, file.filename, encoded_faces[0])
    end = time.perf_counter()
    return {
        'status': 'success',
        'detail': 'Success registering face',
        "filename": file.filename,
        'response_time': end-start
    }


@ app.post("/find/")
def find(file: UploadFile = File(...)):
    start = time.perf_counter()

    server_images = list_files(_dir_encoded, '.jpg')
    encoded_faces = get_pickled_images(server_images)

    save_file(_dir, 'unknowns.jpg', file.file)
    compress_img('unknowns.jpg', (200, 200), 30)
    unknowns = encode_faces('unknowns.jpg')

    number_faces_detected = len(unknowns)
    if number_faces_detected == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail='No face detected')

    if number_faces_detected > 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail='Detected more than one faces. Only support single face')

    data = classify_face(unknowns, encoded_faces)

    end = time.perf_counter()
    return {
        'status': 'success',
        'data': data,
        'response_time': end-start
    }
