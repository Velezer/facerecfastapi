from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import face_recognition as fr
from setuptools.sandbox import save_modules
from helper import _dir, _dir_faces, _dir_encoded, compress_img, delete_file, get_pickled_images, list_files, save_pickle, save_file, encode_faces, read_pickle, classify_face
import time
from typing import List


app = FastAPI(debug=True)


@app.get("/")
def encode_all_images():
    start = time.perf_counter()

    server_images = list_files(_dir_faces, '.jpg')

    total_faces = len(server_images)
    number_encoded = total_faces
    for filename in server_images:
        encoded_faces = encode_faces('/'.join([_dir_faces, filename]))

        number_faces_detected = len(encoded_faces)
        if number_faces_detected > 1 or number_faces_detected < 1:
            number_encoded -= 1
            continue

        save_pickle('/'.join([_dir_encoded, filename]), encoded_faces[0])

    end = time.perf_counter()
    return {
        'status': 'success',
        'message': f'Encoded {number_encoded} of {total_faces} images',
        'response_time': end-start
    }


@app.post("/register/")
def register(name: str, file: UploadFile = File(...)):
    start = time.perf_counter()

    file.filename = '.'.join([name, 'jpg'])
    save_file(_dir_faces, file.filename, file.file)
    encoded_faces = encode_faces('/'.join([_dir_faces, file.filename]))

    number_faces_detected = len(encoded_faces)
    if number_faces_detected > 1:
        delete_file('/'.join([_dir_faces, file.filename]))
        return {
            'status': 'fail',
            'message': 'Detected more than one faces. Only support single face',
            'response_time': time.perf_counter()-start
        }
    if number_faces_detected < 1:
        delete_file('/'.join([_dir_faces, file.filename]))
        return {
            'status': 'fail',
            'message': 'No face detected',
            'response_time': time.perf_counter()-start
        }

    save_pickle(_dir_encoded, file.filename, encoded_faces[0])
    end = time.perf_counter()
    return {
        'status': 'success',
        'message': 'Image uploaded',
        "filename": file.filename,
        'response_time': end-start
    }


@app.post("/find/")
def find(excludes: List[str] = [], file: UploadFile = File(...)):
    start = time.perf_counter()

    server_images = list_files(_dir_encoded, '.jpg')
    encoded_faces = get_pickled_images(server_images)
    for x in excludes:
        encoded_faces.pop(x, None)

    save_file(_dir, 'unknowns.jpg', file.file)
    compress_img('unknowns.jpg', (200, 200), 30)
    unknowns = encode_faces('unknowns.jpg')
    number_faces_detected = len(unknowns)
    if number_faces_detected > 1:
        return {
            'status': 'fail',
            'message': 'Detected more than one faces. Only support single face',
            'response_time': time.perf_counter()-start
        }
    if number_faces_detected < 1:
        return {
            'status': 'fail',
            'message': 'No face detected',
            'response_time': time.perf_counter()-start
        }

    data = classify_face(unknowns, encoded_faces)
    data['excludes'] = excludes

    end = time.perf_counter()
    return {
        'status': 'success',
        'data': data,
        'response_time': end-start
    }
