import os
import shutil
import pickle
from typing import Dict, List, Tuple
import face_recognition as fr
import numpy as np
from PIL import Image


_dir = os.path.dirname(__file__)
_dir_faces = '/'.join([_dir, 'faces'])
_dir_encoded = '/'.join([_dir, 'encoded'])


def save_file(directory: str, filename: str, content: bytes):
    with open('/'.join([directory, filename]), 'wb') as buffer:
        shutil.copyfileobj(content, buffer)


def save_pickle(directory: str, filename: str, content):
    with open('/'.join([directory, filename]), 'wb') as f:
        pickle.dump(content, f)


def read_pickle(filename: str):
    with open(filename, 'rb') as f:
        loaded = pickle.load(f)
        return loaded


def delete_file(filename: str):
    if os.path.exists(filename):
        os.remove(filename)


def list_files(directory: str, extension: str = '.jpg') -> List:
    for _, _, fnames in os.walk(directory):
        files = [f for f in fnames if f.endswith(extension)]

    return files


# def pickling_image(image):
#     filename = image.split('/')[-1]
#     filename = '/'.join([_dir_encoded, filename])
#     if not os.path.isfile(filename):
#         try:
#             content = encode_faces(image)[0]  # encode one face
#         except IndexError:
#             print('No face detected')
#             delete_image(filename.split('/')[-1])
#         else:
#             save_pickle(filename, content)


def get_pickled_images(images: List) -> Dict:
    dict = {}
    for img in images:
        filename = img.split('/')[-1]  # name.jpg
        nama = filename.split(".jpg")[0]  # name
        filename = '/'.join([_dir_encoded, filename])
        dict[nama] = read_pickle(filename)

    return dict


def encode_faces(img_path: str):
    '''return encoded all face in a image'''
    face = fr.load_image_file(img_path)

    flocations = fr.face_locations(face, 2)
    result = fr.face_encodings(face, flocations, model='large')
    # result = fr.face_encodings(face, model='small')

    return result


def compress_img(img_path: str, size: Tuple, quality: int):
    img = Image.open(img_path)
    img_size = img.size
    if img_size[0] > size[0] or img_size[1] > size[1]:
        img.thumbnail(size, Image.ANTIALIAS)
    if img.mode == 'RGBA':  # png support, A stands for alpha which is transparency
        img = img.convert('RGB')
    img.save(img_path, quality=quality)


def classify_face(unknown_face_encodings, encoded_faces: Dict):
    faces_encoded = list(encoded_faces.values())
    known_face_names = list(encoded_faces.keys())

    data = {
        'detected': [],
        'distances': [],
    }
    for face_encoding in unknown_face_encodings:
        face_distances = fr.face_distance(faces_encoded, face_encoding)

        distances = face_distances.argsort()
        for d in distances:
            if face_distances[d] > 0.6:
                break
            name = known_face_names[d]
            data['detected'].append(name)
            data['distances'].append(face_distances[d])
        data['detected'].append('Unknown')

    return data
