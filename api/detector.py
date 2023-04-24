import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import sys
import os

pwd = os.path.dirname(__file__)
# print(pwd)
path = pwd+'/shape_predictor_68_face_landmarks.dat'

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)
# url = "https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat"

# Calculate the eye aspect ratio (EAR) from the eye landmarks
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def detect_drowsiness(image):
    print("image : ",image)
    response = {}
    is_drowsy = False

    try:
        # Load the input image
        img = cv2.imread(image)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = detector(gray)
        # if len(faces) <=0:
        #     # print({
        #     #     "error" : "no face detected"
        #     # })
        #     response['error'] = "no face detected"
        print(faces)

        # Loop over each face
        for face in faces:
            # Detect the facial landmarks for the face
            landmarks = predictor(gray, face)
            
            # Extract the eye landmarks
            left_eye = []
            right_eye = []
            for n in range(36, 42):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                left_eye.append((x, y))
            for n in range(42, 48):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                right_eye.append((x, y))
            
            # Calculate the eye aspect ratio (EAR) for each eye
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            # If the EAR is below a certain threshold, the person is drowsy
            if ear < 0.2:
                response['is_drowsy'] = True
                response['EAR'] = ear
                response['left_ear'] = left_ear
                response['right_ear'] = right_ear
                response['error'] = "none"
                print("drowsy")
                # cv2.putText(img, 'Drowsy', (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                response['is_drowsy'] = False
                response['EAR'] = ear
                response['left_ear'] = left_ear
                response['error'] = "none"
                print("not drowsy")
    except Exception:
        print("Exception : ",Exception)
    finally:
        print("resp : ",response)
        return response
    
# detect_drowsiness("./my_image.jpg")

# sys.modules[__name__] = detector