import cv2
import pytesseract
from flask import Flask, render_template, request, Response, redirect, url_for
from PIL import Image
from io import BytesIO
import numpy as np
import os
import face_recognition
from PIL import Image
from io import BytesIO
import base64
import string

app = Flask(__name__)
known_users = []
known_face_encodings = []
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

USER_DIR = os.path.join(os.getcwd(), "users")

# Function to perform OCR on an image
def ocr(image):
    text = pytesseract.image_to_string(image)
    return text if text.strip() else "no text found"

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# get grayscale image
def get_RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

# Route for the home page
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/users')
def users():
    folders = get_folders(USER_DIR)
    return render_template('userPage.html', folders=folders)

@app.route('/newUser')
def newUser():
    return render_template('newUserPage.html')

@app.route('/folder/<folder_name>')
def folder(folder_name):
    folder_path = os.path.join(USER_DIR, folder_name)
    contents = os.listdir(folder_path)
    return render_template('folder.html', folder_name=folder_name, contents=contents)

def get_folders(directory):
    folders = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            folders.append(item)
    return folders

@app.route('/submit', methods=['POST'])
def submit():
    username = request.form['username']
    image_data = request.form['image_data']
    known_users.append(username)
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    save_path = os.path.join(os.getcwd(), "user_faces", username + ".png")
    image.save(save_path)
    this_image = face_recognition.load_image_file(save_path)
    this_face_encoding = face_recognition.face_encodings(this_image)
    known_face_encodings.append(this_face_encoding)
    directory_path = os.path.join(os.getcwd(), "user_faces", username)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    return f"Directory for username {username} created"


@app.route('/capture', methods=['POST'])
def capture():
    # Capture image from webcam
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    camera.release()
    clean = remove_noise(image)
    gray = get_grayscale(image)
    rgb = get_RGB(image)
    image = get_RGB(image)
    opened = opening(clean)
    thresh = thresholding(gray)
    cannied = canny(clean)
    extracted_text = ocr(rgb)

    pil_image = Image.fromarray(rgb)
    img_buffer = BytesIO()
    pil_image.save(img_buffer, format="JPEG")
    img_str = img_buffer.getvalue()
    import base64
    img_base64 = base64.b64encode(img_str).decode('utf-8')
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_image)
    new_face_encodings = face_recognition.face_encodings(rgb_small_image, face_locations)

    face_names = []
    for face_encoding in new_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = users[best_match_index]
            face_names.append(name)
        else:
            face_names = ["Unknown"] * len(face_locations)

        print(face_names)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the image we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    pil_image = Image.fromarray(image)
    # Convert image to base64 string
    img_buffer = BytesIO()
    pil_image.save(img_buffer, format="JPEG")
    img_str = img_buffer.getvalue()
    import base64
    img_base64 = base64.b64encode(img_str).decode('utf-8')
    return {'text': extracted_text, 'image': img_base64}

@app.route('/captureNewUser', methods=['POST'])
def newUserCapture():
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    camera.release()
    clean = remove_noise(image)
    gray = get_grayscale(image)
    rgb = get_RGB(image)
    opened = opening(clean)
    thresh = thresholding(gray)
    cannied = canny(clean)
    extracted_text = ocr(gray)
    pil_image = Image.fromarray(gray)
    img_buffer = BytesIO()
    pil_image.save(img_buffer, format="JPEG")
    img_str = img_buffer.getvalue()
    import base64
    img_base64 = base64.b64encode(img_str).decode('utf-8')

    return {'text': extracted_text, 'image': img_base64}

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port=8000, debug=True)
    ##python -m http.server 8000 --bind 0.0.0.0
