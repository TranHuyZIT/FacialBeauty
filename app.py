import cv2
import tkinter as tk
from tkinter import Button, Label, Toplevel
from PIL import Image, ImageTk
from main import predict_beauty

# Load the face detection algorithm
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

# Initialize webcam
cam = cv2.VideoCapture(0)

def capture_image():
    _, img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Use the first detected face
        # Bigger frame with the same aspect ratio
        face_img = img[y-30:y+h+100, x-50:x+w+50]

        # Convert the face image to RGB and ImageTk format for display
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img_pil = Image.fromarray(face_img_rgb)
        score = predict_beauty(face_img_pil)
        print(f"Beauty score:", score)

        cv2.putText(face_img, f'Beauty score: {score:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img_pil = Image.fromarray(face_img_rgb)
        face_img_tk = ImageTk.PhotoImage(image=face_img_pil)
        
        # Create a new window to show the cropped face image
        capture_window = Toplevel(root)
        capture_window.title("Captured Face")

        captured_face_label = Label(capture_window, image=face_img_tk)
        captured_face_label.imgtk = face_img_tk  # Keep a reference
        captured_face_label.pack()

    else:
        print("No face detected.")

def show_frame():
    _, img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Convert the frame to RGB and then to ImageTk format for the live feed
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    
    # Update the label with the new frame
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)
    
    key = cv2.waitKey(10)
    if key == 27:  # ESC key to exit
        cam.release()
        cv2.destroyAllWindows()
        root.destroy()
    else:
        root.after(10, show_frame)

# Setup tkinter window
root = tk.Tk()
root.title("Face Detection")

# Create label to display the video feed
video_label = Label(root)
video_label.pack()

# Button to capture image
capture_button = Button(root, text="Capture Image", command=capture_image, width=20, height=2)
capture_button.pack()

# Start the video feed
root.after(10, show_frame)
root.mainloop()
