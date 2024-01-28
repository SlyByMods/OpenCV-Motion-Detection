import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

class MotionDetectionApp:
    def __init__(self, root, cap):
        self.root = root
        self.root.title("Motion Detection Test - MattVoid")
        self.cap = cap
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.detect_faces = False
        self.detect_eyes = False
        self.previous_x, self.previous_y, self.previous_w, self.previous_h = 0, 0, 0, 0
        self.draw_alpha = False
        self.remark_detection = False
        self.alpha_value = 0.3
        self.sensitivity = 30
        self.morphology_iterations = 2
        self.contour_thickness = 2
        self.text_size = 0.5
        self.max_contour_lines = 50
        self.label_alpha = tk.Label(root, text="Transparencia del área dentro de contornos")
        self.label_alpha.pack()
        self.scale_alpha = tk.Scale(root, from_=0.1, to=1.0, resolution=0.1, orient="horizontal", command=self.update_alpha)
        self.scale_alpha.set(self.alpha_value)
        self.scale_alpha.pack()
        self.check_alpha = tk.Checkbutton(root, text="Pintar área dentro de contornos", command=self.update_draw_alpha)
        self.check_alpha.pack()
        self.label_sensitivity = tk.Label(root, text="Sensibilidad del sensor de movimiento")
        self.label_sensitivity.pack()
        self.scale_sensitivity = tk.Scale(root, from_=1, to=100, orient="horizontal", command=self.update_sensitivity)
        self.scale_sensitivity.set(self.sensitivity)
        self.scale_sensitivity.pack()
        self.label_morphology = tk.Label(root, text="Iteraciones de morfología (erode/dilate)")
        self.label_morphology.pack()
        self.scale_morphology = tk.Scale(root, from_=1, to=10, orient="horizontal", command=self.update_morphology)
        self.scale_morphology.set(self.morphology_iterations)
        self.scale_morphology.pack()
        self.label_thickness = tk.Label(root, text="Grosor del contorno")
        self.label_thickness.pack()
        self.scale_thickness = tk.Scale(root, from_=1, to=10, orient="horizontal", command=self.update_thickness)
        self.scale_thickness.set(self.contour_thickness)
        self.scale_thickness.pack()
        self.label_text_size = tk.Label(root, text="Tamaño del texto de información")
        self.label_text_size.pack()
        self.scale_text_size = tk.Scale(root, from_=0.1, to=1.0, resolution=0.1, orient="horizontal", command=self.update_text_size)
        self.scale_text_size.set(self.text_size)
        self.scale_text_size.pack()
        self.label_max_contour_lines = tk.Label(root, text="Cantidad máxima de líneas en el contorno")
        self.label_max_contour_lines.pack()
        self.scale_max_contour_lines = tk.Scale(root, from_=1, to=100, orient="horizontal", command=self.update_max_contour_lines)
        self.scale_max_contour_lines.set(self.max_contour_lines)
        self.scale_max_contour_lines.pack()
        self.check_detect_faces = tk.Checkbutton(root, text="Detectar caras", command=self.update_detect_faces)
        self.check_detect_faces.pack()
        self.check_remark_eyes = tk.Checkbutton(root, text="Remarcar ojos", command=self.update_remark_eyes, state=tk.DISABLED)
        self.check_remark_eyes.pack()
        self.check_remark_detection = tk.Checkbutton(root, text="Remarcar Detección", command=self.update_remark_detection)
        self.check_remark_detection.pack()
        
        # Bucle principal para capturar video en tiempo real
        self.update()
        
    def update_alpha(self, value):
        self.alpha_value = float(value)
    def update_draw_alpha(self):
        self.draw_alpha = not self.draw_alpha
    def update_sensitivity(self, value):
        self.sensitivity = int(value)
    def update_morphology(self, value):
        self.morphology_iterations = int(value)
    def update_thickness(self, value):
        self.contour_thickness = int(value)
    def update_text_size(self, value):
        self.text_size = float(value)
    def update_max_contour_lines(self, value):
        self.max_contour_lines = int(value)
    def update_detect_faces(self):
        self.detect_faces = not self.detect_faces
        if self.detect_faces:
            self.check_remark_eyes.config(state=tk.NORMAL)
        else:
            self.detect_eyes = False
            self.check_remark_eyes.deselect()
            self.check_remark_eyes.config(state=tk.DISABLED)
    def update_remark_eyes(self):
        self.detect_eyes = not self.detect_eyes
    def update_remark_detection(self):
        self.remark_detection = not self.remark_detection
    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error al leer el fotograma.")
            self.root.destroy()
            return
        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=self.morphology_iterations)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=self.morphology_iterations)
        fgmask = cv2.erode(fgmask, None, iterations=self.morphology_iterations)
        fgmask = cv2.dilate(fgmask, None, iterations=self.morphology_iterations)
        if self.detect_faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                info_text = f'Cara Detectada | Posicion: ({x}, {y}), Size: ({w}, {h})'
                if self.remark_detection:
                    cv2.putText(frame, info_text, (x, y - int(10 * self.text_size)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, (0, 255, 0), self.contour_thickness)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), self.contour_thickness)
                if self.draw_alpha and self.remark_detection:
                    mask = np.zeros_like(frame[y:y+h, x:x+w])
                    mask[:, :] = (0, 255, 0)
                    frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 1.0, mask, self.alpha_value, 0)
                if self.detect_eyes:
                    roi_gray = gray[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        eye_info_text = f'Ojo Detectado | Posicion: ({x+ex}, {y+ey}), Size: ({ew}, {eh})'
                        if self.remark_detection:
                            cv2.putText(frame, eye_info_text, (x+ex, y+ey - int(10 * self.text_size)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, (0, 255, 0), self.contour_thickness)
                        else:
                            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), self.contour_thickness)
                        if self.draw_alpha and self.remark_detection:
                            eye_mask = np.zeros_like(frame[y+ey:y+ey+eh, x+ex:x+ex+ew])
                            eye_mask[:, :] = (0, 255, 0)
                            frame[y+ey:y+ey+eh, x+ex:x+ex+ew] = cv2.addWeighted(frame[y+ey:y+ey+eh, x+ex:x+ex+ew], 1.0, eye_mask, self.alpha_value, 0)
                self.previous_x, self.previous_y, self.previous_w, self.previous_h = x, y, w, h
        else:
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_contour = max(contours, key=cv2.contourArea, default=None)
            if max_contour is not None:
                epsilon = self.max_contour_lines * 0.001 * cv2.arcLength(max_contour, True)
                approx = cv2.approxPolyDP(max_contour, epsilon, True)
                x, y, w, h = cv2.boundingRect(approx)
                info_text = f'Movimiento Detectado | Posicion: ({x}, {y}), Size: ({w}, {h})'
                if self.remark_detection:
                    cv2.putText(frame, info_text, (x, y - int(10 * self.text_size)), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, (0, 255, 0), self.contour_thickness)
                else:
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), self.contour_thickness)
                if self.draw_alpha and self.remark_detection:
                    mask = np.zeros_like(frame)
                    cv2.drawContours(mask, [approx], 0, (0, 255, 0), -1)
                    frame = cv2.addWeighted(frame, 1.0, mask, self.alpha_value, 0)
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), self.contour_thickness)
                self.previous_x, self.previous_y, self.previous_w, self.previous_h = x, y, w, h
        cv2.imshow('Camara', frame)
        self.root.after(10, self.update)
    def run(self):
        self.root.mainloop()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error al abrir la camara.")
    exit()
root = tk.Tk()
app = MotionDetectionApp(root, cap)
app.run()


cap.release()
cv2.destroyAllWindows()
