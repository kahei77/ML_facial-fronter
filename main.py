import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

import train_test


class App:

    def __init__(self, window):
        self.window = window
        self.window.title("CSCI 6364 Project")
        self.window.geometry("1200x800")

        self.video_capture_label = tk.Label(window, width=400, height=400)
        self.video_capture_label.place(x=150, y=50)

        self.selected_image_label = tk.Label(window, width=400, height=400)
        self.selected_image_label.place(x=600, y=50)

        self.instruction_label = tk.Label(window, text="Capture Image or Choose a Local Image", font=("Arial", 26))
        self.instruction_label.place(x=400, y=500)

        self.result_label = tk.Label(window, text="Result:", font=("Arial", 24))
        self.result_label.place(x=500, y=550)

        self.capture_button = tk.Button(window, text="Capture", command=self.capture)
        self.capture_button.place(x=400, y=650, width=150, height=50)

        self.choose_image_button = tk.Button(window, text="Choose Image", command=self.choose_local_image)
        self.choose_image_button.place(x=600, y=650, width=150, height=50)

        self.video_cap = cv2.VideoCapture(0)

    def crop_square_image(self, image: Image) -> Image:
        width, height = image.size
        min_size = min(width, height)
        left = (width - min_size) // 2
        top = (height - min_size) // 2
        right = left + min_size
        bottom = top + min_size
        square_image = image.crop((left, top, right, bottom))
        return square_image

    def get_cropped_image_input(self) -> Image:
        res, image = self.video_cap.read()
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = Image.fromarray(image)
        image = self.crop_square_image(image)
        return image

    def capture(self):
        image = self.get_cropped_image_input()
        self.show_captured_or_selected_image(image)
        self.recognize(image)

    def choose_local_image(self):
        file_name = filedialog.askopenfilename()
        image = Image.open(file_name)
        image = self.crop_square_image(image)
        self.show_captured_or_selected_image(image)
        self.recognize(image)

    def show_captured_or_selected_image(self, image: Image):
        image = image.resize((400, 400))
        image = ImageTk.PhotoImage(image)
        self.selected_image_label.configure(image=image)
        self.selected_image_label.image = image

    def show_frames(self):
        image = self.get_cropped_image_input()
        image = image.resize((400, 400))
        photo_image = ImageTk.PhotoImage(image=image)

        self.video_capture_label.imgtk = photo_image
        self.video_capture_label.configure(image=photo_image)
        self.video_capture_label.after(20, self.show_frames)

    def analyze(self, image: Image):
        result = train_test.make_predictions(image)
        return result

    def recognize(self, image: Image):
        res = self.analyze(image)
        self.result_label.config(text="Result: " + res)

    def run(self):
        self.show_frames()
        self.window.mainloop()
        self.video_cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    window = tk.Tk()
    app = App(window)
    app.run()

