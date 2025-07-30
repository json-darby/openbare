import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*CTkLabel Warning: Given image is not CTkImage.*"
)

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import os

# User passthrough rather than redefining everything...
# gold = #403f19

class WebcamCapture(ctk.CTkFrame):
    def __init__(self, parent, capture_callback, width=280, height=316, fg_color="#0c0c0c", *args, **kwargs):
        super().__init__(parent, width=width, height=height, fg_color=fg_color, corner_radius=0, *args, **kwargs)
        self.capture_callback = capture_callback
        self.webcam_active = True
        self.snapshot_taken = False
        self.width = width
        self.height = height
        self.feed_size = 256
        self.control_height = height - self.feed_size

        self.video_label = ctk.CTkLabel(self, text="", width=self.feed_size, height=self.feed_size, corner_radius=0)
        self.video_label.place(x=(width - self.feed_size)//2, y=10)

        self.capture_button = ctk.CTkButton(
            self,
            text="Capture",
            command=self.snapshot,
            width=80,
            height=30,
            corner_radius=0,
            fg_color="#1a4019",
            hover_color="#143314"
        )
        self.capture_button.place(x=width//2, y=self.feed_size + self.control_height//2 - 5, anchor="center")
        
        self.focus_set()
        self.cap = cv2.VideoCapture(0)
        self._after_id = None
        self.after(10, self.update_frame)

    def update_frame(self):
        if not self.webcam_active:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            image = Image.fromarray(frame)
            image = image.resize((self.feed_size, self.feed_size))
            self.current_image = ImageTk.PhotoImage(image)
            self.video_label.configure(image=self.current_image)
        self._after_id = self.after(30, self.update_frame)

    def snapshot(self, event=None):
        if self.snapshot_taken:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            self.snapshot_image = Image.fromarray(frame).resize((self.feed_size, self.feed_size))
            self.current_image = ImageTk.PhotoImage(self.snapshot_image)
            self.video_label.configure(image=self.current_image)
            self.webcam_active = False
            self.snapshot_taken = True
            
            if self._after_id is not None:
                self.after_cancel(self._after_id)
            self.capture_button.place_forget()
            
            self.name_entry = ctk.CTkEntry(
                self,
                width=120,
                placeholder_text="Enter name",
                corner_radius=0,
                fg_color="#1a4019"
            )
            self.name_entry.place(x=self.width//2, y=self.feed_size + self.control_height//2 - 20, anchor="center")
            
            self.yes_button = ctk.CTkButton(
                self,
                text="✔",
                command=self.save_snapshot,
                width=50,
                corner_radius=0,
                fg_color="#1a4019",
                hover_color="#143314"
            )
            self.yes_button.place(x=self.width//2 - 35, y=self.feed_size + self.control_height//2 + 10, anchor="center")
            
            self.no_button = ctk.CTkButton(
                self,
                text="✖",
                command=self.cancel_snapshot,
                width=50,
                corner_radius=0,
                fg_color="#1a4019",
                hover_color="#143314"
            )
            self.no_button.place(x=self.width//2 + 35, y=self.feed_size + self.control_height//2 + 10, anchor="center")

    def save_snapshot(self):
        image_name = self.name_entry.get()
        if image_name.strip() == "":
            image_name = "occluded_image.png"
        else:
            if not image_name.lower().endswith(".png"):
                image_name = image_name + ".png"
        folder = "taken_images"
        if not os.path.exists(folder):
            os.makedirs(folder)
        final_path = os.path.join(folder, image_name)
        self.snapshot_image.save(final_path)
        self.capture_callback(final_path)
        self.close()
        self.master.master.image_display.place(relx=0.5, rely=0.45, anchor="center")  # Parent's parent


    def cancel_snapshot(self):
        self.snapshot_taken = False
        self.webcam_active = True
        self.name_entry.destroy()
        self.yes_button.destroy()
        self.no_button.destroy()
        self.capture_button.place(x=self.width//2, y=self.feed_size + self.control_height//2 - 5, anchor="center")
        self.after(10, self.update_frame)

    def close(self):
        self.webcam_active = False
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.destroy()
