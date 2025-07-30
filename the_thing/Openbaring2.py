import sys
import time
import os
import glob
import shutil
import config
import no_mapping_missy
import mapping
import mapping_plus_plus
import webcam_capture
import threading  # yes

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageEnhance
from notification_manager import NotificationManager
from detectron2_mask_segmentation import mask_detection_detectron
from mask_segmentation import mask_detection_roboflow
from encryption import encrypt_and_package_images

MAIN_WIDTH = 1120 # 960
MAIN_HEIGHT = 630 # 540
BACKGROUND_COLOR = "black"

LEFT_FRAME_COLOR = "#1b0b1b"
RIGHT_FRAME_COLOR = "black"

reconstructed_image = "regenerated_images/reconstructed_output.png"


class SecondPage(ctk.CTkFrame):
    def __init__(self, parent, geometry):
        parent.geometry(geometry)
        super().__init__(parent, width=MAIN_WIDTH, height=MAIN_HEIGHT, fg_color=BACKGROUND_COLOR)
        self.parent = parent
        
        self.image_pathway = None
        self.message_label = None
        self.current_images_race = None
        self.left_display_label = None
        self.mask_present = None
        self.which_image_source = "unknown"
        self.echo_images = []
        self.echo_index = 0
        self.first_load = True
        self.inference_duration = 0.0
        
        self.place(relx=0.5, rely=0.5, anchor="center")
        self.create_layout()

        self.notification_manager = NotificationManager(self)

    def create_layout(self):
        '''Left'''
        self.left_frame = ctk.CTkFrame(
            self,
            width=MAIN_WIDTH // 2,
            height=MAIN_HEIGHT,
            fg_color=LEFT_FRAME_COLOR,
            corner_radius=0
        )
        self.left_frame.place(relx=0.0, rely=0.0, relwidth=0.5, relheight=1.0)
        
        display_width_left = int((MAIN_WIDTH // 2) * 0.9)
        display_height_left = int(MAIN_HEIGHT * 0.5)
        self.left_display_label = ctk.CTkLabel(
            self.left_frame,
            text="New image will appear here",
            text_color="#403f19",
            width=display_width_left,
            height=display_height_left,
            fg_color="transparent"
        )
        self.left_display_label.place(relx=0.5, rely=0.45, anchor="center")
        
        self.generate_button = ctk.CTkButton(
            self.left_frame,
            text="Generate",
            command=self.on_generate_button,
            width=150,
            height=40,
            corner_radius=0,
            fg_color="#3F1940",
            hover_color="#321433"
        )
        self.generate_button.place(relx=0.5, rely=0.8, anchor="center")
        
        '''Right'''   
        self.right_frame = ctk.CTkFrame(
            self,
            width=MAIN_WIDTH // 2,
            height=MAIN_HEIGHT,
            fg_color=RIGHT_FRAME_COLOR,
            corner_radius=0
        )
        self.right_frame.place(relx=0.5, rely=0.0, relwidth=0.5, relheight=1.0)
        
        display_width_right = int((MAIN_WIDTH // 2) * 0.9)
        display_height_right = int(MAIN_HEIGHT * 0.5)
        self.image_display = ctk.CTkLabel(
            self.right_frame,
            text="No image selected",
            width=display_width_right,
            height=display_height_right,
            fg_color="transparent"
        )
        self.image_display.place(relx=0.5, rely=0.45, anchor="center")
        
        self.choose_image_button = ctk.CTkButton(
            self.right_frame,
            text="Choose Image",
            command=self.choose_image,
            width=150,
            height=40,
            corner_radius=0,
            fg_color="#3F1940",
            hover_color="#321433"
        )
        self.choose_image_button.place(relx=0.5, rely=0.8, anchor="center")
        
        camera_img = ctk.CTkImage(Image.open("assets/icons/camera.png"), size=(20, 20))
        self.webcam_toggle_button = ctk.CTkButton(
            self,
            image=camera_img,
            text="",
            command=self.toggle_webcam_mode,
            corner_radius=0,
            width=30,
            height=30,
            fg_color="#3F1940",
            hover_color="#321433"
        )
        self.webcam_toggle_button.place(relx=0.955, rely=0.075, anchor="center")
        
        '''Home'''
        self.home_button = ctk.CTkButton(
            self,
            text="Home Page",
            command=self.go_home,
            corner_radius=0,
            width=150,
            height=40,
            fg_color="#3F1940",
            hover_color="#321433"
        )
        self.home_button.place(relx=0.5, rely=0.925, anchor="center")

    def choose_image(self):
        self.which_image_source = "Upload"
        self.left_display_label.configure(image=None, text="")
        self.remove_loading_message()
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        if file_path:
            self.image_pathway = file_path
            image = Image.open(file_path)
            display_width_right = int((MAIN_WIDTH // 2) * 0.9)
            display_height_right = int(MAIN_HEIGHT * 0.5)
            image.thumbnail((display_width_right, display_height_right))
            chosen_ctk_image = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=(image.width, image.height)
            )
            self.image_display.configure(image=chosen_ctk_image, text="")
            self.image_display.image = chosen_ctk_image
            if self.first_load:
                self.left_display_label.configure(text="New image loaded!\n" + self.image_pathway)
                self.first_load = False
            self.display_loading_message("")
            self.notification_manager.add_message("Image loaded")
            self.is_mask_there(config.FACE_MASK_METHOD)
            
    def toggle_webcam_mode(self):
        if not hasattr(self, 'webcam_widget'):
            self.image_display.place_forget()
            display_width_right = int((MAIN_WIDTH // 2) * 0.5)
            display_height_right = int(MAIN_HEIGHT * 0.5)
            self.webcam_widget = webcam_capture.WebcamCapture(
                self.right_frame,
                self.webcam_capture_callback,
                width=display_width_right,
                height=display_height_right
            )
            self.webcam_widget.place(relx=0.5, rely=0.45, anchor="center")
            self.webcam_toggle_button.configure(fg_color="#1a4019", hover_color="#143314")
        else:
            self.webcam_widget.close()
            del self.webcam_widget
            self.image_display.place(relx=0.5, rely=0.45, anchor="center")
            self.webcam_toggle_button.configure(fg_color="#3F1940", hover_color="#321433")

    def webcam_capture_callback(self, image_path):
        self.which_image_source = "Webcam"
        self.image_pathway = image_path
        image = Image.open(image_path)
        display_width_right = int((MAIN_WIDTH // 2) * 0.9)
        display_height_right = int(MAIN_HEIGHT * 0.5)
        image.thumbnail((display_width_right, display_height_right))
        chosen_ctk_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(image.width, image.height)
        )
        self.image_display.configure(image=chosen_ctk_image, text="")
        self.image_display.image = chosen_ctk_image
        self.display_loading_message("")
        self.notification_manager.add_message("Image loaded")
        self.is_mask_there(config.FACE_MASK_METHOD)
        self.webcam_toggle_button.configure(fg_color="#3F1940")

    def is_mask_there(self, detection_method):
        if detection_method == 1:
            detection_message, self.mask_present = mask_detection_detectron(image_path=self.image_pathway)
        elif detection_method == 2:
            detection_message, self.mask_present = mask_detection_roboflow(image_path=self.image_pathway)
        else:
            detection_message = "Something went wrong."
        self.notification_manager.add_message(detection_message)
        
    def on_generate_button(self):
        if self.mask_present is None:
            self.notification_manager.add_message("No image selected.")
            return
        if not self.mask_present:
            self.notification_manager.add_message(
                "Cannot regenerate - no mask detected. This feature is only available for masked individuals."
            )
            return
        self.show_loading_indicator(0)
        threading.Thread(target=self._do_regeneration, daemon=True).start()

    def _do_regeneration(self):
        start_time = time.time()
        if config.INFERENCE_MODEL == 1:
            no_mapping_missy.main(self.image_pathway)
        elif config.INFERENCE_MODEL == 2:
            mapping.main(self.image_pathway)
        else:
            mapping_plus_plus.main(self.image_pathway)
        duration = time.time() - start_time
        self.after(0, lambda: self._on_regeneration_done(duration))

    def _on_regeneration_done(self, duration):
        self.inference_duration = duration
        self.hide_loading_indicator()
        if config.INFERENCE_MODEL in (1, 2):
            self.echo_images = [reconstructed_image]
        else:
            output_folder = os.path.dirname(reconstructed_image)
            self.echo_images = sorted(glob.glob(os.path.join(output_folder, "echo*")))
        self.echo_index = 0
        self.show_echo_image()
        self.notification_manager.add_message(f"Inference took {duration:.2f}s")

    def show_loading_indicator(self, seconds=0):
        message = "please wait"
        index_to_replace = seconds % len(message)
    
        chars = list(message)
        chars[index_to_replace] = '*'
        animated_text = ''.join(chars)
    
        display = f"{animated_text} - {seconds}s"
    
        if self.message_label:
            self.message_label.destroy()
        self.message_label = ctk.CTkLabel(
            self.right_frame,
            text=display,
            fg_color="transparent",
            font=("Helvetica", 18, "bold")
        )
        self.message_label.place(relx=0.5, rely=0.15, anchor="center")
        self.loading_after_id = self.after(1000, lambda: self.show_loading_indicator(seconds + 1))

    def hide_loading_indicator(self):
        if hasattr(self, 'loading_after_id'):
            self.after_cancel(self.loading_after_id)
            del self.loading_after_id
        if self.message_label:
            self.message_label.configure(text="Finished")
            self.after(2000, self.remove_loading_message)

    def show_echo_image(self):
        display_width_left = int((MAIN_WIDTH // 2) * 0.9)
        display_height_left = int(MAIN_HEIGHT * 0.5)
        for attr in ('prev_button', 'next_button', 'save_name_entry', 'save_yes_button', 'save_no_button'):
            if hasattr(self, attr):
                getattr(self, attr).destroy()
                delattr(self, attr)

        image_path = self.echo_images[self.echo_index]
        image = Image.open(image_path)
        image.thumbnail((display_width_left, display_height_left))
        chosen_ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(image.width, image.height))
        self.left_display_label.configure(image=chosen_ctk_image, text="")
        self.left_display_label.image = chosen_ctk_image

        if len(self.echo_images) > 1:
            self.prev_button = ctk.CTkButton(
                self.left_frame, text="←", command=self.prev_echo,
                width=40, height=int(display_height_left * 0.9), corner_radius=0,
                fg_color="#2d122d", hover_color="#1b0b1b"
            )
            self.prev_button.place(relx=0.15, rely=0.5, anchor="w")

            self.next_button = ctk.CTkButton(
                self.left_frame, text="→", command=self.next_echo,
                width=40, height=int(display_height_left * 0.9), corner_radius=0,
                fg_color="#2d122d", hover_color="#1b0b1b"
            )
            self.next_button.place(relx=0.85, rely=0.5, anchor="e")

        self.prompt_save()

    def prev_echo(self):
        self.echo_index = (self.echo_index - 1) % len(self.echo_images)
        self.show_echo_image()

    def next_echo(self):
        self.echo_index = (self.echo_index + 1) % len(self.echo_images)
        self.show_echo_image()

    def prompt_save(self):
        self.save_name_entry = ctk.CTkEntry(self.left_frame, width=120, placeholder_text="Enter name", corner_radius=0, fg_color="#2d122d")
        self.save_name_entry.place(relx=0.5, rely=0.685, anchor="center")
        self.save_yes_button = ctk.CTkButton(self.left_frame, text="✔", command=self.save_echo, width=50, corner_radius=0, fg_color="#2d122d", hover_color="#1b0b1b")
        self.save_yes_button.place(relx=0.45, rely=0.735, anchor="center")
        self.save_no_button = ctk.CTkButton(self.left_frame, text="✖", command=self.cancel_echo, width=50, corner_radius=0, fg_color="#2d122d", hover_color="#1b0b1b")
        self.save_no_button.place(relx=0.55, rely=0.735, anchor="center")

    def dim_current_image(self):
        if self.left_display_label.cget("image") != "":
            original = self.left_display_label.image._light_image
            enhancer = ImageEnhance.Brightness(original)
            dimmed = enhancer.enhance(0.5)
            dim_ctk = ctk.CTkImage(light_image=dimmed, dark_image=dimmed, size=(original.width, original.height))
            self.left_display_label.configure(image=dim_ctk)
            self.left_display_label.image = dim_ctk

    def save_echo(self):
        image_name = self.save_name_entry.get().strip()
        if image_name == "":
            image_name = "generated_image.png"
        elif not image_name.lower().endswith(".png"):
            image_name += ".png"
        folder = os.path.dirname(reconstructed_image)
        final_path = os.path.join(folder, image_name)
        shutil.copy(self.echo_images[self.echo_index], final_path)
        staff_id = self.parent.session.get("staff_id", "unknown")
        full_name = self.parent.session.get("full_name", "unknown")
        password = self.parent.session.get("password", "")
        message = encrypt_and_package_images(
            [final_path, self.image_pathway],
            staff_id,
            password,
            full_name,
            self.which_image_source,
            self.inference_duration
        )
        self.notification_manager.add_message(message)
        for attr in ('prev_button', 'next_button', 'save_name_entry', 'save_yes_button', 'save_no_button'):
            if hasattr(self, attr):
                getattr(self, attr).destroy()
                delattr(self, attr)
        self.dim_current_image()

    def cancel_echo(self):
        for attr in ('prev_button', 'next_button', 'save_name_entry', 'save_yes_button', 'save_no_button'):
            if hasattr(self, attr):
                getattr(self, attr).destroy()
                delattr(self, attr)
        self.dim_current_image()

    def display_loading_message(self, lil_message):
        if self.message_label:
            self.message_label.destroy()
        
        self.message_label = ctk.CTkLabel(self.right_frame,
            text="",
            fg_color="transparent",
            font=("Helvetica", 18, "bold")
        )
        self.message_label.place(relx=0.5, rely=0.15, anchor="center")
        
        for i in range(6):
            self.message_label.configure(text="." * i)
            self.update()
            time.sleep(0.4)
        
        self.message_label.configure(text=lil_message)

    def remove_loading_message(self):
        if self.message_label:
            self.message_label.destroy()
            self.message_label = None

    def go_home(self):
        from Openbaring1 import HomePage
        current_geometry = self.parent.geometry()
        self.destroy()
        HomePage(self.parent)
