import tkinter as tk
import customtkinter as ctk
import os
import shutil

from Openbaring2 import SecondPage
from Openbaring3 import ThirdPage
from overlays import Settings, LoginOverlay, RegisterOverlay
from PIL import Image, ImageEnhance
from notification_manager import NotificationManager


MAIN_WIDTH = 1120
MAIN_HEIGHT = 630
OVERLAY_WIDTH = int(MAIN_WIDTH * 0.9)
OVERLAY_HEIGHT = int(MAIN_HEIGHT * 0.9)
BACKGROUND_COLOR = "black"
BG_DIM_FACTOR = 0.25
OVERLAY_DIM_FACTOR = 0.35
OVERLAY_REL_WIDTH = 0.8
OVERLAY_REL_HEIGHT = 0.7
OVERLAY_CORNER_RADIUS = 20

'''
The name doesn't display the first time I log in  # FIXED

Some messages are too long  # FIXED, SCOLL WHEN CERTAIN LENGTH

App freezes when waiting for inference  # FIXED, threading

'''

class HomePage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, width=MAIN_WIDTH, height=MAIN_HEIGHT, fg_color=BACKGROUND_COLOR)
        self.parent = parent
        self.overlay = None  # overlays.py
        self.logged_in_staff_id = parent.session.get("staff_id", "")
        self.logged_in_name = parent.session.get("full_name", "")

        self.place(relx=0.5, rely=0.5, anchor="center")
        self.create_framework()
        
        self.notification_manager = NotificationManager(self)
       
    def create_framework(self):
        self.bg_image = Image.open("assets/backgrounds/Aurora Borealis.jpg")
        self.bg_dim_image = ImageEnhance.Brightness(self.bg_image).enhance(BG_DIM_FACTOR)
        self.bg_ctk_image = ctk.CTkImage(
            light_image=self.bg_dim_image,
            dark_image=self.bg_dim_image,
            size=(MAIN_WIDTH, MAIN_HEIGHT)
        )
        self.bg_label = ctk.CTkLabel(self, text="", image=self.bg_ctk_image, fg_color="transparent")
        self.bg_label.place(relx=0.5, rely=0.5, anchor="center")
        self.bg_label.image = self.bg_ctk_image
    
        overlay_width = int(MAIN_WIDTH * OVERLAY_REL_WIDTH)  # Setup inner overlay image
        overlay_height = int(MAIN_HEIGHT * OVERLAY_REL_HEIGHT)  # ||
        self.overlay_image = ImageEnhance.Brightness(self.bg_image).enhance(OVERLAY_DIM_FACTOR)
    
        '''
        size of the logo is relative to the pixels of the image (5184 x 3456)
        '''
        logo = Image.open("assets/logo/openbare_white.png").convert("RGBA")
        logo = logo.resize((862, 1020), Image.Resampling.LANCZOS)

        overlay_w, overlay_h = self.overlay_image.size   # 5184, 3456
        logo_w, logo_h = logo.size
        
        x = (overlay_w - logo_w) * 86 // 100
        y = int(overlay_h * 0.39)  # up atm

        self.overlay_image.paste(logo, (x, y), logo)
    
        self.overlay_ctk_image = ctk.CTkImage(
            light_image=self.overlay_image,
            dark_image=self.overlay_image,
            size=(overlay_width, overlay_height)
        )
        self.overlay_frame = ctk.CTkFrame(
            self,
            width=overlay_width,
            height=overlay_height,
            fg_color="transparent",
            corner_radius=OVERLAY_CORNER_RADIUS
        )
        self.overlay_frame.place(relx=0.5, rely=0.5, anchor="center")
    
        self.overlay_label = ctk.CTkLabel(
            self.overlay_frame,
            text="            Openbare\n\n",
            font=("Helvetica", 90),
            text_color="white",
            image=self.overlay_ctk_image,
            fg_color="transparent",
            compound="center"
        )
        self.overlay_label.place(relx=0.5, rely=0.5, anchor="center")
        self.overlay_label.image = self.overlay_ctk_image
    
        self.user_display_label = ctk.CTkLabel(
            self,
            text=self.logged_in_name,
            font=("Helvetica", 16, "bold"),
            text_color="gold",
            fg_color="transparent"
        )
        self.user_display_label.place(relx=0.015, rely=0.03, anchor="nw")
    
        self.nav_button = ctk.CTkButton(
            self,
            text="Try now",
            command=self.main_functionality_page,
            corner_radius=0,
            width=150,
            height=40,
            fg_color="#3F1940",
            hover_color="#321433"
        )
        self.nav_button.place(relx=0.6, rely=0.925, anchor="center")
    
        terms_img = ctk.CTkImage(Image.open("assets/icons/terms-info.png"), size=(20, 20))
        self.pdf_button = ctk.CTkButton(
            self,
            image=terms_img,
            text="",
            command=self.documentation_page,
            corner_radius=0,
            width=30,
            height=30,
            fg_color="#3F1940",
            hover_color="#321433"
        )
        self.pdf_button.place(relx=0.915, rely=0.075, anchor="center")
        
        settings_img = ctk.CTkImage(Image.open("assets/icons/settings.png"), size=(20, 20))
        self.settings_button = ctk.CTkButton(
            self,
            image=settings_img,
            text="",
            command=self.settings_sheet,
            corner_radius=0,
            width=30,
            height=30,
            fg_color="#3F1940",
            hover_color="#321433"
        )
        self.settings_button.place(relx=0.955, rely=0.075, anchor="center")
        
        button_text = "Logout" if self.logged_in_staff_id else "Login"
        self.login_button = ctk.CTkButton(
            self,
            text=button_text,
            command=self.handle_login_logout,
            corner_radius=0,
            width=150,
            height=40,
            fg_color="#3F1940",
            hover_color="#321433"
        )
        self.login_button.place(relx=0.4, rely=0.925, anchor="center")

    def settings_sheet(self):
        if self.overlay and self.overlay.winfo_exists():
            return
        self.overlay = Settings(self.parent, notify_callback=self.notification_manager.add_message, close_callback=self.on_overlay_closed)

    def handle_login_logout(self):
        if self.logged_in_staff_id:
            self.logged_in_staff_id = ""
            self.logged_in_name = ""
            self.user_display_label.configure(text="")
            self.login_button.configure(text="Login")
            self.parent.session = {}  # Empty the dictionary
            self.notification_manager.add_message("Logged out successfully.")
            # print("Logged out successfully.")
        else:
            self.login_overlay()

# Save login info to the main App so it remains even if pages or overlays are destroyed
# The actual main App object...self.session, which is a dictionary...

    def update_user_info(self, staff_id, full_name):
        self.logged_in_staff_id = staff_id
        self.logged_in_name = full_name
        self.parent.session["staff_id"] = staff_id
        self.parent.session["full_name"] = full_name
        self.user_display_label.configure(text=full_name)
        self.login_button.configure(text="Logout")
        ##################################################

    def login_overlay(self):
        if self.overlay and self.overlay.winfo_exists():
            return
        self.overlay = LoginOverlay(self.parent,
                                    login_success_callback=self.update_user_info,
                                    switch_to_register_callback=self.switch_to_register,
                                    notify_callback=self.notification_manager.add_message,
                                    close_callback=self.on_overlay_closed)

    def switch_to_register(self):
        if self.overlay and self.overlay.winfo_exists():
            self.overlay.destroy()
        self.overlay = RegisterOverlay(
            self.parent,
            switch_to_login_callback=self.login_overlay,
            notify_callback=self.notification_manager.add_message,
            close_callback=self.on_overlay_closed
        )

    def on_overlay_closed(self):
        self.overlay = None

    def main_functionality_page(self):
        if not self.logged_in_staff_id:
            message = "Error: Must login first."
            self.notification_manager.add_message(message)
            # print(message)
            return message
        current_geometry = self.parent.geometry()
        self.destroy()
        SecondPage(self.parent, current_geometry)

    def documentation_page(self):
        current_geometry = self.parent.geometry()
        self.destroy()
        ThirdPage(self.parent, current_geometry)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Openbaring")
        self.geometry(f"{MAIN_WIDTH}x{MAIN_HEIGHT}")
        self.resizable(False, False)
        self.configure(fg_color=BACKGROUND_COLOR)
        self.session = {}
        self.home_page = HomePage(self)
        self.regenerated_images = "regenerated_images"

    def clear_folder_contents(self, folder_path):
        if os.path.exists(folder_path):
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Error deleting {item_path}: {e}")

    def on_closing(self):
        self.clear_folder_contents(self.regenerated_images)
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
