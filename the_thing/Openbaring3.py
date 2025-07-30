import customtkinter as ctk
import tkinter as tk
import os
from PIL import Image, ImageEnhance  # image is still blurry somewhat

MAIN_WIDTH = 1120
MAIN_HEIGHT = 630
BACKGROUND_COLOR = "black"

class ThirdPage(ctk.CTkFrame):
    def __init__(self, parent, geometry):
        parent.geometry(geometry)
        super().__init__(parent, width=MAIN_WIDTH, height=MAIN_HEIGHT, fg_color=BACKGROUND_COLOR)
        self.parent = parent
        self.place(relx=0.5, rely=0.5, anchor="center")
        self.create_layout()

    def create_layout(self):
        pdf_path = "Openbare.pdf"

        if os.path.exists(pdf_path):
            self.scroll_frame = ctk.CTkScrollableFrame(
                self,
                width=int(MAIN_WIDTH * 0.85),
                height=int(MAIN_HEIGHT * 0.75),
                fg_color="gray20"
            )
            self.scroll_frame.place(relx=0.5, rely=0.45, anchor="center")

            try:
                from pdf2image import convert_from_path

                # Keep this relative for exe
                poppler_path = r'poppler-24.08.0/Library/bin'
                
                # Convert PDF pages to PIL images
                pages = convert_from_path(pdf_path, poppler_path=poppler_path)
                self.page_images = []  # List to hold references to CTkImage objects

                for i, page in enumerate(pages):
                    # Only resize if too wide
                    page_width, page_height = page.size
                    max_width = int(MAIN_WIDTH * 0.85)
                    if page_width > max_width:
                        ratio = max_width / page_width
                        new_size = (max_width, int(page_height * ratio))
                        
                        # Replace ANTIALIAS with LANCZOS - latest version
                        page = page.resize(new_size, Image.LANCZOS)

                    ctk_page = ctk.CTkImage(light_image=page, dark_image=page, size=page.size)
                    label = ctk.CTkLabel(self.scroll_frame, image=ctk_page, text="")
                    label.pack(pady=10)
                    
                    self.page_images.append(ctk_page)
                    
            except Exception as e:
                error_label = ctk.CTkLabel(
                    self.scroll_frame,
                    text=f"Error reading PDF: {e}",
                    fg_color="transparent"
                )
                error_label.pack(pady=20)
        else:
            self.label_not_found = ctk.CTkLabel(
                self,
                text="PDF not found!",
                fg_color="transparent",
                font=("Helvetica", 32)
            )
            self.label_not_found.place(relx=0.5, rely=0.5, anchor="center")

        self.home_button = ctk.CTkButton(
            self,
            text="Home Page",
            command=self.go_home,
            corner_radius=0,
            width=150,
            height=40,
            fg_color="#3F1940",
            hover_color="#1a4019"
        )
        self.home_button.place(relx=0.5, rely=0.925, anchor="center")

    def go_home(self):
        # To avoid circular dependency
        from Openbaring1 import HomePage
        current_geometry = self.parent.geometry()
        self.destroy()
        HomePage(self.parent)
