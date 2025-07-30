import customtkinter as ctk
import tkinter as tk
import database
import config

from PIL import Image


class Settings(ctk.CTkFrame):
    def __init__(self, parent, notify_callback=None, close_callback=None):
        parent.update_idletasks()
        p_width = parent.winfo_width()
        p_height = parent.winfo_height()
        overlay_width = int(p_width * 0.5)
        overlay_height = int(p_height * 0.5)
        super().__init__(parent, width=overlay_width, height=overlay_height, fg_color="black")
        self.notify_callback = notify_callback
        self.close_callback = close_callback
        self.place(relx=0.5, rely=0.5, anchor="center")
        self.title_label = ctk.CTkLabel(
            self,
            text="Settings",
            font=("Helvetica", 24),
            text_color="white",
            fg_color="black",
            bg_color="black"
        )
        self.title_label.place(relx=0.5, rely=0.1, anchor="center")
        
        exit_img = ctk.CTkImage(Image.open("assets/icons/rectangle-xmark.png"), size=(20, 20))
        self.close_button = ctk.CTkButton(
            self,
            image=exit_img,
            text="",
            command=self.settings_sheet,
            corner_radius=0,
            width=30,
            height=30,
            fg_color="#3F1940",
            hover_color="#321433",
            text_color="black"
        )
        self.close_button.place(relx=0.955, rely=0.075, anchor="center")
        
        self.mask_segmentation_label = ctk.CTkLabel(
            self,
            text="Mask segmentation type",
            font=("Helvetica", 14, "italic"),
            text_color="white",
            fg_color="black"
        )
        self.mask_segmentation_label.place(relx=0.5, rely=0.24, anchor="center")
        
        self.face_method_variable = tk.IntVar(value=config.FACE_MASK_METHOD)
        
        self.radio1 = ctk.CTkRadioButton(
            self,
            text="Detectron2 (Local)",
            variable=self.face_method_variable,
            value=1,
            command=self.update_face_method,
            fg_color="#3F1940",
            hover_color="#321433",
            text_color="white"
        )
        self.radio1.place(relx=0.5, rely=0.32, anchor="center")
        
        self.radio2 = ctk.CTkRadioButton(
            self,
            text="Roboflow (Cloud)",
            variable=self.face_method_variable,
            value=2,
            command=self.update_face_method,
            fg_color="#3F1940",
            hover_color="#321433",
            text_color="white"
        )
        self.radio2.place(relx=0.5, rely=0.42, anchor="center")
        
        self.inpainting_model_label = ctk.CTkLabel(
            self,
            text="Inpainting model",
            font=("Helvetica", 14, "italic"),
            text_color="white",
            fg_color="black"
        )
        self.inpainting_model_label.place(relx=0.5, rely=0.6, anchor="center")
        
        self.inference_model_variable = tk.IntVar(value=config.INFERENCE_MODEL)
        
        self.model_radio1 = ctk.CTkRadioButton(
            self,
            text="GANs (no mapping)",
            variable=self.inference_model_variable,
            value=1,
            command=self.update_inference_model,
            fg_color="#3F1940",
            hover_color="#321433",
            text_color="white"
        )
        self.model_radio1.place(relx=0.5, rely=0.68, anchor="center")
        
        self.model_radio2 = ctk.CTkRadioButton(
            self,
            text="GANs (mapping)",
            variable=self.inference_model_variable,
            value=2,
            command=self.update_inference_model,
            fg_color="#3F1940",
            hover_color="#321433",
            text_color="white"
        )
        self.model_radio2.place(relx=0.5, rely=0.78, anchor="center")
        
        self.model_radio3 = ctk.CTkRadioButton(
            self,
            text="EchoShiftGAN (mapping)",
            variable=self.inference_model_variable,
            value=3,
            command=self.update_inference_model,
            fg_color="#3F1940",
            hover_color="#321433",
            text_color="white"
        )
        self.model_radio3.place(relx=0.5, rely=0.88, anchor="center")
        
    def update_face_method(self):
        config.FACE_MASK_METHOD = self.face_method_variable.get()
        if self.face_method_variable.get() == 1:
            self.notify_callback("Detectron2 segmentation model selected.")
        elif self.face_method_variable.get() == 2:
            self.notify_callback("Roboflow segmentation model selected.")
            
    def update_inference_model(self):
        config.INFERENCE_MODEL = self.inference_model_variable.get()
        if config.INFERENCE_MODEL == 1:
            self.notify_callback("GAN model selected for inference.")
        elif config.INFERENCE_MODEL == 2:
            self.notify_callback("Mapping‑enabled GAN selected for inference.")
        elif config.INFERENCE_MODEL == 3:
            self.notify_callback("EchoShiftGAN selected: multi-output inference with landmark variation.")

    def settings_sheet(self):
        if self.close_callback:
            self.close_callback()
        self.destroy()


class LoginOverlay(ctk.CTkFrame):
    def __init__(self, parent, login_success_callback=None, switch_to_register_callback=None, notify_callback=None, close_callback=None):
        parent.update_idletasks()
        p_width = parent.winfo_width()
        p_height = parent.winfo_height()
        overlay_width = int(p_width * 0.5)
        overlay_height = int(p_height * 0.5)
        super().__init__(parent, width=overlay_width, height=overlay_height, fg_color="black")
        self.login_success_callback = login_success_callback
        self.switch_to_register_callback = switch_to_register_callback
        self.notify_callback = notify_callback
        self.close_callback = close_callback
        self.place(relx=0.5, rely=0.5, anchor="center")
        
        self.title_label = ctk.CTkLabel(
            self,
            text="Login",
            font=("Helvetica", 24),
            text_color="white",
            fg_color="black"
        )
        self.title_label.place(relx=0.5, rely=0.1, anchor="center")
        
        self.message_label = ctk.CTkLabel(
            self,
            text="Are you already registered?",
            font=("Helvetica", 14),
            text_color="white",
            fg_color="black"
        )
        self.message_label.place(relx=0.5, rely=0.2, anchor="center")
        
        exit_img = ctk.CTkImage(Image.open("assets/icons/rectangle-xmark.png"), size=(20, 20))
        self.close_button = ctk.CTkButton(
            self,
            image=exit_img,
            text="",
            command=self.close_overlay,
            corner_radius=0,
            width=30,
            height=30,
            text_color="black",
            fg_color="#1a4019",
            hover_color="#143314"
        )
        self.close_button.place(relx=0.955, rely=0.075, anchor="center")
        
        self.form_frame = ctk.CTkFrame(
            self,
            width=overlay_width * 0.8,
            height=overlay_height * 0.6,
            fg_color="grey",
            corner_radius=0
        )
        self.form_frame.place(relx=0.5, rely=0.55, anchor="center")
        self.form_frame.grid_propagate(False)
        half_width = int((overlay_width * 0.8) / 2)
        self.form_frame.columnconfigure(0, minsize=half_width, pad=10)
        self.form_frame.columnconfigure(1, minsize=half_width, pad=10)
        
        field_font = ("Helvetica", 14)

        self.staff_id_label = ctk.CTkLabel(
            self.form_frame,
            text="Staff ID:",
            font=field_font,
            text_color="black",
            fg_color="grey"
        )
        
        self.staff_id_label.grid(row=0, column=0, sticky="e", padx=10, pady=(60,10))
        self.staff_id_entry = ctk.CTkEntry(
            self.form_frame,
            placeholder_text="Enter Staff ID",
            text_color="black",
            fg_color="white",
            corner_radius=0,
            font=field_font
        )
        self.staff_id_entry.grid(row=0, column=1, sticky="w", padx=10, pady=(60,10))
        
        self.password_label = ctk.CTkLabel(
            self.form_frame,
            text="Password:",
            font=field_font,
            text_color="black",
            fg_color="grey"
        )
        self.password_label.grid(row=1, column=0, sticky="e", padx=10, pady=10)
        self.password_entry = ctk.CTkEntry(
            self.form_frame,
            placeholder_text="Enter Password",
            text_color="black",
            fg_color="white",
            show="*",
            corner_radius=0,
            font=field_font
        )
        self.password_entry.grid(row=1, column=1, sticky="w", padx=10, pady=10)
        
        self.form_frame.rowconfigure(2, weight=1)
        
        self.enter_button = ctk.CTkButton(
            self.form_frame,
            text="Enter",
            font=field_font,
            command=self.login_action,
            corner_radius=0,
            text_color="white",
            fg_color="#1a4019",
            hover_color="#143314"
        )
        self.enter_button.grid(row=3, column=0, columnspan=2, sticky="s", pady=(0,10))
        self.form_frame.grid_columnconfigure(0, weight=1)
        self.form_frame.grid_columnconfigure(1, weight=1)
        
        self.link_frame = ctk.CTkFrame(
            self,
            width=overlay_width * 0.8,
            height=30,
            fg_color="black"
        )
        self.link_frame.place(relx=0.5, rely=0.92, anchor="center")
        self.new_here_label = ctk.CTkLabel(
            self.link_frame,
            text="New here? ",
            font=field_font,
            text_color="white",
            fg_color="black"
        )
        self.new_here_label.pack(side="left")
        self.register_link = ctk.CTkLabel(
            self.link_frame,
            text="Register",
            font=(field_font[0], field_font[1], "underline"),
            text_color="#1a4019",
            fg_color="black",
            cursor="hand2"
        )
        self.register_link.pack(side="left")
        self.register_link.bind("<Button-1>", self.switch_to_register)
    
    def login_action(self):
        staff_id = self.staff_id_entry.get()
        password = self.password_entry.get()
        result = database.login_user(staff_id, password)
        
        if isinstance(result, tuple):
            message, staff_id, full_name = result
            #******************************#
            if self.notify_callback:
                self.notify_callback(message)
            #******************************#
            if self.login_success_callback:
                self.master.session["password"] = password
                self.login_success_callback(staff_id, full_name)
            self.destroy()
            return result
        else:
            if self.notify_callback:
                self.notify_callback(result)
            # print(result)
            return result
    
    def switch_to_register(self, event):
        if self.switch_to_register_callback:
            self.switch_to_register_callback()
        else:
            self.destroy()
            from overlays import RegisterOverlay
            RegisterOverlay(self.master)

    def close_overlay(self):
        if self.close_callback:
            self.close_callback()
        self.destroy()


class RegisterOverlay(ctk.CTkFrame):
    def __init__(self, parent, switch_to_login_callback=None, notify_callback=None, close_callback=None):
        parent.update_idletasks()
        p_width = parent.winfo_width()
        p_height = parent.winfo_height()
        overlay_width = int(p_width * 0.5)
        overlay_height = int(p_height * 0.5)
        super().__init__(parent, width=overlay_width, height=overlay_height, fg_color="black")
        self.switch_to_login_callback = switch_to_login_callback
        self.notify_callback = notify_callback
        self.close_callback = close_callback
        self.place(relx=0.5, rely=0.5, anchor="center")
        
        self.title_label = ctk.CTkLabel(
            self,
            text="Register",
            font=("Helvetica", 24),
            text_color="white",
            fg_color="black"
        )
        self.title_label.place(relx=0.5, rely=0.1, anchor="center")
        
        self.message_label = ctk.CTkLabel(
            self,
            text="New user? Create an account below",
            font=("Helvetica", 14),
            text_color="white",
            fg_color="black"
        )
        self.message_label.place(relx=0.5, rely=0.2, anchor="center")
        
        exit_img = ctk.CTkImage(Image.open("assets/icons/rectangle-xmark.png"), size=(20, 20))
        self.close_button = ctk.CTkButton(
            self,
            image=exit_img,
            text="",
            command=self.close_overlay,
            corner_radius=0,
            width=30,
            height=30,
            text_color="black",
            fg_color="#1a4019",
            hover_color="#143314"
        )
        self.close_button.place(relx=0.955, rely=0.075, anchor="center")
        
        self.form_frame = ctk.CTkFrame(
            self,
            width=overlay_width * 0.8,
            height=overlay_height * 0.6,
            fg_color="grey",
            corner_radius=0
        )
        self.form_frame.place(relx=0.5, rely=0.55, anchor="center")
        self.form_frame.grid_propagate(False)
        half_width = int((overlay_width * 0.8) / 2)
        self.form_frame.columnconfigure(0, minsize=half_width, pad=10)
        self.form_frame.columnconfigure(1, minsize=half_width, pad=10)
        
        field_font = ("Helvetica", 12)
        
        self.firstname_label = ctk.CTkLabel(
            self.form_frame,
            text="First Name:",
            font=field_font,
            text_color="black",
            fg_color="grey"
        )
        self.firstname_label.grid(row=0, column=0, sticky="e", padx=10, pady=5)
        self.firstname_entry = ctk.CTkEntry(
            self.form_frame,
            placeholder_text="Enter First Name",
            text_color="black",
            fg_color="white",
            corner_radius=0
        )
        self.firstname_entry.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        
        self.surname_label = ctk.CTkLabel(
            self.form_frame,
            text="Surname:",
            font=field_font,
            text_color="black",
            fg_color="grey"
        )
        self.surname_label.grid(row=1, column=0, sticky="e", padx=10, pady=5)
        self.surname_entry = ctk.CTkEntry(
            self.form_frame,
            placeholder_text="Enter Surname",
            text_color="black",
            fg_color="white",
            corner_radius=0
        )
        self.surname_entry.grid(row=1, column=1, sticky="w", padx=10, pady=5)
        
        self.staff_id_label = ctk.CTkLabel(
            self.form_frame,
            text="Staff ID:",
            font=field_font,
            text_color="black",
            fg_color="grey"
        )
        self.staff_id_label.grid(row=2, column=0, sticky="e", padx=10, pady=5)
        self.staff_id_entry = ctk.CTkEntry(
            self.form_frame,
            placeholder_text="Enter Staff ID",
            text_color="black",
            fg_color="white",
            corner_radius=0
        )
        self.staff_id_entry.grid(row=2, column=1, sticky="w", padx=10, pady=5)
        
        self.password_label = ctk.CTkLabel(
            self.form_frame,
            text="Password:",
            font=field_font,
            text_color="black",
            fg_color="grey"
        )
        self.password_label.grid(row=3, column=0, sticky="e", padx=10, pady=5)
        self.password_entry = ctk.CTkEntry(
            self.form_frame,
            placeholder_text="Enter Password",
            text_color="black",
            fg_color="white",
            show="*",
            corner_radius=0
        )
        self.password_entry.grid(row=3, column=1, sticky="w", padx=10, pady=5)
        
        self.confirm_label = ctk.CTkLabel(
            self.form_frame,
            text="Confirm Password:",
            font=field_font,
            text_color="black",
            fg_color="grey"
        )
        self.confirm_label.grid(row=4, column=0, sticky="e", padx=10, pady=5)
        self.confirm_entry = ctk.CTkEntry(
            self.form_frame,
            placeholder_text="Confirm Password",
            text_color="black",
            fg_color="white",
            show="*",
            corner_radius=0
        )
        self.confirm_entry.grid(row=4, column=1, sticky="w", padx=10, pady=5)
        
        self.form_frame.rowconfigure(5, weight=1)
        self.register_button = ctk.CTkButton(
            self.form_frame,
            text="Register",
            font=field_font,
            command=self.register_action,
            corner_radius=0,
            text_color="white",
            fg_color="#1a4019",
            hover_color="#143314"
        )
        self.register_button.grid(row=6, column=0, columnspan=2, sticky="s", pady=(0,10))
        self.form_frame.grid_columnconfigure(0, weight=1)
        self.form_frame.grid_columnconfigure(1, weight=1)
        
        self.link_frame = ctk.CTkFrame(
            self,
            width=overlay_width * 0.8,
            height=30,
            fg_color="black"
        )
        self.link_frame.place(relx=0.5, rely=0.92, anchor="center")
        self.already_label = ctk.CTkLabel(
            self.link_frame,
            text="Already registered? ",
            font=field_font,
            text_color="white",
            fg_color="black"
        )
        self.already_label.pack(side="left")
        self.login_link = ctk.CTkLabel(
            self.link_frame,
            text="Login",
            font=(field_font[0], field_font[1], "underline"),
            text_color="#1a4019",
            fg_color="black",
            cursor="hand2"
        )
        self.login_link.pack(side="left")
        self.login_link.bind("<Button-1>", self.switch_to_login)
    
    def register_action(self):
        first_name = self.firstname_entry.get()
        surname = self.surname_entry.get()
        staff_id = self.staff_id_entry.get()
        password = self.password_entry.get()
        confirm_password = self.confirm_entry.get()
        
        message = database.register_user(staff_id, first_name, surname, password, confirm_password)
        
        if self.notify_callback:
            self.notify_callback(message)
        
    def switch_to_login(self, event):
        self.destroy()
        if self.switch_to_login_callback:
            self.switch_to_login_callback()
        else:
            from overlays import LoginOverlay
            LoginOverlay(self.master)
    
    def close_overlay(self):
        if self.close_callback:
            self.close_callback()
        self.destroy()
