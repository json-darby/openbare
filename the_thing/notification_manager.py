import customtkinter as ctk
import tkinter as tk

class NotificationManager(ctk.CTkFrame):
    def __init__(self, parent, width=350, height=60, settle_y=0,
                 bg_color="black", text_color="gold", *args, **kwargs):
        super().__init__(parent, width=width, height=height, fg_color=bg_color, *args, **kwargs)
        self.parent = parent
        self.width = width
        self.height = height
        self.settle_y = settle_y  # Final y-coordinate, akin to iphone dynamic island...
        self.bg_color = bg_color
        self.text_color = text_color

        self.queue = []  # Queue for messages.
        self.visible = False  # Indicates if the bar is already visible

        self.message_label = ctk.CTkLabel(self, text="", fg_color=bg_color,
                                          text_color=text_color,
                                          font=("Helvetica", 14, "bold"))
        self.message_label.place(relx=0.5, rely=0.5, anchor="center")

        self.place_forget()  # Hidden initially

    def add_message(self, message):
        # Add a message to the queue and, if the bar isn’t visible, start the notification
        self.queue.append(message)
        if not self.visible:
            self.start_notification()

    def start_notification(self):
        self.visible = True
        self.parent.update_idletasks()
        self.place(relx=0.5, y=-self.height, anchor="n")
        self.animate_slide(-self.height, self.settle_y, direction="down", callback=self.display_message)

    def display_message(self):
        # After the bar is fully down, display the first message in the queue
        if self.queue:
            message = self.queue.pop(0)
            if len(message) <= 20:
                new_font_size = 16
                self.message_label.configure(text=message, font=("Helvetica", new_font_size, "bold"))
                self.after(3000, self.notification_queue)
            elif len(message) < 45:
                new_font_size = 14
                self.message_label.configure(text=message, font=("Helvetica", new_font_size, "bold"))
                self.after(3000, self.notification_queue)
            else:
                self.slide_message(message)
        else:
            self.hide_notification()

    def notification_queue(self):
        if self.queue:
            # After a short delay, process the next message.
            self.after(1000, self.display_message)
        else:
            self.hide_notification()

    def hide_notification(self):
        current_y = int(self.place_info()['y'])
        self.animate_slide(current_y, -self.height, direction="up", callback=self.no_notifications)

    def no_notifications(self):
        self.place_forget()
        self.visible = False

    def animate_slide(self, start_y, end_y, direction="down", callback=None, step=10, delay=20):
        if direction == "down":
            if start_y < end_y:
                new_y = min(start_y + step, end_y)
                self.place_configure(y=new_y)
                self.after(delay, lambda: self.animate_slide(new_y, end_y, direction, callback, step, delay))
            else:
                if callback:
                    callback()
        elif direction == "up":
            if start_y > end_y:
                new_y = max(start_y - step, end_y)
                self.place_configure(y=new_y)
                self.after(delay, lambda: self.animate_slide(new_y, end_y, direction, callback, step, delay))
            else:
                if callback:
                    callback()

    def slide_message(self, message):
        self.message_label.configure(text="")
        sliding_label = ctk.CTkLabel(self, text=message, font=("Helvetica", 14, "bold"),
                                     text_color=self.text_color, fg_color=self.bg_color)
        sliding_label.place(relx=1.1, rely=0.5, anchor="w")
        def slide(current_relx=0.3):
            if current_relx > -1.2:
                sliding_label.place_configure(relx=current_relx)
                self.after(20, slide, current_relx - 0.005)
            else:
                sliding_label.destroy()
                self.notification_queue()
        slide()
