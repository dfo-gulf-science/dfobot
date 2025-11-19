import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import shutil
import csv

# Directories
IN_DIR = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/raw/"
OUT_DIR = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/ref_edge/"
CSV_FILE = "/home/stoyelq/my_hot_storage/dfobot/yellowtail/ref_edges.csv"

# Create out directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)


class GoodnessLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Review App")

        self.labeled_images = self.load_labeled_images()
        self.image_files = self.get_unlabeled_images()
        self.total_images = len(self.image_files)
        self.current_index = 0
        self.history = []

        self.image_height = 1000
        self.image_width = 1000

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
        self.image_label.bind("<Button-1>", self.record_click)

        self.progress_label = tk.Label(root, text="")
        self.progress_label.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.undo_button = tk.Button(self.button_frame, text="Undo (U)", width=10, command=self.undo_last)
        self.undo_button.pack(side=tk.LEFT, padx=10)

        self.root.bind('<u>', lambda e: self.undo_last())

        self.load_next_image()

    def load_labeled_images(self):
        return [f for f in os.listdir(OUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    def get_unlabeled_images(self):
        all_images = [f for f in os.listdir(IN_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        return [f for f in all_images if f not in self.labeled_images]

    def load_next_image(self):
        if self.current_index >= len(self.image_files):
            self.image_label.config(image='', text="No more images to review.")
            self.progress_label.config(text="")
            self.undo_button.config(state=tk.DISABLED)
            return

        filename = self.image_files[self.current_index]
        image_path = os.path.join(IN_DIR, filename)
        image = Image.open(image_path)
        image = image.resize((self.image_width, self.image_height))
        self.tk_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.tk_image)
        self.progress_label.config(text=f"Image {self.current_index + 1} of {self.total_images}")

    def record_click(self, event):
        x_pct = event.x / self.image_width
        y_pct = event.y / self.image_height
        print("clicked at", x_pct, y_pct)

        uuid = self.image_files[self.current_index].split(".")[0]
        filename = self.image_files[self.current_index]
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([uuid, x_pct, y_pct])

        src = os.path.join(IN_DIR, filename)
        dst = os.path.join(OUT_DIR, filename)
        shutil.copy(src, dst)

        self.history.append((filename, uuid, x_pct))
        self.current_index += 1
        self.load_next_image()

    def undo_last(self):
        if not self.history:
            messagebox.showinfo("Undo", "Nothing to undo.")
            return

        filename, uuid, response = self.history.pop()
        self.current_index -= 1

        # Remove image from out dir
        dst = os.path.join(OUT_DIR, filename)
        os.remove(dst)

        # Remove last CSV entry, by overwritting whole thing....
        with open(CSV_FILE, 'r') as f:
            rows = list(csv.reader(f))
            rows = [row for row in rows if row and (row[0] != uuid or row[1] != response)]
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        self.load_next_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = GoodnessLabelerApp(root)
    root.mainloop()
