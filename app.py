import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import threading
import time

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("model/cnn_model.h5")
classes = {0: "🐱 CAT", 1: "🐶 DOG"}

# ---------------- ROOT WINDOW ----------------
root = tk.Tk()
root.title("🐶🐱 Cat vs Dog Classifier")
root.geometry("850x750")  # thoda chhota height
root.configure(bg="#e0f2fe")  # soft blue
root.minsize(700, 650)

# ---------------- HEADER ----------------
header = tk.Frame(root, bg="#3b82f6", height=80)
header.pack(fill="x")

tk.Label(
    header,
    text="🐶🐱 Cat vs Dog Image Classifier",
    font=("Segoe UI", 22, "bold"),
    fg="white",
    bg="#3b82f6"
).pack(pady=20)

# ---------------- MAIN BODY ----------------
body = tk.Frame(root, bg="#e0f2fe")
body.pack(fill="both", expand=True, padx=10, pady=10)

# ---------------- SCROLLABLE RESULT FRAME ----------------
container = tk.Frame(body, bg="#e0f2fe")
container.pack(fill="both", expand=True)

canvas = tk.Canvas(container, bg="#e0f2fe", highlightthickness=0)
scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
result_frame = tk.Frame(canvas, bg="#e0f2fe")

result_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0,0), window=result_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# ---------------- AUTO ALERT FUNCTION ----------------
def show_alert(message, duration=2500):
    alert = tk.Label(root, text=message,
                     font=("Segoe UI", 14, "bold"),
                     bg="#3b82f6", fg="white", bd=2, relief="raised")
    alert.place(relx=0.5, rely=0.02, anchor="n")
    root.after(duration, alert.destroy)

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(path):
    img = Image.open(path).convert("RGB")
    img_model = img.resize((150, 150))
    img_array = np.array(img_model) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)[0][0]
    cls = int(pred >= 0.5)

    label = classes[cls]
    confidence = pred*100 if cls==1 else (1-pred)*100
    return label, confidence, img

def upload_images():
    files = filedialog.askopenfilenames(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not files:
        return

    # clear previous results
    clear_results()

    def process_images():
        for path in files:
            label, conf, img = predict_image(path)

            # ---------------- CARD ----------------
            card = tk.Frame(result_frame, bg="white", bd=0, highlightthickness=2, highlightbackground="#3b82f6")
            card.pack(pady=10, padx=10, fill="x")

            # Hover animation
            def on_enter(e, c=card):
                c.config(bg="#bfdbfe")
            def on_leave(e, c=card):
                c.config(bg="white")
            card.bind("<Enter>", on_enter)
            card.bind("<Leave>", on_leave)

            # ---------------- IMAGE ----------------
            imgtk = ImageTk.PhotoImage(img.resize((120,120)))
            img_lbl = tk.Label(card, image=imgtk, bg="white")
            img_lbl.image = imgtk
            img_lbl.pack(side="left", padx=10, pady=10)

            # ---------------- TEXT ----------------
            text_frame = tk.Frame(card, bg="white")
            text_frame.pack(side="left", fill="both", expand=True, padx=10)

            tk.Label(
                text_frame,
                text=label,
                font=("Segoe UI", 16, "bold"),
                bg="white",
                fg="#16a34a" if "CAT" in label else "#ef4444"
            ).pack(anchor="w")

            # ---------------- CONFIDENCE BAR ----------------
            conf_frame = tk.Frame(text_frame, bg="white")
            conf_frame.pack(fill="x", pady=5)
            conf_bar = ttk.Progressbar(conf_frame, orient="horizontal", length=300, mode="determinate")
            conf_bar.pack(side="left", pady=5)
            tk.Label(
                conf_frame,
                text=f"{conf:.2f}%",
                font=("Segoe UI", 12),
                bg="white",
                fg="#1e40af"
            ).pack(side="left", padx=10)

            # Animate progress bar
            def animate_bar(value):
                for i in range(int(value)+1):
                    conf_bar['value'] = i
                    time.sleep(0.005)
            threading.Thread(target=animate_bar, args=(conf,)).start()

            # Auto alert
            show_alert(f"{label} predicted with {conf:.2f}% confidence!", 2000)

    threading.Thread(target=process_images).start()

# ---------------- BUTTONS ----------------
btn_frame = tk.Frame(body, bg="#e0f2fe")
btn_frame.pack(pady=15)

def create_btn(master, text, bg_color, fg_color, command):
    btn = tk.Button(
        master,
        text=text,
        command=command,
        font=("Segoe UI", 14, "bold"),
        bg=bg_color,
        fg=fg_color,
        activebackground=bg_color,
        activeforeground=fg_color,
        bd=0,
        relief="flat",
        padx=25,
        pady=12,
        cursor="hand2"
    )
    # hover effect
    def on_enter(e, b=btn):
        b.config(bg="#60a5fa")
    def on_leave(e, b=btn, bg_color=bg_color):
        b.config(bg=bg_color)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn

upload_btn = create_btn(btn_frame, "📂 Upload Images", "#3b82f6", "white", upload_images)
upload_btn.grid(row=0, column=0, padx=10)

clear_btn = create_btn(btn_frame, "🧹 Clear", "#60a5fa", "white", lambda: clear_results())
clear_btn.grid(row=0, column=1, padx=10)

exit_btn = create_btn(btn_frame, "❌ Exit", "#ef4444", "white", root.destroy)
exit_btn.grid(row=0, column=2, padx=10)


# ---------------- CLEAR FUNCTION ----------------
def clear_results():
    for widget in result_frame.winfo_children():
        widget.destroy()

# ---------------- STYLE ----------------
style = ttk.Style()
style.theme_use('clam')
style.configure("TProgressbar", thickness=20, troughcolor="#e0f2fe", background="#3b82f6", bordercolor="#e0f2fe")

# ---------------- RUN ----------------
root.mainloop()