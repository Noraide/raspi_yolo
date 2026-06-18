import tkinter as tk
import os

# Setup window
root = tk.Tk()
root.title("Raspberry Pi Temperature")

label = tk.Label(root, font=("Ubuntu", 20), background="black", foreground="orange")
label.pack(padx=20, pady=20)

def update_temp():
    temp = os.popen("vcgencmd measure_temp").readline().replace("temp=", "").strip()
    label.config(text=f"CPU Temp: {temp}")
    label.after(2000, update_temp)  # refresh setiap 2 saat

update_temp()
root.mainloop()
