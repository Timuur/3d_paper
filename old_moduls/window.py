from tkinter import *
from tkinter import ttk
import tempfile, base64, zlib

ICON = zlib.decompress(base64.b64decode("eJxjYGAEQgEBBiDJwZDBysAgxsDAoAHEQCEGBQaIOAg4sDIgACMUj4JRMApGwQgF/ykEAFXxQRc="))

_, ICON_PATH = tempfile.mkstemp()
with open(ICON_PATH, "wb") as icon_file:
    icon_file.write(ICON)

root = Tk()
root.title("План в модель")
root.geometry("950x700")
root.iconbitmap(default=ICON_PATH)
def finish():
    root.destroy()  # ручное закрытие окна и всего приложения
    print("Закрытие приложения")
root.protocol("WM_DELETE_WINDOW", finish)

def click():
    window = Tk()
    window.title("Новое окно")
    window.geometry("250x200")
    label = ttk.Label(window, text="Принципиально новое окно")
    label.pack(anchor=CENTER, expand=1)
#
button = ttk.Button(text="Обработка плана", command=click)
button.pack(anchor=CENTER, expand=1)
button = ttk.Button(text="Создать модель", command=click)
button.pack(anchor=CENTER, expand=1)

root.mainloop()