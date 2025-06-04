from tkinter import *
from tkinter import ttk, filedialog
from PIL import ImageTk, Image

#______________________________________________________________________________________________________________________
# ICON = zlib.decompress(base64.b64decode("eJxjYGAEQgEBBiDJwZDBysAgxsDAoAHEQCEGBQaIOAg4sDIgACMUj4JRMApGwQgF/ykEAFXxQRc="))
# _, ICON_PATH = tempfile.mkstemp()
# with open(ICON_PATH, "wb") as icon_file:
#     icon_file.write(ICON)
#
file_t = [("JPG Files", "*.jpg"), ("PNG Files", "*.png"), ("JPEG Files", "*.jpeg"), ("SVG Files", "*.svg")]


#______________________________________________________________________________________________________________________
root = Tk()
root.title("План в модель")
root.geometry("950x700")

# root.iconbitmap(default=ICON_PATH)

def finish():
    root.destroy()  # ручное закрытие окна и всего приложения
    print("Закрытие приложения")
root.protocol("WM_DELETE_WINDOW", finish)

#______________________________________________________________________________________________________________________
import img2wall as i2w
import gen_mod1 as gm1
#______________________________________________________________________________________________________________________
def click():

    plan = entry_f.get()
    model_scale = entry_ms.get()
    wall_hight = entry_wh.get()
    p2m = entry_mod.get()
    print(plan)
    print(model_scale)
    print(wall_hight)
    print(p2m)

    try:
        # Обработка плана помещения
        (wall_contours, image_size, filtered_contours_door, filtered_contours_window,
         filtered_contours_box, filtered_contours_toilet) = i2w.process_floor_plan(plan)

        # Параметры моделирования
        # model_scale = 0.05 # 1 пиксель = 5 см
        # wall_hight = 20  # Высота потолков 2.7 метра

        # Создание 3D-модели
        scene1 = gm1.build_3d_model(
            wall_contours,
            image_size,
            model_scale,
            wall_hight,
        )

        scene1.add_geometry(gm1.build_door(filtered_contours_door, model_scale))
        scene1.add_geometry(gm1.build_window(filtered_contours_window, wall_contours, model_scale, wall_hight))

        # Визуализация результата
        scene1.show()
        # filename = filedialog.asksaveasfilename()
        # # Экспорт модели (опционально)
        # scene1.export(filename)

    except FileNotFoundError as e:
        print(f"Ошибка загрузки файла: {e}")
    # except ValueError as e:
    #     print(f"Ошибка обработки данных: {e}")
    # except Exception as e:
    #     print(f"Неизвестная ошибка: {e}")

#______________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________
e_ms = DoubleVar()
e_wh = DoubleVar()
e_esp = DoubleVar()
e_img = StringVar()
e_mod = StringVar()

def img_read():
    entry_f.delete(0, last="end")
    name_img = filedialog.askopenfilename(title = "Выбор плана", filetypes = file_t)
    print(f"Выбран план - {name_img}")
    entry_f.insert(0, name_img)
    if name_img:
        try:
            # Открываем изображение с помощью PIL
            img = Image.open(name_img)
            w, h = img.size
            w= w / 4
            h = h / 4
            img = img.resize((int(w), int(h)), Image.Resampling.LANCZOS)
            # Сохраняем ссылку на новое изображение
            current_image = ImageTk.PhotoImage(img)
            # Обновляем Label
            entry_p.configure(image=current_image)
            entry_p.image = current_image
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")

label_f = ttk.Label(text = "Файл")
label_f.grid(row=0, column=0, sticky = 'w', padx=[15,0], pady=[15, 4])
entry_f = ttk.Entry()
entry_f.grid(row=1, column=0, padx=[15,0])
entry_f.insert(0,"Выберите план")

entry_p = ttk.Label()
entry_p.grid(row=1, column=1, padx=[5,0], rowspan=160)

# img = Image.open('G:/Pract_Prog/Pract_Prog_F/kristal/1k.png')
# img = img.resize((300, 200), Image.Resampling.LANCZOS)
# current_image = ImageTk.PhotoImage(img)
# entry_p['image'] = current_image

label_ms = ttk.Label(text = "Множетель размера")
label_ms.grid(row=2, column=0, sticky = 'w', padx=[15,0], pady=[5, 4])
entry_ms = ttk.Entry(textvariable=e_ms)
entry_ms.grid(row=3, column=0, padx=[15,0])
e_ms.set(0.05)

label_wh = ttk.Label(text = "Высота стен")
label_wh.grid(row=4, column=0, sticky = 'w', padx=[15,0], pady=[5, 4])
entry_wh = ttk.Entry(textvariable=e_wh)
entry_wh.grid(row=5, column=0, padx=[15,0])
e_wh.set(20)

label_mod = ttk.Label(text = "Название модели(необязательно)")
label_mod.grid(row=6, column=0, sticky = 'w', padx=[15,0], pady=[5, 4])
entry_mod = ttk.Entry(textvariable=e_mod)
entry_mod.grid(row=7, column=0, padx=[15,0])
entry_mod.insert(0,"model.obj")
#______________________________________________________________________________________________________________________

button = ttk.Button(text="Показать план", command=img_read)
button.grid(row=0, column=1, pady=10)
button = ttk.Button(text="Создать модель", command=click)
button.grid(row=10, column=0, pady=10)
#______________________________________________________________________________________________________________________

root.mainloop()