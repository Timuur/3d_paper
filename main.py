import sys
import os
import time
from PySide6.QtWidgets import *
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap
import trimesh
import pyvista as pv
from pyvistaqt import QtInteractor

import img2wall as i2w
import gen_mod1 as gm1

class ProcessingWorker(QThread):
    # Сигналы для безопасной связи с GUI
    progress = Signal(str, int)  # (текст этапа, процент)
    finished = Signal(object)  # trimesh.Scene
    error = Signal(str)

    def __init__(self, plan_path: str, scale: float, wall_height: float):
        super().__init__()
        self.plan_path = plan_path
        self.scale = scale
        self.wall_height = wall_height * 7.14

    def run(self):
        try:
            # 1. Загрузка и анализ плана ИИ
            self.progress.emit("1/4: Анализ плана нейросетью...", 20)
            # time.sleep(0.3)  # ← замените на: i2w.process_floor_plan(self.plan_path)
            (wall_contours, image_size, filtered_contours_obj) = i2w.process_floor_plan(self.plan_path)

            # 2. Генерация стен
            self.progress.emit("2/4: Построение 3D-геометрии стен...", 45)
            # time.sleep(0.3)  # ← замените на: gm1.build_3d_model(...)
            # scene = gm1.build_3d_model(wall_contours, self.scale, self.wall_height, filtered_contours_obj['Door'],
            #                            filtered_contours_obj['Window'], merge_scene=False, original_image_size=image_size)
            scene = gm1.build_3d_model(wall_contours, self.scale, self.wall_height)
            # scene = trimesh.Scene()  # заглушка для примера

            # 3. Окна/двери/мебель
            # self.progress.emit("3/4: Расстановка проёмов и мебели...", 75)
            self.progress.emit("3/4: Расстановка проёмов...", 55)
            time.sleep(0.3)
            # scene.add_geometry(gm1.build_opening(filtered_contours_obj['Door'], gm1.mesh_door, self.scale, self.wall_height, True))
            scene.add_geometry(gm1.build_door(filtered_contours_obj['Door'], wall_contours, self.scale, self.wall_height))
            self.progress.emit("3/4: Расстановка мебели...", 65)
            # scene.add_geometry(gm1.build_opening(filtered_contours_obj['Window'], gm1.mesh_window, self.scale, self.wall_height, False))
            scene.add_geometry(gm1.build_window(filtered_contours_obj['Window'], wall_contours, self.scale, self.wall_height))
            # self.progress.emit("3/4: Расстановка...", 75)
            scene.add_geometry(gm1.build_obj(filtered_contours_obj, wall_contours, self.scale))
            # self._add_fallback_furniture(scene, filtered_contours_obj['Unknown'])

            # scene.show()

            # 4. Финализация
            self.progress.emit("4/4: Оптимизация и подготовка к экспорту...", 100)
            self.finished.emit(scene)

        except Exception as e:
            self.error.emit(str(e))

    def _add_fallback_furniture(self, scene, unknown_objects):
        """Генерация коробок/цилиндров для нераспознанных объектов"""
        for obj in unknown_objects:
            if obj.get("shape") == "cylinder":
                mesh = trimesh.creation.cylinder(radius=obj["r"], height=obj["h"])
            else:
                mesh = trimesh.creation.box(extents=[obj["w"], obj["d"], obj["h"]])
            mesh.apply_translation(obj["pos"])
            scene.add_geometry(mesh)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("План → 3D Модель")
        self.resize(1200, 800)
        self.current_scene = None
        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # ЛЕВАЯ ПАНЕЛЬ: Управление
        left = QVBoxLayout()

        # Выбор файла
        btn_open = QPushButton("📂 Выбрать план")
        btn_open.clicked.connect(self._select_plan)
        left.addWidget(btn_open)

        # Предпросмотр 2D
        self.preview_label = QLabel("Предпросмотр не выбран")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background: #2a2a2a; border: 1px solid #444;")
        self.preview_label.setMinimumSize(300, 200)
        left.addWidget(self.preview_label)

        # Параметры
        left.addWidget(QLabel("Масштаб (cм/пиксель):"))
        self.input_scale = QDoubleSpinBox()
        self.input_scale.setRange(0.001, 1.0)
        self.input_scale.setValue(0.05)
        left.addWidget(self.input_scale)

        left.addWidget(QLabel("Высота стен (м):"))
        self.input_height = QDoubleSpinBox()
        self.input_height.setRange(0.5, 10.0)
        self.input_height.setValue(2.8)
        left.addWidget(self.input_height)

        # Прогресс
        self.progress_label = QLabel("Готово к работе")
        self.progress_bar = QProgressBar()
        left.addWidget(self.progress_label)
        left.addWidget(self.progress_bar)

        # Кнопка запуска
        self.btn_run = QPushButton("▶ Создать модель")
        self.btn_run.clicked.connect(self._start_processing)
        left.addWidget(self.btn_run)

        # Экспорт
        self.btn_save = QPushButton("💾 Сохранить OBJ")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._export_obj)
        left.addWidget(self.btn_save)

        layout.addLayout(left, stretch=1)

        # ПРАВАЯ ПАНЕЛЬ: 3D Вьюпорт
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")
        layout.addWidget(self.plotter, stretch=2)

    def _select_plan(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите план", "", "Images (*.png *.jpg *.jpeg *.svg)")
        if not path: return

        # Обновляем превью
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.preview_label.setText("Ошибка загрузки")
            return
        self.preview_label.setPixmap(pixmap.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self._plan_path = path

    def _start_processing(self):
        if not hasattr(self, '_plan_path'):
            QMessageBox.warning(self, "Внимание", "Сначала выберите план!")
            return

        # Блокируем интерфейс, сбрасываем прогресс
        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Запуск...")

        # Запуск потока
        self.worker = ProcessingWorker(
            self._plan_path,
            self.input_scale.value(),
            self.input_height.value()
        )
        self.worker.progress.connect(self._update_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _update_progress(self, text: str, value: int):
        self.progress_label.setText(text)
        self.progress_bar.setValue(value)

    def _on_finished(self, scene: trimesh.Scene):
        self.current_scene = scene
        # Рендер в QtInteractor
        self.plotter.clear()
        # self.plotter.add_mesh(scene.dump(), color="lightblue", show_edges=True, smooth_shading=True)

        # dump() может вернуть список, один меш или None
        meshes = scene.dump()
        if meshes is None:
            self.statusBar().showMessage("⚠️ Сцена пуста", 3000)
            self.btn_run.setEnabled(True)
            return

        if not isinstance(meshes, list):
            meshes = [meshes]

        for mesh in meshes:
            if mesh is not None:
                # PyVista автоматически конвертирует trimesh.Trimesh в PolyData
                self.plotter.add_mesh(mesh, color="lightblue", show_edges=True, smooth_shading=True)

        self.plotter.reset_camera()

        self.progress_label.setText("✅ Модель готова")
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(True)

    def _on_error(self, msg: str):
        self.progress_label.setText(f"❌ Ошибка: {msg}")
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Ошибка обработки", msg)

    def _export_obj(self):
        if not self.current_scene: return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить модель", "", "OBJ (*.obj)")
        if path:
            self.current_scene.export(path, file_type="obj")
            QMessageBox.information(self, "Успех", f"Модель сохранена:\n{path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Современный стиль
    win = MainWindow()
    win.show()
    sys.exit(app.exec())