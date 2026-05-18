import sys
import os
import time
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QPolygonF, QColor, QCursor
from typing import Optional

import trimesh
import pyvista as pv
from pyvistaqt import QtInteractor

import img2wall as i2w
import gen_mod1 as gm1


class ProcessingWorker(QThread):
    progress = Signal(str, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, plan_path: str, scale: float, wall_height: float, edited_regions: Optional[dict] = None):
        super().__init__()
        self.plan_path = plan_path
        self.scale = scale
        self.wall_height = wall_height * 7.14  # коэффициент масштабирования
        self.edited_regions: Optional[dict] = edited_regions

    def run(self):
        try:
            # 1. Анализ плана
            self.progress.emit("1/4: Анализ плана нейросетью...", 20)
            # (wall_contours, image_size, _) = i2w.process_floor_plan(self.plan_path)

            wall_contours_cv, image_size, raw_contours = i2w.process_floor_plan(self.plan_path)

            # Если пользователь правил области в GUI → используем их, иначе → сырой ИИ-вывод
            contours = self.edited_regions if self.edited_regions else raw_contours

            # Стены часто нужны отдельно для построения геометрии
            # if wall_contours_cv:
            #     wall_contours = wall_contours_cv
            # else:
            #     wall_contours = contours.get('Wall', raw_contours.get('Wall', []))
            wall_contours = contours.get('Wall', raw_contours.get('Wall', []))
            h_wall_contours = contours.get('h-wall', raw_contours.get('h-wall', []))

            # 2. Генерация стен
            self.progress.emit("2/4: Построение 3D-геометрии стен...", 45)
            scene = gm1.build_3d_model(wall_contours, self.scale, self.wall_height)
            scene.add_geometry(gm1.build_3d_model(h_wall_contours, self.scale, self.wall_height))

            # 3. Проёмы и мебель
            self.progress.emit("3/4: Расстановка проёмов и мебели...", 75)
            if 'Door' in contours:
                scene.add_geometry(gm1.build_door(contours['Door'], wall_contours, self.scale, self.wall_height))
            if 'Window' in contours:
                scene.add_geometry(gm1.build_window(contours['Window'], wall_contours, self.scale, self.wall_height))
            # Остальные объекты (GasPlate, sink, box, toulet и т.д.)
            scene.add_geometry(gm1.build_obj(contours, wall_contours, self.scale))

            # 4. Финализация (scene.show() убран, чтобы не блокировать GUI)
            self.progress.emit("4/4: Оптимизация и подготовка...", 100)
            self.finished.emit(scene)

        except Exception as e:
            self.error.emit(str(e))

# ================= 2D РЕДАКТОР ПЛАНА =================
class EditableRegionItem(QGraphicsObject):
    region_modified = Signal(object, str)
    region_deleted = Signal(object)

    def __init__(self, polygon: list, region_class: str = "Unknown", parent=None):
        super().__init__(parent)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.region_class = region_class
        self.drag_index = -1
        self.HANDLE_SIZE = 6
        self.points = [QPointF(float(p[0]), float(p[1])) for p in polygon if len(p) >= 2]

    def boundingRect(self) -> QRectF:
        if not self.points: return QRectF()
        min_x = min(p.x() for p in self.points) - self.HANDLE_SIZE
        min_y = min(p.y() for p in self.points) - self.HANDLE_SIZE
        max_x = max(p.x() for p in self.points) + self.HANDLE_SIZE
        max_y = max(p.y() for p in self.points) + self.HANDLE_SIZE
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def paint(self, painter: QPainter, option, widget):
        base_color = QColor("#00b8ff"); base_color.setAlpha(60)
        sel_color = QColor("#ffcc00"); sel_color.setAlpha(80)
        is_selected = self.isSelected()
        brush = QBrush(sel_color if is_selected else base_color)
        pen = QPen(QColor("#ffcc00" if is_selected else "#00b8ff"), 2)
        painter.setPen(pen); painter.setBrush(brush)
        painter.drawPolygon(QPolygonF(self.points))
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.setBrush(QBrush(QColor("#00ff88")))
        for pt in self.points:
            painter.drawEllipse(pt, self.HANDLE_SIZE, self.HANDLE_SIZE)

    def _find_closest_handle(self, pos: QPointF) -> int:
        for i, pt in enumerate(self.points):
            if (pt - pos).manhattanLength() < self.HANDLE_SIZE * 2:
                return i
        return -1

    def hoverMoveEvent(self, event):
        if self._find_closest_handle(event.pos()) != -1:
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_index = self._find_closest_handle(event.pos())
            if self.drag_index != -1:
                # Тянем за угол: курсор меняется, super() НЕ вызываем (блокируем перемещение всей области)
                self.setCursor(Qt.SizeAllCursor)
            else:
                # Кликнули в тело: делегируем Qt стандартное перемещение объекта
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_index != -1:
            # Перемещение вершины с привязкой к пикселям
            pos = event.pos()
            self.points[self.drag_index] = QPointF(round(pos.x()), round(pos.y()))
            self.prepareGeometryChange()
        else:
            # Перемещение всей области средствами Qt
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.drag_index = -1
        self.unsetCursor()
        self.update()
        # Экспорт ВСЕГДА в глобальных координатах сцены (учитывает и pos() объекта, и локальные вершины)
        coords = [(round(self.mapToScene(p).x(), 2), round(self.mapToScene(p).y(), 2)) for p in self.points]
        self.region_modified.emit(coords, self.region_class)
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        change_cls = menu.addAction(f"🏷 Класс: {self.region_class}")
        delete_act = menu.addAction("🗑 Удалить область")
        action = menu.exec(event.screenPos())

        if action == change_cls:
            new_cls, ok = QInputDialog.getText(None, "Изменить класс", "Новый класс:")
            if ok and new_cls:
                self.region_class = new_cls
                coords = [(round(self.mapToScene(p).x(), 2), round(self.mapToScene(p).y(), 2)) for p in self.points]
                self.region_modified.emit(coords, self.region_class)
        elif action == delete_act:
            self.region_deleted.emit(self)
            if self.scene():
                self.scene().removeItem(self)


class PlanEditorView(QGraphicsView):
    """Виджет для отображения и редактирования 2D-плана"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #2a2a2a; border: 1px solid #444; border-radius: 4px;")

        self.add_mode = False
        self.rect_start = None
        self.temp_rect = None
        self.regions: list[EditableRegionItem] = []

    def set_background(self, image_path: str):
        # Очистка уже выполнена в reset_editor(), загружаем только новый фон
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.scene.addPixmap(pixmap)
            self.setSceneRect(pixmap.rect())
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        win = self.window()
        if win and hasattr(win, 'statusBar'):
            win.statusBar().showMessage("📐 Новый план загружен", 2000)

    def reset_editor(self):
        """Полный сброс данных, сцены и сигналов"""
        # 🔑 Безопасное отключение сигналов (защита от крэша при повторном вызове)
        for item in self.regions:
            try:
                item.region_deleted.disconnect(self._on_region_deleted)
            except:
                pass
            try:
                item.region_modified.disconnect(self._on_region_updated)
            except:
                pass

        self.regions.clear()  # Очищаем Python-список
        self.scene.clear()  # Очищаем Qt-сцену
        self.add_mode = False  # Выключаем режим рисования
        self.rect_start = None
        self.temp_rect = None

        main_win = self.window()
        if main_win and hasattr(main_win, 'btn_add_region'):
            main_win.btn_add_region.setChecked(False)

    def add_region_from_ai(self, polygon: list, cls: str = "Unknown"):
        if not polygon or len(polygon) < 3: return
        item = EditableRegionItem(polygon, cls)
        item.region_modified.connect(self._on_region_updated)
        item.region_deleted.connect(self._on_region_deleted)
        self.scene.addItem(item)
        self.regions.append(item)

    def _on_region_deleted(self, item: EditableRegionItem):
        """Синхронизирует список данных при удалении области"""
        if item in self.regions:
            self.regions.remove(item)
        win = self.window()
        if win and hasattr(win, 'statusBar'):
            win.statusBar().showMessage("🗑 Область удалена из данных", 2000)

    def set_add_mode(self, enabled: bool):
        self.add_mode = enabled
        self.setDragMode(QGraphicsView.NoDrag if enabled else QGraphicsView.ScrollHandDrag)
        self.setCursor(QCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor))

    def mousePressEvent(self, event):
        if self.add_mode and event.button() == Qt.LeftButton:
            self.rect_start = self.mapToScene(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.add_mode and self.rect_start:
            if not self.temp_rect:
                self.temp_rect = QGraphicsRectItem(self.scene)
                self.temp_rect.setPen(QPen(Qt.DashLine))
            end = self.mapToScene(event.pos())
            self.temp_rect.setRect(QRectF(self.rect_start, end).normalized())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.add_mode and self.rect_start:
            rect = QRectF(self.rect_start, self.mapToScene(event.pos())).normalized()
            if rect.width() > 10 and rect.height() > 10:
                polygon = [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
                self.add_region_from_ai([(p.x(), p.y()) for p in polygon], "NewRegion")
            if self.temp_rect:
                self.scene.removeItem(self.temp_rect)
                self.temp_rect = None
            self.rect_start = None
        super().mouseReleaseEvent(event)

    def _on_region_updated(self, coords, cls):
        win = self.window()
        if win and hasattr(win, 'statusBar'):
            win.statusBar().showMessage(f"📐 Координаты обновлены: {coords[0]}", 2000)

    def get_regions_data(self) -> list[dict]:
        """Экспорт всех областей с координатами, привязанными к фону изображения"""
        return [
            {
                # 🔑 FIX: item.mapToScene() корректно работает с QPointF
                "coords": [(round(item.mapToScene(p).x(), 2), round(item.mapToScene(p).y(), 2)) for p in item.points],
                "class": item.region_class
            }
            for item in self.regions
        ]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("План → 3D Модель")
        self.resize(1400, 900)
        self.current_scene = None
        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ================= ВЕРХНЯЯ ПАНЕЛЬ: ОРГАНЫ УПРАВЛЕНИЯ =================
        ctrl_widget = QWidget()
        ctrl_layout = QHBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(5, 5, 5, 5)

        # 1. Выбор файла
        file_layout = QVBoxLayout()
        btn_open = QPushButton("📂 Выбрать план")
        btn_open.clicked.connect(self._select_plan)
        btn_open.setMinimumHeight(35)
        file_layout.addWidget(btn_open)
        file_layout.addStretch()

        # 2. Параметры
        params_layout = QFormLayout()
        self.input_scale = QDoubleSpinBox()
        self.input_scale.setRange(0.001, 1.0)
        self.input_scale.setValue(0.05)
        self.input_scale.setDecimals(3)
        params_layout.addRow("Масштаб (см/пиксель):", self.input_scale)

        self.input_height = QDoubleSpinBox()
        self.input_height.setRange(0.5, 10.0)
        self.input_height.setValue(2.8)
        self.input_height.setDecimals(1)
        params_layout.addRow("Высота стен (м):", self.input_height)

        self.btn_add_region = QPushButton("➕ Добавить область")
        self.btn_add_region.setCheckable(True)
        self.btn_add_region.clicked.connect(self._toggle_add_mode)
        self.btn_add_region.setMinimumHeight(35)
        ctrl_layout.addWidget(self.btn_add_region)

        # 3. Прогресс и действия
        action_layout = QVBoxLayout()
        self.progress_label = QLabel("Готово к работе")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)

        self.btn_run = QPushButton("▶ Создать модель")
        self.btn_run.clicked.connect(self._start_processing)
        self.btn_run.setMinimumHeight(35)

        self.btn_save = QPushButton("💾 Сохранить OBJ")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._export_obj)
        self.btn_save.setMinimumHeight(35)

        # 🔑 КНОПКА ОЧИСТКИ
        self.btn_clear = QPushButton("🗑 Очистить всё")
        self.btn_clear.clicked.connect(self._clear_all)
        self.btn_clear.setMinimumHeight(35)
        self.btn_clear.setStyleSheet("background-color: #c62828; color: white; font-weight: bold;")

        action_layout.addWidget(self.progress_label)
        action_layout.addWidget(self.progress_bar)
        action_layout.addWidget(self.btn_run)
        action_layout.addWidget(self.btn_save)
        action_layout.addWidget(self.btn_clear)  # ← Добавлено

        # Собираем верхнюю панель
        ctrl_layout.addLayout(file_layout, 1)
        ctrl_layout.addLayout(params_layout, 1)
        ctrl_layout.addLayout(action_layout, 2)
        main_layout.addWidget(ctrl_widget)

        # ================= НИЖНЯЯ ПАНЕЛЬ: ПРЕДПРОСМОТР (2D | 3D) =================
        self.preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.preview_splitter.setChildrenCollapsible(False)

        # Левая часть: 2D изображение
        # self.preview_label = QLabel("Предпросмотр не выбран")
        # self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.preview_label.setStyleSheet("background: #2a2a2a; border: 1px solid #444; border-radius: 4px;")
        # self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.preview_splitter.addWidget(self.preview_label)

        self.editor_2d = PlanEditorView(self)
        self.preview_splitter.addWidget(self.editor_2d)

        # Правая часть: 3D вьюпорт
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_splitter.addWidget(self.plotter)

        # Пропорции при старте (2D : 3D = 1 : 2)
        self.preview_splitter.setSizes([500, 500])
        main_layout.addWidget(self.preview_splitter, stretch=1)

    def _select_plan(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите план", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path: return

        if hasattr(i2w, 'process_floor_plan') and hasattr(i2w.process_floor_plan, 'cache_clear'):
            i2w.process_floor_plan.cache_clear()

        self._plan_path = path
        self.current_scene = None  # Сбрасываем старую 3D-модель

        # 🔑 1. ЖЁСТКИЙ СБРОС РЕДАКТОРА (очищает списки, сцену, отключает сигналы)
        self.editor_2d.reset_editor()

        # 2. Загрузка нового фона
        self.editor_2d.set_background(path)

        # 3. Вызов ИИ для НОВОГО плана
        try:
            _, _, regions_dict = i2w.process_floor_plan(self._plan_path)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка ИИ", f"Не удалось обработать план: {e}")
            return

        # 4. Заполнение ТОЛЬКО новыми областями
        total_regions = 0
        for cls_name, polygons in regions_dict.items():
            if not polygons: continue
            for poly_array in polygons:
                poly_list = poly_array.tolist() if hasattr(poly_array, 'tolist') else list(poly_array)
                self.editor_2d.add_region_from_ai(poly_list, cls=cls_name)
                total_regions += 1

        self.editor_2d.fitInView(self.editor_2d.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.statusBar().showMessage(f"📐 План обновлён. Загружено {total_regions} областей.", 3000)

    def _toggle_add_mode(self):
        self.editor_2d.set_add_mode(self.btn_add_region.isChecked())

    def _clear_all(self):
        """Полная очистка редактора и связанных состояний"""
        if not self.editor_2d.regions:
            self.statusBar().showMessage("ℹ️ Список областей уже пуст", 2000)
            return

        if hasattr(i2w, 'clear_cache'):
            i2w.clear_cache()
        if hasattr(gm1, 'clear_cache'):
            gm1.clear_cache()

        self.editor_2d.reset_editor()  # Вызывает ваш готовый сброс
        self.btn_save.setEnabled(False)
        self.statusBar().showMessage("🗑 Все области удалены, сценарий сброшен", 2000)

    def _rebuild_regions_dict(self) -> dict:
        """Преобразует данные из 2D-редактора обратно в формат {класс: [numpy.array]}"""
        import numpy as np
        rebuilt = {}
        for item in self.editor_2d.get_regions_data():
            cls = item['class']
            if cls not in rebuilt:
                rebuilt[cls] = []
            # gm1 ожидает numpy-массивы, поэтому конвертируем list -> np.array
            rebuilt[cls].append(np.array(item['coords'], dtype=np.float32))
        print(f'len = {len(rebuilt)}')
        return rebuilt

    def _start_processing(self):
        if not hasattr(self, '_plan_path'):
            QMessageBox.warning(self, "Внимание", "Сначала выберите план!")
            return

        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

        # 🔑 Собираем актуальные данные из 2D-редактора
        edited_regions = self._rebuild_regions_dict()
        if edited_regions:
            self.statusBar().showMessage(f"📐 Используем {len(edited_regions)} классов из редактора", 2000)
        else:
            self.statusBar().showMessage("📐 Используются данные нейросети без правок", 2000)

        self.btn_run.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Запуск обработки...")

        self.worker = ProcessingWorker(
            self._plan_path,
            self.input_scale.value(),
            self.input_height.value(),
            edited_regions=edited_regions  # ← Передаём в поток
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
        self.plotter.clear()

        meshes = scene.dump()
        if meshes is None:
            self.statusBar().showMessage("⚠️ Сцена пуста", 3000)
            self.btn_run.setEnabled(True)
            return

        if not isinstance(meshes, list):
            meshes = [meshes]

        for mesh in meshes:
            if mesh is not None:
                self.plotter.add_mesh(mesh, color="lightblue", show_edges=True, smooth_shading=True)

        self.plotter.reset_camera()
        self.progress_label.setText("✅ Модель успешно построена")
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(True)

    def _on_error(self, msg: str):
        self.progress_label.setText(f"❌ Ошибка: {msg}")
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Ошибка обработки", msg)

    def _export_obj(self):
        if not self.current_scene: return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить модель", "", "Wavefront OBJ (*.obj)")
        if path:
            self.current_scene.export(path, file_type="obj")
            QMessageBox.information(self, "Успех", f"Модель сохранена:\n{path}")


if __name__ == "__main__":
    # Опционально: высокая четкость для 4K мониторов
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())