import sys
import os
from typing import Optional
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QThread
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QPolygonF, QColor, QCursor

import trimesh
import pyvista as pv
from pyvistaqt import QtInteractor
import cv2

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


import img2wall as i2w
import gen_mod1 as gm1


class ProcessingWorker(QThread):
    progress = Signal(str, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, plan_path: str, scale: float, wall_height: float, axis_tol: int, gap_tol: int, edited_regions: Optional[dict] = None, precomputed_contours: Optional[dict] = None, use_ai_walls: bool = False):
        super().__init__()
        self.plan_path = plan_path
        self.scale = scale
        self.axis_tol = axis_tol
        # self.thick_tol = thick_tol
        self.gap_tol = gap_tol
        self.wall_height = wall_height * 7.14  # коэффициент масштабирования
        self.edited_regions: Optional[dict] = edited_regions
        self.use_ai_walls = use_ai_walls  # 🔧 FIX: принимаем состояние переключателя
        self.precomputed_contours = precomputed_contours

    def run(self):
        try:
            # 1. Анализ плана
            self.progress.emit("1/4: Анализ плана нейросетью...", 20)
            if self.precomputed_contours:
                raw_contours = self.precomputed_contours

            wall_contours_cv, image_size, raw_contours = i2w.process_floor_plan(self.plan_path,20, True, self.axis_tol, 5, self.gap_tol)

            contours = self.edited_regions if self.edited_regions else raw_contours

            # 🔧 FIX: логика выбора контуров на основе переключателя
            if self.use_ai_walls:
                wall_contours = contours.get('Wall', raw_contours.get('Wall', []))
            else:
                # Если пользователь правил области → используем их, иначе → фоллбэк на ИИ
                wall_contours = wall_contours_cv

            # Стены часто нужны отдельно для построения геометрии
            h_wall_contours = contours.get('h-wall', raw_contours.get('h-wall', []))

            # 2. Генерация стен
            self.progress.emit("2/4: Построение 3D-геометрии стен...", 45)
            scene = gm1.build_3d_model(wall_contours, self.scale, self.wall_height)
            if h_wall_contours and self.use_ai_walls:
                scene.add_geometry(gm1.build_3d_model(h_wall_contours, self.scale, self.wall_height))

            # 3. Проёмы и мебель
            self.progress.emit("3/4: Расстановка проёмов и мебели...", 75)
            if 'Door' in contours:
                scene.add_geometry(gm1.build_door(contours['Door'], wall_contours, self.scale, self.wall_height))
            if 'Window' in contours:
                scene.add_geometry(gm1.build_window(contours['Window'], wall_contours, self.scale, self.wall_height))

            scene.add_geometry(gm1.build_obj(contours, wall_contours, self.scale))

            # 4. Финализация
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
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
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

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        base_color = QColor("#00b8ff")
        base_color.setAlpha(60)
        sel_color = QColor("#ffcc00")
        sel_color.setAlpha(80)
        is_selected = self.isSelected()
        brush = QBrush(sel_color if is_selected else base_color)
        pen = QPen(QColor("#ffcc00" if is_selected else "#00b8ff"), 2)
        painter.setPen(pen)
        painter.setBrush(brush)
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
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_index = self._find_closest_handle(event.pos())
            if self.drag_index != -1:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_index != -1:
            pos = event.pos()
            self.points[self.drag_index] = QPointF(round(pos.x()), round(pos.y()))
            self.prepareGeometryChange()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.drag_index = -1
        self.unsetCursor()
        self.update()
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #2a2a2a; border: 1px solid #444; border-radius: 4px;")

        self.add_mode = False
        self.rect_start = None
        self.temp_rect = None
        self.regions: list[EditableRegionItem] = []

    def set_background(self, image_path: str):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.scene.addPixmap(pixmap)
            self.setSceneRect(pixmap.rect())
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        win = self.window()
        if win and hasattr(win, 'statusBar'):
            win.statusBar().showMessage("📐 Новый план загружен", 2000)

    def reset_editor(self):
        for item in self.regions:
            try:
                item.region_deleted.disconnect(self._on_region_deleted)
            except:
                pass
            try:
                item.region_modified.disconnect(self._on_region_updated)
            except:
                pass

        self.regions.clear()
        self.scene.clear()
        self.add_mode = False
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
        if item in self.regions:
            self.regions.remove(item)
        win = self.window()
        if win and hasattr(win, 'statusBar'):
            win.statusBar().showMessage("🗑 Область удалена из данных", 2000)

    def set_add_mode(self, enabled: bool):
        self.add_mode = enabled
        self.setDragMode(QGraphicsView.DragMode.NoDrag if enabled else QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor))

    def mousePressEvent(self, event):
        if self.add_mode and event.button() == Qt.MouseButton.LeftButton:
            self.rect_start = self.mapToScene(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.add_mode and self.rect_start:
            if not self.temp_rect:
                self.temp_rect = QGraphicsRectItem()
                self.scene.addItem(self.temp_rect)
                self.temp_rect.setPen(QPen(Qt.PenStyle.DashLine))
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
        return [
            {
                "coords": [(round(item.mapToScene(p).x(), 2), round(item.mapToScene(p).y(), 2)) for p in item.points],
                "class": item.region_class
            }
            for item in self.regions
        ]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор 3D Планировок")
        self.resize(1500, 900)
        self.current_scene = None

        # Проверка конфигов при старте
        if not gm1.obj_mesh:
            QMessageBox.warning(
                self, "⚠️ Конфигурация не загружена",
                "Файл config/mesh_paths.json не найден.\n3D-мебли не будет."
            )

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # ================= ВЕРХНЯЯ ПАНЕЛЬ УПРАВЛЕНИЯ =================
        controls_container = QFrame()
        controls_container.setFrameShape(QFrame.Shape.StyledPanel)
        controls_container.setStyleSheet("QFrame { background: #f0f0f0; border-radius: 4px; padding: 5px; }")

        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.setSpacing(5)

        # --- 1. Строка инструментов (Файл + Режимы) ---
        tools_layout = QHBoxLayout()

        btn_open = QPushButton("📂 Загрузить план")
        btn_open.clicked.connect(self._select_plan)
        btn_open.setStyleSheet("font-weight: bold;")
        tools_layout.addWidget(btn_open, 0)  # Не растягивать

        tools_layout.addSpacing(10)

        self.btn_add_region = QPushButton("➕ Рисовать")
        self.btn_add_region.setCheckable(True)
        self.btn_add_region.clicked.connect(self._toggle_add_mode)
        tools_layout.addWidget(self.btn_add_region)

        self.btn_type_wall = QCheckBox("🤖 Стены от ИИ (сырые)")
        self.btn_type_wall.setChecked(True)
        self.btn_type_wall.toggled.connect(self._toggle_wall_mode)
        tools_layout.addWidget(self.btn_type_wall)

        tools_layout.addStretch()
        controls_layout.addLayout(tools_layout)

        # --- 2. Строка параметров и действий ---
        params_actions_layout = QHBoxLayout()

        # Группа параметров
        params_group = QGroupBox("Параметры")
        params_group_layout = QHBoxLayout(params_group)
        params_group_layout.setContentsMargins(5, 5, 5, 5)

        self.input_scale = QDoubleSpinBox()
        self.input_scale.setRange(0.001, 10.0)
        self.input_scale.setValue(0.05)
        self.input_scale.setDecimals(3)
        params_group_layout.addWidget(QLabel("Масштаб:"))
        params_group_layout.addWidget(self.input_scale)

        params_group_layout.addSpacing(10)

        self.input_target_scale = QDoubleSpinBox()
        self.input_target_scale.setRange(0.01, 20.0)
        self.input_target_scale.setValue(0.5)
        self.input_target_scale.setDecimals(3)
        self.input_target_scale.setToolTip("Целевой масштаб изображения (см на 1 пиксель)")

        params_group_layout.addWidget(QLabel("Эталон (см/px): "))
        params_group_layout.addWidget(self.input_target_scale)

        # Авто-синхронизация: масштаб для 3D (м/px) всегда = эталон (см/px) / 100
        self.input_target_scale.valueChanged.connect(lambda v: self.input_scale.setValue(v / 100.0))

        params_group_layout.addSpacing(10)

        self.input_height = QDoubleSpinBox()
        self.input_height.setRange(0.5, 10.0)
        self.input_height.setValue(2.8)
        self.input_height.setDecimals(2)
        params_group_layout.addWidget(QLabel("Высота:"))
        params_group_layout.addWidget(self.input_height)

        self.input_axis_tol = QSpinBox()
        self.input_axis_tol.setRange(0, 30.0)
        self.input_axis_tol.setValue(5)
        params_group_layout.addWidget(QLabel("Допуск смещения оси:"))
        params_group_layout.addWidget(self.input_axis_tol)

        params_group_layout.addSpacing(10)

        # self.input_thick_tol = QSpinBox()
        # self.input_thick_tol.setRange(0, 100.0)
        # self.input_thick_tol.setValue(5)
        # params_group_layout.addWidget(QLabel("Допуск разницы толщины:"))
        # params_group_layout.addWidget(self.input_thick_tol)
        # params_group_layout.addSpacing(10)

        self.input_gap_tol = QSpinBox()
        self.input_gap_tol.setRange(0, 500.0)
        self.input_gap_tol.setValue(250)
        params_group_layout.addWidget(QLabel("Максимальный разрыв для объединения:"))
        params_group_layout.addWidget(self.input_gap_tol)

        params_group_layout.addSpacing(10)
        params_actions_layout.addWidget(params_group, 0)  # Фиксированный размер

        params_actions_layout.addStretch()

        # Группа кнопок действий
        actions_group = QHBoxLayout()

        self.btn_run = QPushButton("▶ СОЗДАТЬ")
        self.btn_run.clicked.connect(self._start_processing)
        self.btn_run.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 5px 15px; border-radius: 3px;")
        actions_group.addWidget(self.btn_run)

        self.chk_optimize = QCheckBox("🗜 Оптимизировать")
        self.chk_optimize.setChecked(True)
        self.chk_optimize.setToolTip("Сжать геометрию и текстуры для веба (меньше размер, чуть дольше)")
        actions_group.addWidget(self.chk_optimize)

        self.btn_export_gltf = QPushButton("🌐 GLTF")
        self.btn_export_gltf.setEnabled(False)
        self.btn_export_gltf.clicked.connect(self._export_gltf)
        self.btn_export_gltf.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 5px 12px; border-radius: 3px;")
        actions_group.addWidget(self.btn_export_gltf)

        self.btn_save = QPushButton("💾 OBJ")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._export_obj)
        actions_group.addWidget(self.btn_save)

        self.btn_clear = QPushButton("🗑 Очистить")
        self.btn_clear.clicked.connect(self._clear_all)
        self.btn_clear.setStyleSheet("background-color: #ef5350; color: white; font-weight: bold;")
        actions_group.addWidget(self.btn_clear)

        params_actions_layout.addLayout(actions_group)
        controls_layout.addLayout(params_actions_layout)

        # --- 3. Статусная панель (Прогресс) ---
        status_panel = QWidget()
        status_panel.setStyleSheet("background: #e0e0e0; border-radius: 3px;")
        status_layout = QHBoxLayout(status_panel)
        status_layout.setContentsMargins(8, 2, 8, 2)

        self.progress_label = QLabel("🟢 Готов к работе")
        self.progress_label.setStyleSheet("font-weight: bold; color: #333;")
        status_layout.addWidget(self.progress_label, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v%")
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #ccc; border-radius: 3px; text-align: center; }")
        status_layout.addWidget(self.progress_bar, 1)  # Растягивается

        controls_layout.addWidget(status_panel)

        main_layout.addWidget(controls_container)

        # ================= НИЖНЯЯ ПАНЕЛЬ (ПРЕДПРОСМОТР) =================
        self.preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.preview_splitter.setChildrenCollapsible(False)
        self.preview_splitter.setStyleSheet("QSplitter::handle { background: #888; width: 3px; }")

        # 2D Редактор
        self.editor_2d = PlanEditorView(self)
        self.preview_splitter.addWidget(self.editor_2d)

        # 3D Вьюпорт
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")
        self.preview_splitter.addWidget(self.plotter)

        # Пропорции при старте
        self.preview_splitter.setSizes([600, 600])
        main_layout.addWidget(self.preview_splitter, stretch=1)

    def _toggle_wall_mode(self, is_ai):
        status = " Режим: Сырые контуры ИИ" if is_ai else "✏️ Режим: Отредактированные контуры"
        self.progress_label.setText(status)

    def _select_plan(self):
        #TODO: проверить паралельность обработки, либо забить и указывать на долгую работу в первый запуск
        path, _ = QFileDialog.getOpenFileName(self, "Выберите план", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path: return

        self._plan_path = path
        self.current_scene = None
        self.editor_2d.reset_editor()
        self.editor_2d.set_background(path)

        self.progress_label.setText("⏳ Масштабирование плана... ")
        QApplication.processEvents()

        try:
            target_cm_px = self.input_target_scale.value()
            resize_res = i2w.calculate_and_resize_image(self._plan_path, target_cm_per_pixel=target_cm_px)
            scaled_img = resize_res['scaled_image']

            # Сохраняем отмасштабированное изображение
            self._scaled_plan_path = "temp_scaled_plan.png"
            cv2.imwrite(self._scaled_plan_path, scaled_img)

            self.progress_label.setText("⏳ Обработка плана нейросетью... ")
            QApplication.processEvents()

            # Запускаем ИИ уже на отмасштабированном изображении
            _, _, regions_dict = i2w.process_floor_plan(self._scaled_plan_path)
            self._raw_contours = regions_dict  # Сохраняем для воркера
        except Exception as e:
            QMessageBox.warning(self, "Ошибка ИИ", f"Не удалось обработать план: {e}")
            self.progress_label.setText("❌ Ошибка обработки")
            return

        # Отображаем в 2D-редакторе уже масштабированный план
        self.editor_2d.set_background(self._scaled_plan_path)

        total_regions = 0
        for cls_name, polygons in regions_dict.items():
            if not polygons: continue
            for poly_array in polygons:
                poly_list = poly_array.tolist() if hasattr(poly_array, 'tolist') else list(poly_array)
                self.editor_2d.add_region_from_ai(poly_list, cls=cls_name)
                total_regions += 1

        self.editor_2d.fitInView(self.editor_2d.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.progress_label.setText(f"✅ План загружен. Найдено {total_regions} объектов. ")

    def _toggle_add_mode(self):
        self.editor_2d.set_add_mode(self.btn_add_region.isChecked())

    def _clear_all(self):
        if not self.editor_2d.regions and not self.current_scene:
            self.progress_label.setText("️ Нечего очищать")
            return

        if hasattr(i2w, 'clear_cache'): i2w.clear_cache()
        if hasattr(gm1, 'clear_cache'): gm1.clear_cache()

        self.editor_2d.reset_editor()
        self.current_scene = None
        self.plotter.clear()
        self.btn_save.setEnabled(False)
        self.progress_label.setText("🧹 Очищено")

    def _rebuild_regions_dict(self) -> dict:
        rebuilt = {}
        for item in self.editor_2d.get_regions_data():
            cls = item['class']
            if cls not in rebuilt: rebuilt[cls] = []
            rebuilt[cls].append(np.array(item['coords'], dtype=np.float32))
        return rebuilt

    def _start_processing(self):
        if not hasattr(self, '_plan_path'):
            QMessageBox.warning(self, "Внимание", "Сначала выберите план!")
            return

        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

        edited_regions = self._rebuild_regions_dict()

        self.btn_run.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("🏗 Генерация модели...")

        self.worker = ProcessingWorker(
            getattr(self, '_scaled_plan_path', self._plan_path),
            self.input_scale.value(),
            self.input_height.value(),
            self.input_axis_tol.value(),
            # self.input_height.value(),
            self.input_gap_tol.value(),
            edited_regions=edited_regions,
            use_ai_walls=self.btn_type_wall.isChecked(),
            precomputed_contours=getattr(self, '_raw_contours', None)
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
            self.progress_label.setText("⚠️ Сцена пуста")
            self.btn_run.setEnabled(True)
            return

        if not isinstance(meshes, list): meshes = [meshes]

        for mesh in meshes:
            if mesh is not None:
                self.plotter.add_mesh(mesh, color="lightblue", show_edges=True, smooth_shading=True)

        self.plotter.reset_camera()
        self.progress_label.setText("✅ Модель построена")
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_export_gltf.setEnabled(True)

    def _on_error(self, msg: str):
        self.progress_label.setText(f"❌ Ошибка: {msg}")
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Ошибка обработки", msg)

    def _export_obj(self):
        if not self.current_scene: return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить модель", "", "Wavefront OBJ (*.obj)")
        if path:
            try:
                self.current_scene.export(path, file_type="obj")
                QMessageBox.information(self, "Успех", f"Модель сохранена:\n{path}")
                self.progress_label.setText(f"💾 Сохранено в {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка экспорта", str(e))

    def _export_gltf(self):
        """Экспорт в GLTF/GLB с выбором формата и оптимизацией."""
        if not self.current_scene:
            QMessageBox.warning(self, "Внимание", "Сначала создайте модель!")
            return

        # 1. Диалог с явным выбором формата
        filters = "GLB (всё в одном файле) (*.glb);;GLTF (JSON + внешние файлы) (*.gltf)"
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт 3D-модели", "", filters)
        if not path:
            return

        # 2. Определяем формат по расширению (QFileDialog подставляет его автоматически)
        file_ext = Path(path).suffix.lower()
        file_type = file_ext.lstrip('.')  # 'glb' или 'gltf'

        try:
            self.progress_label.setText("📦 Подготовка и экспорт...")
            QApplication.processEvents()

            # 3. Работаем с копией сцены, чтобы оптимизация не изменила модель в 3D-вьюпорте
            scene_to_export = self.current_scene.copy()

            # 4. Оптимизация (если включена чекбоксом)
            if self.chk_optimize.isChecked():
                for name, mesh in list(scene_to_export.geometry.items()):
                    if mesh is None or mesh.is_empty:
                        continue

                    # Упрощение геометрии для больших мешей
                    if len(mesh.faces) > 10000 and hasattr(mesh, 'simplify_quadratic_decimation'):
                        try:
                            target = max(500, len(mesh.faces) // 3)
                            scene_to_export.geometry[name] = mesh.simplify_quadratic_decimation(target)
                        except Exception as e:
                            logger.warning(f"⚠️ Пропущено упрощение {name}: {e}")

                    # Фикс нормалей (важно для корректного освещения)
                    if not mesh.is_watertight:
                        try:
                            mesh.fix_normals()
                        except:
                            pass

            # 5. Экспорт
            scene_to_export.export(path, file_type=file_type)

            # 6. Обратная связь
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            status = f"✅ {file_type.upper()} сохранён: {size_mb:.1f} МБ"
            if self.chk_optimize.isChecked():
                status += " (оптимизировано)"

            self.progress_label.setText(status)
            QMessageBox.information(
                self, "Успех",
                f"Файл: {Path(path).name}\nФормат: {file_type.upper()}\nРазмер: {size_mb:.1f} МБ"
            )

        except Exception as e:
            logger.error(f"❌ Ошибка экспорта: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{str(e)}")


if __name__ == "__main__":
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())