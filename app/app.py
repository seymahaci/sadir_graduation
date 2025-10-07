#!/usr/bin/env python3

import os
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QPushButton, QLabel, QFileDialog, QCheckBox, 
                             QListWidget, QHBoxLayout, QSplitter, QProgressBar,
                             QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class AnalysisThread(QThread):
    update_progress = pyqtSignal(int)
    analysis_complete = pyqtSignal()

    def __init__(self, image_paths, weapon_model, pedo_model, analyze_weapon, analyze_pedo):
        super().__init__()
        self.image_paths = image_paths
        self.weapon_model = weapon_model
        self.pedo_model = pedo_model
        self.analyze_weapon = analyze_weapon
        self.analyze_pedo = analyze_pedo
        self.weapon_results = []
        self.pedo_results = []

    def run(self):
        total_images = len(self.image_paths)
        for i, img_path in enumerate(self.image_paths):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if self.analyze_weapon:
                    weapon_detected = self.detect_weapon(img_rgb)
                    if weapon_detected:
                        self.weapon_results.append(img_path)
                
                if self.analyze_pedo:
                    pedo_detected = self.detect_pedo(img_rgb)
                    if pedo_detected:
                        self.pedo_results.append(img_path)

                progress = int((i + 1) / total_images * 100)
                self.update_progress.emit(progress)

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

        self.analysis_complete.emit()

    def detect_weapon(self, image):
        """Run weapon detection model"""
        results = self.weapon_model(image)
        detections = results.pandas().xyxy[0]
        
        weapon_detections = detections[(detections['name'].isin(['rifle', 'handgun'])) & (detections['confidence'] > 0.4)]
        return len(weapon_detections) > 0

    def detect_pedo(self, image):
        """Pedofili tespiti - sadece child algılanırsa pozitif say"""
        results = self.pedo_model(image)
        detections = results.pandas().xyxy[0]
    
        # Sadece 'child' class'ını ve confidence threshold'unu kontrol et
        child_detections = detections[(detections['name'] == 'Child') & (detections['confidence'] > 0.5)]
        return len(child_detections) > 0

class ImageAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_models()
        
    def initUI(self):
        self.setWindowTitle('SADIR: System for Analyzing Digital Information and Reconnaissance')
        self.setGeometry(100, 100, 1000, 700)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_btn = QPushButton('Dizin Seç')
        self.dir_btn.clicked.connect(self.select_directory)
        self.dir_label = QLabel('Seçili dizin: Hiçbiri')
        dir_layout.addWidget(self.dir_btn)
        dir_layout.addWidget(self.dir_label)
        layout.addLayout(dir_layout)
        
        # Analysis options
        options_layout = QHBoxLayout()
        self.weapon_cb = QCheckBox('Silah Tespiti (rifle, handgun)')
        self.pedo_cb = QCheckBox('Pedofili Tespiti')
        options_layout.addWidget(self.weapon_cb)
        options_layout.addWidget(self.pedo_cb)
        layout.addWidget(QLabel('Analiz Türleri:'))
        layout.addLayout(options_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Run button
        self.run_btn = QPushButton('Analiz Başlat')
        self.run_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_btn)
        
        # Results area
        self.result_splitter = QSplitter(Qt.Vertical)
        
        self.weapon_results = QListWidget()
        self.weapon_results.setWindowTitle('Silah Tespit Edilenler')
        self.weapon_results.itemDoubleClicked.connect(self.show_image)
        
        self.pedo_results = QListWidget()
        self.pedo_results.setWindowTitle('Pedofili Tespit Edilenler')
        self.pedo_results.itemDoubleClicked.connect(self.show_image)
        
        self.result_splitter.addWidget(self.weapon_results)
        self.result_splitter.addWidget(self.pedo_results)
        
        layout.addWidget(self.result_splitter)
        
    def load_models(self):
        """Load YOLOv5 models"""
        try:
            # Load models with torch.hub
            self.weapon_model = torch.hub.load('ultralytics/yolov5', 'custom', path='detect_weapon.pt')
            self.pedo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='detect_phedo.pt')
            
            # Set models to evaluation mode
            self.weapon_model.eval()
            self.pedo_model.eval()
            
            # Use GPU if available
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.weapon_model.to(self.device)
            self.pedo_model.to(self.device)
            
        except Exception as e:
            QMessageBox.critical(self, "Model Yükleme Hatası", 
                               f"Modeller yüklenirken hata oluştu:\n{str(e)}")
            sys.exit(1)
            
    def select_directory(self):
        """Directory selection dialog"""
        dir_path = QFileDialog.getExistingDirectory(self, 'Dizin Seç')
        if dir_path:
            self.dir_label.setText(f'Seçili dizin: {dir_path}')
            self.selected_dir = dir_path
            
    def run_analysis(self):
        """Start analysis process"""
        if not hasattr(self, 'selected_dir'):
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir dizin seçin")
            return
            
        analyze_weapon = self.weapon_cb.isChecked()
        analyze_pedo = self.pedo_cb.isChecked()
        
        if not analyze_weapon and not analyze_pedo:
            QMessageBox.warning(self, "Uyarı", "Lütfen en az bir analiz türü seçin")
            return
            
        # Find all images
        image_paths = []
        for root, _, files in os.walk(self.selected_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            QMessageBox.information(self, "Bilgi", "Seçilen dizinde görüntü dosyası bulunamadı")
            return
            
        # Clear previous results
        self.weapon_results.clear()
        self.pedo_results.clear()
        self.progress_bar.setValue(0)
        
        # Disable UI during analysis
        self.set_ui_enabled(False)
        
        # Create and start analysis thread
        self.analysis_thread = AnalysisThread(
            image_paths, self.weapon_model, self.pedo_model,
            analyze_weapon, analyze_pedo
        )
        self.analysis_thread.update_progress.connect(self.update_progress)
        self.analysis_thread.analysis_complete.connect(self.analysis_finished)
        self.analysis_thread.start()
    
    def set_ui_enabled(self, enabled):
        """Enable/disable UI elements during analysis"""
        self.dir_btn.setEnabled(enabled)
        self.weapon_cb.setEnabled(enabled)
        self.pedo_cb.setEnabled(enabled)
        self.run_btn.setEnabled(enabled)
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def analysis_finished(self):
        """Called when analysis is complete"""
        self.set_ui_enabled(True)
        
        # Add results to lists
        for result in self.analysis_thread.weapon_results:
            self.weapon_results.addItem(result)
        
        for result in self.analysis_thread.pedo_results:
            self.pedo_results.addItem(result)
        
        # Show summary
        total = len(self.analysis_thread.image_paths)
        weapon_count = len(self.analysis_thread.weapon_results)
        pedo_count = len(self.analysis_thread.pedo_results)
        
        QMessageBox.information(
            self, "Analiz Tamamlandı",
            f"Toplam {total} görüntü analiz edildi.\n\n"
            f"Silah tespit edilen görüntüler: {weapon_count}\n"
            f"Pedofili tespit edilen görüntüler: {pedo_count}"
        )
    
    def show_image(self, item):
        """Show selected image with detections"""
        from PIL import Image
        img_path = item.text()
        
        # Determine which model to use for visualization
        if item.listWidget() == self.weapon_results:
            model = self.weapon_model
        else:
            model = self.pedo_model
        
        # Get detections
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        
        # Render detections
        rendered_img = results.render()[0]
        
        # Convert to PIL Image and show
        pil_img = Image.fromarray(rendered_img)
        pil_img.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set style
    app.setStyle('Fusion')
    
    ex = ImageAnalyzerApp()
    ex.show()
    sys.exit(app.exec_())
