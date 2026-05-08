import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
import subprocess
import time
import random
import shutil
import json
import csv
import re
import gc
import warnings
from pathlib import Path

BG_COLOR = "#1e1e2e"
FG_COLOR = "#3f4f86"
ACCENT_COLOR = "#89b4fa"
BUTTON_BG = "#313244"
BUTTON_FG = "#cdd6f4"
ENTRY_BG = "#313244"
ENTRY_FG = "#cdd6f4"
SUCCESS_COLOR = "#a6e3a1"
WARNING_COLOR = "#f9e2af"
ERROR_COLOR = "#f38ba8"

class YoloTrainerApp:
    def __init__(self, root):
        self.root = root

        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        suffix = " [CUDA]" if cuda_available else " [CPU]"
        self.root.title("AI Trash - YOLO Model Manager" + suffix)

        default_device = "0" if cuda_available else "cpu"

        self.current_process = None
        self.root.geometry("1000x800")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(900, 700)
        self.selected_model_path = tk.StringVar()
        self.dataset_path = tk.StringVar()
        self.save_folder = tk.StringVar()
        self.source_path = tk.StringVar()
        self.device_var = tk.StringVar(value=default_device)
        self.train_device = tk.StringVar(value=default_device)
        self.current_mode = "new"
        self.adv_inf_open = False
        self.adv_train_open = False
        self.build_main_screen()

    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def reset_state(self):
        if self.current_process is not None:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=2)
            except Exception:
                try:
                    self.current_process.kill()
                except Exception:
                    pass
            self.current_process = None

        self.selected_model_path.set("")
        self.dataset_path.set("")
        self.save_folder.set("")
        self.source_path.set("")

        try:
            import torch
            default_device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            default_device = "cpu"
        self.device_var.set(default_device)
        self.train_device.set(default_device)

        self.current_mode = "new"
        self.adv_inf_open = False
        self.adv_train_open = False

        for attr in [
            "adv_entries", "optimizer", "cache_var", "amp_var",
            "exist_ok_var", "export_mode_var", "half_var", "open_results_var",
            "max_det", "classes_filter", "inf_imgsz", "arch_var",
            "epochs", "imgsz", "batch", "workers", "project_name", "run_name",
            "train_split", "val_split", "test_split", "test_dataset_path", "test_imgsz", "test_device"
        ]:
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass

        gc.collect()
        self.clear_frame()

    def build_main_screen(self):
        self.reset_state()

        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        suffix = " [CUDA]" if cuda_available else " [CPU]"
        self.root.title("AI Trash - YOLO Model Manager" + suffix)

        header = tk.Frame(self.root, bg=BG_COLOR)
        header.pack(pady=50)
        tk.Label(header, text="🗑️ AI Trash", font=("Segoe UI", 36, "bold"),
                bg=BG_COLOR, fg=ACCENT_COLOR).pack()
        tk.Label(header, text="YOLO Model Training & Inference", font=("Segoe UI", 16),
                bg=BG_COLOR, fg=FG_COLOR).pack(pady=15)
        btn_frame = tk.Frame(self.root, bg=BG_COLOR)
        btn_frame.pack(expand=True)
        tk.Button(btn_frame, text="📂 Select Existing Model",
                 font=("Segoe UI", 18, "bold"),
                 bg=BUTTON_BG, fg=BUTTON_FG,
                 activebackground=ACCENT_COLOR, activeforeground=BG_COLOR,
                 width=28, height=3, cursor="hand2",
                 command=self.select_existing_model_screen).pack(pady=20)
        tk.Button(btn_frame, text="🆕 Train New Model",
                 font=("Segoe UI", 18, "bold"),
                 bg=BUTTON_BG, fg=BUTTON_FG,
                 activebackground=SUCCESS_COLOR, activeforeground=BG_COLOR,
                 width=28, height=3, cursor="hand2",
                 command=self.train_new_model_screen).pack(pady=20)
        footer = tk.Frame(self.root, bg=BG_COLOR)
        footer.pack(side="bottom", pady=30)
        tk.Label(footer, text="Make your own AI — No coding required — Charlz Galdo",
                font=("Segoe UI", 11, "italic"),
                bg=BG_COLOR, fg="#6c7086").pack()

    def _clean_ansi(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text = ansi_escape.sub('', text)
        text = text.replace('\r', '')
        text = text.replace('[K', '')
        return text.strip()

    def _save_results_summary(self, results, save_dir, mode):
        import json
        import csv
        from pathlib import Path

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        new_entries = []
        for i, r in enumerate(results):
            item = {"image": getattr(r, 'path', f'image_{i}')}
            if mode in ("top1", "all"):
                item["top1"] = {
                    "class": r.names[r.probs.top1],
                    "confidence": round(float(r.probs.top1conf) * 100, 2)
                }
            if mode in ("top5", "all"):
                top5_indices = r.probs.top5
                top5_confs = r.probs.top5conf
                item["top5"] = [
                    {
                        "rank": j + 1,
                        "class": r.names[idx],
                        "confidence": round(float(conf) * 100, 2)
                    }
                    for j, (idx, conf) in enumerate(zip(top5_indices, top5_confs))
                ]
            new_entries.append(item)

        json_path = os.path.join(save_dir, "predictions_summary.json")
        existing_json = []
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing_json = json.load(f)
                if not isinstance(existing_json, list):
                    existing_json = []
            except Exception:
                existing_json = []

        json_batch = {
            "timestamp": timestamp,
            "predictions": new_entries
        }
        existing_json.append(json_batch)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_json, f, indent=2)

        csv_path = os.path.join(save_dir, "predictions_summary.csv")
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if file_exists:
                writer.writerow([])
            writer.writerow([timestamp])

            if mode == "top1":
                if not file_exists:
                    writer.writerow(["image", "predicted_class", "confidence_%"])
                for item in new_entries:
                    writer.writerow([item["image"], item["top1"]["class"], item["top1"]["confidence"]])
            elif mode == "top5":
                if not file_exists:
                    writer.writerow(["image", "rank", "class", "confidence_%"])
                for item in new_entries:
                    for entry in item["top5"]:
                        writer.writerow([item["image"], entry["rank"], entry["class"], entry["confidence"]])
            else:
                if not file_exists:
                    writer.writerow(["image", "top1_class", "top1_confidence_%", "top5_json"])
                for item in new_entries:
                    writer.writerow([
                        item["image"],
                        item["top1"]["class"],
                        item["top1"]["confidence"],
                        json.dumps(item["top5"])
                    ])

        return json_path, csv_path

    def select_existing_model_screen(self):
        self.clear_frame()
        self.build_header("Select Existing Model")
        canvas = tk.Canvas(self.root, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=BG_COLOR)
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw", width=980)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(10,0))
        scrollbar.pack(side="right", fill="y")
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        content.pack(pady=40)
        tk.Label(content, text="Step 1: Select your trained model file (.pt)",
                font=("Segoe UI", 14, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w", pady=(0,10))
        model_frame = tk.Frame(content, bg=BG_COLOR)
        model_frame.pack(fill="x", pady=5)
        tk.Entry(model_frame, textvariable=self.selected_model_path,
                font=("Segoe UI", 12), bg=ENTRY_BG, fg=ENTRY_FG,
                insertbackground=FG_COLOR, width=55).pack(side="left", padx=(0,10))
        tk.Button(model_frame, text="Browse", font=("Segoe UI", 11),
                 bg=BUTTON_BG, fg=BUTTON_FG, cursor="hand2",
                 command=self.browse_model).pack(side="left")
        btn_frame = tk.Frame(content, bg=BG_COLOR)
        btn_frame.pack(pady=50)
        tk.Button(btn_frame, text="← Back", font=("Segoe UI", 13),
                 bg=BUTTON_BG, fg=BUTTON_FG, width=14, cursor="hand2",
                 command=self.build_main_screen).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Use Model →", font=("Segoe UI", 13, "bold"),
                 bg=ACCENT_COLOR, fg=BG_COLOR, width=16, cursor="hand2",
                 command=self.use_model_screen).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Improve Model →", font=("Segoe UI", 13, "bold"),
                 bg=SUCCESS_COLOR, fg=BG_COLOR, width=16, cursor="hand2",
                 command=self.improve_model_screen).pack(side="left", padx=10)

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.selected_model_path.set(path)

    def use_model_screen(self):
        if not self.selected_model_path.get():
            messagebox.showerror("Error", "Please select a model file first!")
            return
        self.clear_frame()
        self.build_header("Use Model — Run Predictions")

        self.inf_main_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.inf_main_frame.pack(fill="both", expand=True)

        self.inf_settings_frame = tk.Frame(self.inf_main_frame, bg=BG_COLOR)
        self.inf_settings_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(self.inf_settings_frame, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.inf_settings_frame, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=BG_COLOR)
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw", width=980)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(10,0))
        scrollbar.pack(side="right", fill="y")
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        content.pack(pady=20, padx=50)

        tk.Label(content, text="Source (Image or Folder):",
                font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w", pady=(10,5))
        src_frame = tk.Frame(content, bg=BG_COLOR)
        src_frame.pack(fill="x", pady=5)
        tk.Entry(src_frame, textvariable=self.source_path,
                font=("Segoe UI", 11), bg=ENTRY_BG, fg=ENTRY_FG,
                insertbackground=FG_COLOR, width=55).pack(side="left", padx=(0,10))
        tk.Button(src_frame, text="Browse File", font=("Segoe UI", 10),
                 bg=BUTTON_BG, fg=BUTTON_FG, cursor="hand2",
                 command=lambda: self.browse_source("file")).pack(side="left", padx=2)
        tk.Button(src_frame, text="Browse Folder", font=("Segoe UI", 10),
                 bg=BUTTON_BG, fg=BUTTON_FG, cursor="hand2",
                 command=lambda: self.browse_source("folder")).pack(side="left", padx=2)

        tk.Label(content, text="Save Predictions To:",
                font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w", pady=(20,5))
        save_frame = tk.Frame(content, bg=BG_COLOR)
        save_frame.pack(fill="x", pady=5)
        tk.Entry(save_frame, textvariable=self.save_folder,
                font=("Segoe UI", 11), bg=ENTRY_BG, fg=ENTRY_FG,
                insertbackground=FG_COLOR, width=55).pack(side="left", padx=(0,10))
        tk.Button(save_frame, text="Browse", font=("Segoe UI", 10),
                 bg=BUTTON_BG, fg=BUTTON_FG, cursor="hand2",
                 command=self.browse_save_folder).pack(side="left")

        settings_frame = tk.LabelFrame(content, text=" Essential Settings ",
                                      font=("Segoe UI", 13, "bold"),
                                      bg=BG_COLOR, fg=ACCENT_COLOR,
                                      bd=2, relief="groove")
        settings_frame.pack(fill="x", pady=25, padx=5)
        tk.Label(settings_frame, text="Device:",
                font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR).grid(row=0, column=0, sticky="w", padx=15, pady=10)

        inf_dev_combo = ttk.Combobox(settings_frame, textvariable=self.device_var,
                    values=["0", "0,1","cpu"],
                    state="readonly", width=18)
        inf_dev_combo.grid(row=0, column=1, sticky="w", padx=15, pady=10)
        self._style_combo(inf_dev_combo)
        tk.Label(settings_frame, text="Classification models predict the single most likely class per image.",
                font=("Segoe UI", 9), bg=BG_COLOR, fg="#6c7086").grid(row=1, column=0, columnspan=2, sticky="w", padx=15, pady=10)

        self.adv_inf_frame = tk.LabelFrame(content, text=" Advanced Settings ",
                                           font=("Segoe UI", 13, "bold"),
                                           bg=BG_COLOR, fg=WARNING_COLOR,
                                           bd=2, relief="groove")
        self.adv_inf_btn = tk.Button(content, text="▼ Show Advanced Settings", font=("Segoe UI", 11),
                 bg=BUTTON_BG, fg=WARNING_COLOR, cursor="hand2",
                 command=self.toggle_advanced_inference)
        self.adv_inf_btn.pack(pady=10)

        self.btn_frame = tk.Frame(content, bg=BG_COLOR)
        self.btn_frame.pack(pady=20)
        tk.Button(self.btn_frame, text="← Back", font=("Segoe UI", 12),
                 bg=BUTTON_BG, fg=BUTTON_FG, width=14, cursor="hand2",
                 command=self.select_existing_model_screen).pack(side="left", padx=10)
        tk.Button(self.btn_frame, text="▶ Run Prediction", font=("Segoe UI", 15, "bold"),
                 bg=SUCCESS_COLOR, fg=BG_COLOR, width=20, cursor="hand2",
                 command=self.show_console_view).pack(side="left", padx=10)
        tk.Button(self.btn_frame, text="🧪 Test Model (Accuracy)", font=("Segoe UI", 13, "bold"),
                 bg=WARNING_COLOR, fg=BG_COLOR, width=20, cursor="hand2",
                 command=self.test_model_screen).pack(side="left", padx=10)

        self.inf_console_frame = tk.Frame(self.inf_main_frame, bg=BG_COLOR)

        console_header = tk.Frame(self.inf_console_frame, bg=BG_COLOR)
        console_header.pack(fill="x", pady=20)
        tk.Label(console_header, text="Running Inference...", font=("Segoe UI", 22, "bold"),
                bg=BG_COLOR, fg=SUCCESS_COLOR).pack()

        self.build_log_area(self.inf_console_frame)
        self.log_text.configure(height=28)

        console_btn_frame = tk.Frame(self.inf_console_frame, bg=BG_COLOR)
        console_btn_frame.pack(pady=20)
        tk.Button(console_btn_frame, text="← Back to Settings", font=("Segoe UI", 12),
                 bg=BUTTON_BG, fg=BUTTON_FG, width=18, cursor="hand2",
                 command=self.show_settings_view).pack(side="left", padx=10)

    def show_console_view(self):
        model = self.selected_model_path.get()
        source = self.source_path.get()
        save_dir = self.save_folder.get()
        if not model or not os.path.exists(model):
            messagebox.showerror("Error", "Please select a valid model file!")
            return
        if not source:
            messagebox.showerror("Error", "Please select a source image or folder!")
            return
        if not save_dir:
            messagebox.showerror("Error", "Please select a save folder!")
            return

        self.inf_settings_frame.pack_forget()
        self.inf_console_frame.pack(fill="both", expand=True)

        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, "end")
        self.log_text.configure(state="disabled")

        self._do_run_inference()

    def show_settings_view(self):
        if self.current_process is not None:
            try:
                self.current_process.terminate()
                self.log("\n⏹ Inference stopped by user.")
            except Exception:
                pass
            self.current_process = None
        self.inf_console_frame.pack_forget()
        self.inf_settings_frame.pack(fill="both", expand=True)

    def toggle_advanced_inference(self):
        if not self.adv_inf_open:
            self.adv_inf_frame.pack(fill="x", pady=15, padx=5, before=self.adv_inf_btn)
            self.build_advanced_inference()
            self.adv_inf_open = True
            self.adv_inf_btn.config(text="▲ Hide Advanced Settings")
        else:
            self.adv_inf_frame.pack_forget()
            for w in self.adv_inf_frame.winfo_children():
                w.destroy()
            self.adv_inf_open = False
            self.adv_inf_btn.config(text="▼ Show Advanced Settings")

    def build_advanced_inference(self):
        tk.Label(self.adv_inf_frame, text="Export Format:",
                font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR).grid(row=5, column=0, sticky="w", padx=15, pady=8)
        self.export_mode_var = tk.StringVar(value="top1")
        export_combo = ttk.Combobox(self.adv_inf_frame, textvariable=self.export_mode_var,
                    values=["top1", "top5", "all"],
                    state="readonly", width=18)
        export_combo.grid(row=5, column=1, sticky="w", padx=15, pady=8)
        self._style_combo(export_combo)
        tk.Label(self.adv_inf_frame, text="Max Detections:",
                font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR).grid(row=0, column=0, sticky="w", padx=15, pady=8)
        self.max_det = tk.Entry(self.adv_inf_frame, font=("Segoe UI", 11),
                               bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=18)
        self.max_det.insert(0, "300")
        self.max_det.grid(row=0, column=1, sticky="w", padx=15, pady=8)
        tk.Label(self.adv_inf_frame, text="Filter Classes (e.g. 0,1,2):",
                font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR).grid(row=1, column=0, sticky="w", padx=15, pady=8)
        self.classes_filter = tk.Entry(self.adv_inf_frame, font=("Segoe UI", 11),
                                      bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=18)
        self.classes_filter.grid(row=1, column=1, sticky="w", padx=15, pady=8)
        tk.Label(self.adv_inf_frame, text="Image Size (imgsz):",
                font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR).grid(row=2, column=0, sticky="w", padx=15, pady=8)
        self.inf_imgsz = tk.Entry(self.adv_inf_frame, font=("Segoe UI", 11),
                                 bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=18)
        self.inf_imgsz.insert(0, "224")
        self.inf_imgsz.grid(row=2, column=1, sticky="w", padx=15, pady=8)
        self.half_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.adv_inf_frame, text="Half Precision (FP16)",
                      variable=self.half_var,
                      font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR,
                      selectcolor=BG_COLOR).grid(row=3, column=0, columnspan=2, sticky="w", padx=15, pady=8)
        self.open_results_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.adv_inf_frame, text="Open result images when done",
                      variable=self.open_results_var,
                      font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR,
                      selectcolor=BG_COLOR).grid(row=4, column=0, columnspan=2, sticky="w", padx=15, pady=8)

    def browse_source(self, mode):
        if mode == "file":
            path = filedialog.askopenfilename(
                title="Select Image/Video",
                filetypes=[("Images/Videos", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov"), ("All files", "*.*")]
            )
        else:
            path = filedialog.askdirectory(title="Select Folder of Images")
        if path:
            self.source_path.set(path)

    def improve_model_screen(self):
        if not self.selected_model_path.get():
            messagebox.showerror("Error", "Please select a model file first!")
            return
        self.train_screen(mode="improve")

    def train_new_model_screen(self):
        self.selected_model_path.set("")
        self.train_screen(mode="new")

    def train_screen(self, mode="new"):
        self.clear_frame()
        title = "Improve Model (Fine-tune)" if mode == "improve" else "Train New Model"
        self.build_header(title)

        self.train_main_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.train_main_frame.pack(fill="both", expand=True)

        self.train_settings_frame = tk.Frame(self.train_main_frame, bg=BG_COLOR)
        self.train_settings_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(self.train_settings_frame, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.train_settings_frame, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=BG_COLOR)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=980)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(10,0))
        scrollbar.pack(side="right", fill="y")
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        content = scroll_frame

        if mode == "improve":
            tk.Label(content, text="Base Model: " + self.selected_model_path.get(),
                    font=("Segoe UI", 11, "italic"), bg=BG_COLOR, fg=SUCCESS_COLOR,
                    wraplength=900).pack(anchor="w", pady=(10,5), padx=25)
        else:
            tk.Label(content, text="Select Base Architecture:",
                    font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w", pady=(10,5), padx=25)
            arch_frame = tk.Frame(content, bg=BG_COLOR)
            arch_frame.pack(fill="x", pady=5, padx=25)
            self.arch_var = tk.StringVar(value="yolov8s-cls.pt")
            architectures = ["yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt",
                           "yolov9t-cls.pt", "yolov9s-cls.pt", "yolov9m-cls.pt", "yolov9c-cls.pt", "yolov9e-cls.pt",
                           "yolov10n-cls.pt", "yolov10s-cls.pt", "yolov10m-cls.pt", "yolov10b-cls.pt", "yolov10l-cls.pt", "yolov10x-cls.pt"]
            arch_combo = ttk.Combobox(arch_frame, textvariable=self.arch_var,
                        values=architectures, state="readonly", width=22)
            arch_combo.pack(side="left")
            self._style_combo(arch_combo)

        tk.Label(content, text="Dataset Folder (folder with class subfolders like cardboard/, glass/, etc.):",
                font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w", pady=(20,5), padx=25)
        ds_frame = tk.Frame(content, bg=BG_COLOR)
        ds_frame.pack(fill="x", pady=5, padx=25)
        tk.Entry(ds_frame, textvariable=self.dataset_path,
                font=("Segoe UI", 11), bg=ENTRY_BG, fg=ENTRY_FG,
                insertbackground=FG_COLOR, width=55).pack(side="left", padx=(0,10))
        tk.Button(ds_frame, text="Browse", font=("Segoe UI", 10),
                 bg=BUTTON_BG, fg=BUTTON_FG, cursor="hand2",
                 command=self.browse_dataset_folder).pack(side="left")

        tk.Label(content, text="Save Model To Folder:",
                font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w", pady=(20,5), padx=25)
        save_frame = tk.Frame(content, bg=BG_COLOR)
        save_frame.pack(fill="x", pady=5, padx=25)
        tk.Entry(save_frame, textvariable=self.save_folder,
                font=("Segoe UI", 11), bg=ENTRY_BG, fg=ENTRY_FG,
                insertbackground=FG_COLOR, width=55).pack(side="left", padx=(0,10))
        tk.Button(save_frame, text="Browse", font=("Segoe UI", 10),
                 bg=BUTTON_BG, fg=BUTTON_FG, cursor="hand2",
                 command=self.browse_save_folder).pack(side="left")

        ess_frame = tk.LabelFrame(content, text=" Essential Training Settings ",
                                 font=("Segoe UI", 13, "bold"),
                                 bg=BG_COLOR, fg=ACCENT_COLOR,
                                 bd=2, relief="groove")
        ess_frame.pack(fill="x", pady=25, padx=25)

        tk.Label(ess_frame, text="Epochs:", font=("Segoe UI", 11),
                bg=BG_COLOR, fg=FG_COLOR).grid(row=0, column=0, sticky="w", padx=15, pady=10)
        self.epochs = tk.Entry(ess_frame, font=("Segoe UI", 11),
                              bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=18)
        self.epochs.insert(0, "100")
        self.epochs.grid(row=0, column=1, sticky="w", padx=15, pady=10)

        tk.Label(ess_frame, text="Image Size (imgsz):", font=("Segoe UI", 11),
                bg=BG_COLOR, fg=FG_COLOR).grid(row=1, column=0, sticky="w", padx=15, pady=10)
        self.imgsz = tk.Entry(ess_frame, font=("Segoe UI", 11),
                             bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=18)
        self.imgsz.insert(0, "224")
        self.imgsz.grid(row=1, column=1, sticky="w", padx=15, pady=10)

        tk.Label(ess_frame, text="Batch Size:", font=("Segoe UI", 11),
                bg=BG_COLOR, fg=FG_COLOR).grid(row=2, column=0, sticky="w", padx=15, pady=10)
        self.batch = tk.Entry(ess_frame, font=("Segoe UI", 11),
                             bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=18)
        self.batch.insert(0, "16")
        self.batch.grid(row=2, column=1, sticky="w", padx=15, pady=10)

        tk.Label(ess_frame, text="Device:", font=("Segoe UI", 11),
                bg=BG_COLOR, fg=FG_COLOR).grid(row=3, column=0, sticky="w", padx=15, pady=10)
        train_dev_combo = ttk.Combobox(ess_frame, textvariable=self.train_device,
                    values=["0", "0,1","cpu"], state="readonly", width=18)
        train_dev_combo.grid(row=3, column=1, sticky="w", padx=15, pady=10)
        self._style_combo(train_dev_combo)

        tk.Label(ess_frame, text="Workers:", font=("Segoe UI", 11),
                bg=BG_COLOR, fg=FG_COLOR).grid(row=4, column=0, sticky="w", padx=15, pady=10)
        self.workers = tk.Entry(ess_frame, font=("Segoe UI", 11),
                               bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=18)
        self.workers.insert(0, "8")
        self.workers.grid(row=4, column=1, sticky="w", padx=15, pady=10)

        tk.Label(ess_frame, text="Project Name:", font=("Segoe UI", 11),
                bg=BG_COLOR, fg=FG_COLOR).grid(row=0, column=2, sticky="w", padx=40, pady=10)
        self.project_name = tk.Entry(ess_frame, font=("Segoe UI", 11),
                                    bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=22)
        self.project_name.insert(0, "AI_Trash_Training")
        self.project_name.grid(row=0, column=3, sticky="w", padx=15, pady=10)

        tk.Label(ess_frame, text="Run Name:", font=("Segoe UI", 11),
                bg=BG_COLOR, fg=FG_COLOR).grid(row=1, column=2, sticky="w", padx=40, pady=10)
        self.run_name = tk.Entry(ess_frame, font=("Segoe UI", 11),
                                bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=22)
        self.run_name.insert(0, "exp")
        self.run_name.grid(row=1, column=3, sticky="w", padx=15, pady=10)

        # Split ratios
        tk.Label(ess_frame, text="Train Split %:", font=("Segoe UI", 11),
                bg=BG_COLOR, fg=FG_COLOR).grid(row=5, column=0, sticky="w", padx=15, pady=10)
        self.train_split = tk.Entry(ess_frame, font=("Segoe UI", 11),
                                   bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=18)
        self.train_split.insert(0, "70")
        self.train_split.grid(row=5, column=1, sticky="w", padx=15, pady=10)

        tk.Label(ess_frame, text="Val Split %:", font=("Segoe UI", 11),
                bg=BG_COLOR, fg=FG_COLOR).grid(row=6, column=0, sticky="w", padx=15, pady=10)
        self.val_split = tk.Entry(ess_frame, font=("Segoe UI", 11),
                                 bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=18)
        self.val_split.insert(0, "20")
        self.val_split.grid(row=6, column=1, sticky="w", padx=15, pady=10)

        tk.Label(ess_frame, text="Test Split %:", font=("Segoe UI", 11),
                bg=BG_COLOR, fg=FG_COLOR).grid(row=7, column=0, sticky="w", padx=15, pady=10)
        self.test_split = tk.Entry(ess_frame, font=("Segoe UI", 11),
                                  bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=18)
        self.test_split.insert(0, "10")
        self.test_split.grid(row=7, column=1, sticky="w", padx=15, pady=10)

        self.adv_train_frame = tk.LabelFrame(content, text=" Advanced Training Settings ",
                                            font=("Segoe UI", 13, "bold"),
                                            bg=BG_COLOR, fg=WARNING_COLOR,
                                            bd=2, relief="groove")
        self.adv_train_btn = tk.Button(content, text="▼ Show Advanced Settings", font=("Segoe UI", 11),
                 bg=BUTTON_BG, fg=WARNING_COLOR, cursor="hand2",
                 command=self.toggle_advanced_training)
        self.adv_train_btn.pack(pady=10)

        self.train_btn_frame = tk.Frame(content, bg=BG_COLOR)
        self.train_btn_frame.pack(pady=25)
        tk.Button(self.train_btn_frame, text="← Back", font=("Segoe UI", 12),
                 bg=BUTTON_BG, fg=BUTTON_FG, width=14, cursor="hand2",
                 command=self.build_main_screen).pack(side="left", padx=10)
        run_text = "▶ Fine-tune Model" if mode == "improve" else "▶ Train New Model"
        run_color = SUCCESS_COLOR if mode == "improve" else ACCENT_COLOR
        tk.Button(self.train_btn_frame, text=run_text, font=("Segoe UI", 15, "bold"),
                 bg=run_color, fg=BG_COLOR, width=22, cursor="hand2",
                 command=self.show_train_console_view).pack(side="left", padx=10)

        self.train_console_frame = tk.Frame(self.train_main_frame, bg=BG_COLOR)

        console_header = tk.Frame(self.train_console_frame, bg=BG_COLOR)
        console_header.pack(fill="x", pady=20)
        tk.Label(console_header, text="Training in Progress...", font=("Segoe UI", 22, "bold"),
                bg=BG_COLOR, fg=SUCCESS_COLOR).pack()

        self.build_log_area(self.train_console_frame)
        self.log_text.configure(height=28)

        console_btn_frame = tk.Frame(self.train_console_frame, bg=BG_COLOR)
        console_btn_frame.pack(pady=20)

        tk.Button(console_btn_frame, text="⏹ STOP (Finish Current Epoch)", font=("Segoe UI", 12, "bold"),
                 bg=ERROR_COLOR, fg=BG_COLOR, width=28, cursor="hand2",
                 command=self.request_training_stop).pack(side="left", padx=10)

        tk.Button(console_btn_frame, text="← Back to Settings", font=("Segoe UI", 12),
                 bg=BUTTON_BG, fg=BUTTON_FG, width=18, cursor="hand2",
                 command=self.show_train_settings_view).pack(side="left", padx=10)

        self.current_mode = mode

    def show_train_console_view(self):
        dataset = self.dataset_path.get()
        save_dir = self.save_folder.get()
        if not dataset or not os.path.exists(dataset):
            messagebox.showerror("Error", "Please select a valid dataset folder!")
            return
        try:
            subdirs = [d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d))]
            if not subdirs:
                messagebox.showerror("Error", "Dataset folder must contain subfolders (one per class, e.g. cardboard/, glass/, etc.)")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Could not read dataset folder: {e}")
            return
        if not save_dir:
            messagebox.showerror("Error", "Please select a save folder!")
            return

        self.train_settings_frame.pack_forget()
        self.train_console_frame.pack(fill="both", expand=True)

        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, "end")
        self.log_text.configure(state="disabled")

        self._do_run_training()

    def show_train_settings_view(self):
        self._request_and_wait_stop()
        self.train_console_frame.pack_forget()
        self.train_settings_frame.pack(fill="both", expand=True)

    def request_training_stop(self):
        self._request_and_wait_stop()

    def _request_and_wait_stop(self):
        if hasattr(self, '_stop_flag_path'):
            try:
                with open(self._stop_flag_path, 'w') as f:
                    f.write('stop')
                self.log("\n⏹ Stop requested — finishing current epoch...")
            except Exception as e:
                self.log(f"\n⚠️ Could not write stop flag: {e}")

        if self.current_process is not None and self.current_process.poll() is None:
            try:
                self.current_process.wait(timeout=5)
                self.log("\n✅ Training stopped gracefully.")
            except Exception:
                self.log("\n⚠️ Forcing termination...")
                try:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=2)
                except Exception:
                    try:
                        self.current_process.kill()
                        self.current_process.wait(timeout=2)
                    except Exception:
                        pass
                self.log("\n⏹ Training force-stopped.")
            finally:
                self._auto_export_log("training_stopped")
                self.current_process = None
        else:
            self.current_process = None

    def _auto_export_log(self, prefix):
        try:
            save_dir = self.save_folder.get()
            if not save_dir:
                save_dir = os.path.expanduser("~")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_log_{timestamp}.txt"
            filepath = os.path.join(save_dir, filename)
            self.log_text.configure(state="normal")
            content = self.log_text.get(1.0, "end")
            self.log_text.configure(state="disabled")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            self.log(f"\n💾 Log auto-saved: {filepath}")
        except Exception as e:
            self.log(f"\n⚠️ Could not auto-save log: {e}")

    def _do_run_training(self):
        mode = self.current_mode
        dataset = self.dataset_path.get()
        save_dir = self.save_folder.get()

        if not dataset or not os.path.exists(dataset):
            messagebox.showerror("Error", "Please select a valid dataset folder!")
            self.show_train_settings_view()
            return

        if mode == "improve":
            model_path = self.selected_model_path.get()
            if not model_path:
                messagebox.showerror("Error", "No base model selected for fine-tuning!")
                self.show_train_settings_view()
                return
        else:
            model_path = self.arch_var.get()

        # Auto-split dataset
        random.seed(42)
        train_ratio = int(self.train_split.get() or 70) / 100
        val_ratio = int(self.val_split.get() or 20) / 100
        test_ratio = int(self.test_split.get() or 10) / 100

        split_base = Path(save_dir) / "dataset_split"
        split_base.mkdir(parents=True, exist_ok=True)

        for split in ["train", "val", "test"]:
            split_path = split_base / split
            if split_path.exists():
                shutil.rmtree(split_path)

        class_names = []
        for class_dir in sorted(Path(dataset).iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            class_names.append(class_name)
            images = [f for f in class_dir.iterdir() 
                      if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif')]
            images.sort()
            random.shuffle(images)
            total = len(images)
            if total == 0:
                continue
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)
            splits = {
                "train": images[:train_end],
                "val": images[train_end:val_end],
                "test": images[val_end:]
            }
            for split_name, split_images in splits.items():
                split_class_dir = split_base / split_name / class_name
                split_class_dir.mkdir(parents=True, exist_ok=True)
                for img in split_images:
                    shutil.copy2(img, split_class_dir / img.name)

        self.log(f"\n📊 Dataset split complete:")
        for split in ["train", "val", "test"]:
            count = sum(1 for _ in (split_base / split).rglob('*') if _.is_file())
            self.log(f"   {split}: {count} images")

        data_path = str(split_base)

        kwargs = {
            "data": data_path,
            "epochs": int(self.epochs.get() or 100),
            "imgsz": int(self.imgsz.get() or 224),
            "batch": int(self.batch.get() or 16),
            "device": self.train_device.get(),
            "workers": int(self.workers.get() or 8),
            "project": os.path.join(save_dir, self.project_name.get() or "AI_Trash_Classification"),
            "name": self.run_name.get() or "exp",
            "exist_ok": False,
        }
        if self.adv_train_open and hasattr(self, 'adv_entries'):
            for key, entry in self.adv_entries.items():
                val = entry.get()
                if val:
                    try:
                        if '.' in val:
                            kwargs[key] = float(val)
                        else:
                            kwargs[key] = int(val)
                    except ValueError:
                        kwargs[key] = val
            if hasattr(self, 'optimizer'):
                kwargs["optimizer"] = self.optimizer.get()
            if hasattr(self, 'cache_var') and self.cache_var.get():
                kwargs["cache"] = True
            if hasattr(self, 'amp_var'):
                kwargs["amp"] = self.amp_var.get()
            if hasattr(self, 'exist_ok_var'):
                kwargs["exist_ok"] = self.exist_ok_var.get()

        kwargs_str = ",\n    ".join([k + "=" + repr(v) for k, v in kwargs.items()])

        script_lines = [
            'import sys, os, time, warnings, json, csv, shutil',
            'sys.path.insert(0, ".")',
            'from pathlib import Path',
            'from ultralytics import YOLO',
            'import matplotlib',
            'matplotlib.use("Agg")',
            'import matplotlib.pyplot as plt',
            'warnings.filterwarnings("ignore")',
            '',
            'STOP_FLAG = Path(r"' + os.path.join(save_dir, '.stop_training_flag') + '")',
            '',
            'def print_epoch_table(trainer):',
            '    metrics = trainer.metrics',
            '    epoch = trainer.epoch + 1',
            '    total = trainer.epochs',
            '    sep = "=" * 70',
            '    print()',
            '    print(sep)',
            '    print("  EPOCH " + str(epoch) + "/" + str(total) + " RESULTS")',
            '    print("-" * 70)',
            '    for k, v in metrics.items():',
            '        if v is not None:',
            '            print("  " + str(k).ljust(25) + ": " + str(round(float(v), 4)))',
            '    print(sep)',
            '    print()',
            '',
            'def on_train_epoch_end(trainer):',
            '    print_epoch_table(trainer)',
            '    if STOP_FLAG.exists():',
            '        print("[STOP] Graceful stop requested.")',
            '        trainer.stop = True',
            '',
            'def finalize_all_artifacts(trainer, results, stopped_early=False):',
            '    """Generate every artifact YOLO normally creates on full completion."""',
            '    try:',
            '        import pandas as pd',
            '        import numpy as np',
            '        from ultralytics.utils.plotting import plot_results, plot_images',
            '        from ultralytics.utils.metrics import ConfusionMatrix',
            '        from ultralytics.utils.torch_utils import strip_optimizer',
            '        ',
            '        save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(trainer.save_dir)',
            '        weights_dir = save_dir / "weights"',
            '        weights_dir.mkdir(parents=True, exist_ok=True)',
            '        ',
            '        print("=" * 60)',
            '        print("Finalizing all training artifacts...")',
            '        ',
            '        # 1. Save best.pt and last.pt weights',
            '        if hasattr(trainer, "best") and trainer.best is not None:',
            '            best_src = Path(trainer.best)',
            '            if best_src.exists():',
            '                best_dst = weights_dir / "best.pt"',
            '                shutil.copy2(best_src, best_dst)',
            '                print("  ✓ weights/best.pt saved")',
            '        ',
            '        if hasattr(trainer, "last") and trainer.last is not None:',
            '            last_src = Path(trainer.last)',
            '            if last_src.exists():',
            '                last_dst = weights_dir / "last.pt"',
            '                shutil.copy2(last_src, last_dst)',
            '                print("  ✓ weights/last.pt saved")',
            '        ',
            '        # 2. Save args.yaml',
            '        if hasattr(trainer, "args") and trainer.args is not None:',
            '            import yaml',
            '            args_path = save_dir / "args.yaml"',
            '            args_dict = vars(trainer.args) if hasattr(trainer.args, "__dict__") else dict(trainer.args)',
            '            # Filter non-serializable items',
            '            clean_args = {}',
            '            for k, v in args_dict.items():',
            '                try:',
            '                    json.dumps({k: v})',
            '                    clean_args[k] = v',
            '                except (TypeError, ValueError):',
            '                    clean_args[k] = str(v)',
            '            with open(args_path, "w") as f:',
            '                yaml.dump(clean_args, f, default_flow_style=False)',
            '            print("  ✓ args.yaml saved")',
            '        ',
            '        # 3. Build and save confusion matrices',
            '        if hasattr(trainer, "validator") and trainer.validator is not None:',
            '            val = trainer.validator',
            '            names = trainer.data.get("names", {}) if hasattr(trainer, "data") else {}',
            '            nc = len(names)',
            '            ',
            '            if hasattr(val, "confusion_matrix") and val.confusion_matrix is not None:',
            '                cm = val.confusion_matrix',
            '                # Normalized confusion matrix',
            '                cm_path = save_dir / "confusion_matrix_normalized.png"',
            '                cm.plot(normalize=True, save_dir=save_dir, names=names)',
            '                if (save_dir / "confusion_matrix.png").exists():',
            '                    shutil.move(save_dir / "confusion_matrix.png", cm_path)',
            '                print("  ✓ confusion_matrix_normalized.png saved")',
            '                ',
            '                # Non-normalized confusion matrix',
            '                cm_path_raw = save_dir / "confusion_matrix.png"',
            '                cm.plot(normalize=False, save_dir=save_dir, names=names)',
            '                print("  ✓ confusion_matrix.png saved")',
            '        ',
            '        # 4. Save results.csv',
            '        csv_path = save_dir / "results.csv"',
            '        if hasattr(trainer, "csv_path") and trainer.csv_path and Path(trainer.csv_path).exists():',
            '            shutil.copy2(trainer.csv_path, csv_path)',
            '            print("  ✓ results.csv saved")',
            '        else:',
            '            # Build from metrics history',
            '            if hasattr(trainer, "metrics") and trainer.metrics:',
            '                df = pd.DataFrame([trainer.metrics])',
            '                df.to_csv(csv_path, index=False)',
            '                print("  ✓ results.csv saved (from metrics)")',
            '        ',
            '        # 5. Generate results.png (training curves)',
            '        try:',
            '            results_png = save_dir / "results.png"',
            '            if csv_path.exists():',
            '                df = pd.read_csv(csv_path)',
            '                fig, axes = plt.subplots(2, 5, figsize=(16, 8), tight_layout=True)',
            '                fig.suptitle("Training Results", fontsize=16)',
            '                axes = axes.flatten()',
            '                columns = [c for c in df.columns if c != "epoch"]',
            '                for i, col in enumerate(columns[:10]):',
            '                    if i < len(axes):',
            '                        axes[i].plot(df["epoch"] if "epoch" in df.columns else range(len(df)), df[col])',
            '                        axes[i].set_title(col)',
            '                        axes[i].set_xlabel("epoch")',
            '                for j in range(len(columns), len(axes)):',
            '                    axes[j].axis("off")',
            '                fig.savefig(results_png, dpi=150)',
            '                plt.close(fig)',
            '                print("  ✓ results.png saved")',
            '        except Exception as e:',
            '            print("  ⚠ Could not generate results.png:", str(e))',
            '        ',
            '        # 6. Save train_batch images (sample training batches)',
            '        try:',
            '            if hasattr(trainer, "train_loader") and trainer.train_loader is not None:',
            '                batch_dir = save_dir / "train_batch"',
            '                batch_dir.mkdir(exist_ok=True)',
            '                for i, batch in enumerate(trainer.train_loader):',
            '                    if i >= 3:  # Save first 3 batches max',
            '                        break',
            '                    # Try to extract and save batch visualization',
            '                    try:',
            '                        imgs = batch["img"] if isinstance(batch, dict) else batch[0]',
            '                        fig, ax = plt.subplots(1, 1, figsize=(12, 12))',
            '                        grid = imgs[:16] if len(imgs) > 16 else imgs',
            '                        grid = grid.cpu().numpy() if hasattr(grid, "cpu") else grid',
            '                        # Normalize for display',
            '                        grid = np.transpose(grid, (0, 2, 3, 1)) if grid.ndim == 4 else grid',
            '                        grid = np.clip(grid, 0, 1)',
            '                        n = int(np.ceil(np.sqrt(len(grid))))',
            '                        mosaic = np.zeros((n * grid.shape[1], n * grid.shape[2], grid.shape[3]))',
            '                        for idx, img in enumerate(grid):',
            '                            row, col = divmod(idx, n)',
            '                            mosaic[row*img.shape[0]:(row+1)*img.shape[0], col*img.shape[1]:(col+1)*img.shape[1]] = img',
            '                        ax.imshow(mosaic)',
            '                        ax.axis("off")',
            '                        ax.set_title(f"Train Batch {i}")',
            '                        fig.savefig(batch_dir / f"train_batch{i}.jpg", dpi=100, bbox_inches="tight")',
            '                        plt.close(fig)',
            '                    except Exception:',
            '                        pass',
            '                if list(batch_dir.iterdir()):',
            '                    print("  ✓ train_batch/ images saved")',
            '        except Exception as e:',
            '            print("  ⚠ Could not save train batches:", str(e))',
            '        ',
            '        # 7. Save val_batch images (sample validation batches)',
            '        try:',
            '            if hasattr(trainer, "validator") and hasattr(trainer.validator, "dataloader"):',
            '                val_dir = save_dir / "val_batch"',
            '                val_dir.mkdir(exist_ok=True)',
            '                val_loader = trainer.validator.dataloader',
            '                for i, batch in enumerate(val_loader):',
            '                    if i >= 3:',
            '                        break',
            '                    try:',
            '                        imgs = batch["img"] if isinstance(batch, dict) else batch[0]',
            '                        fig, ax = plt.subplots(1, 1, figsize=(12, 12))',
            '                        grid = imgs[:16] if len(imgs) > 16 else imgs',
            '                        grid = grid.cpu().numpy() if hasattr(grid, "cpu") else grid',
            '                        grid = np.transpose(grid, (0, 2, 3, 1)) if grid.ndim == 4 else grid',
            '                        grid = np.clip(grid, 0, 1)',
            '                        n = int(np.ceil(np.sqrt(len(grid))))',
            '                        mosaic = np.zeros((n * grid.shape[1], n * grid.shape[2], grid.shape[3]))',
            '                        for idx, img in enumerate(grid):',
            '                            row, col = divmod(idx, n)',
            '                            mosaic[row*img.shape[0]:(row+1)*img.shape[0], col*img.shape[1]:(col+1)*img.shape[1]] = img',
            '                        ax.imshow(mosaic)',
            '                        ax.axis("off")',
            '                        ax.set_title(f"Val Batch {i}")',
            '                        fig.savefig(val_dir / f"val_batch{i}_labels.jpg", dpi=100, bbox_inches="tight")',
            '                        plt.close(fig)',
            '                    except Exception:',
            '                        pass',
            '                if list(val_dir.iterdir()):',
            '                    print("  ✓ val_batch/ images saved")',
            '        except Exception as e:',
            '            print("  ⚠ Could not save val batches:", str(e))',
            '        ',
            '        # 8. Strip optimizer from best.pt for deployment',
            '        try:',
            '            best_pt = weights_dir / "best.pt"',
            '            if best_pt.exists():',
            '                strip_optimizer(best_pt)',
            '                print("  ✓ Optimizer stripped from best.pt")',
            '        except Exception:',
            '            pass',
            '        ',
            '        print("All artifacts finalized in: " + str(save_dir))',
            '        print("  - weights/best.pt")',
            '        print("  - weights/last.pt")',
            '        print("  - args.yaml")',
            '        print("  - confusion_matrix.png")',
            '        print("  - confusion_matrix_normalized.png")',
            '        print("  - results.csv")',
            '        print("  - results.png")',
            '        print("  - train_batch/")',
            '        print("  - val_batch/")',
            '        ',
            '    except Exception as e:',
            '        print("Warning: Could not finalize all artifacts: " + str(e))',
            '        import traceback',
            '        print(traceback.format_exc())',
            '',
            'print("Loading model...")',
            'model = YOLO(r"' + model_path + '")',
            'print(r"Model loaded: ' + model_path + '")',
            'print(r"Dataset path: ' + data_path + '")',
            'print("=" * 60)',
            '',
            'model.add_callback("on_train_epoch_end", on_train_epoch_end)',
            '',
            'try:',
            '    results = model.train(',
            '        ' + kwargs_str,
            '    )',
            '    print("=" * 60)',
            '    print("Training complete!")',
            '    finalize_all_artifacts(model.trainer, results, stopped_early=False)',
            'except Exception as e:',
            '    print("Training interrupted or stopped early: " + str(e))',
            '    if hasattr(model, "trainer") and model.trainer is not None:',
            '        try:',
            '            class FakeResults:',
            '                def __init__(self, trainer):',
            '                    self.save_dir = trainer.save_dir',
            '                    self.best = getattr(trainer, "best", None)',
            '            finalize_all_artifacts(model.trainer, FakeResults(model.trainer), stopped_early=True)',
            '        except Exception as e2:',
            '            print("Could not finalize after interrupt: " + str(e2))',
            '    import traceback',
            '    print(traceback.format_exc())',
        ]
        script = "\n".join(script_lines)

        self._stop_flag_path = str(Path(save_dir) / '.stop_training_flag')
        if os.path.exists(self._stop_flag_path):
            os.remove(self._stop_flag_path)

        cmd = [sys.executable, "-c", script]
        self.log("=" * 60)
        self.log("Starting " + ("fine-tuning" if mode == "improve" else "training") + "...")
        self.log("Model: " + model_path)
        self.log("Dataset: " + data_path)
        self.log("Split: " + str(int(train_ratio*100)) + "% train / " + str(int(val_ratio*100)) + "% val / " + str(int(test_ratio*100)) + "% test")
        self.log("Save to: " + save_dir)
        self.log("-" * 60)
        thread = threading.Thread(target=self._run_training_thread, args=(cmd,))
        thread.daemon = True
        thread.start()

    def _run_training_thread(self, cmd):
        try:
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace'
            )
            last_progress_line = None
            for line in iter(self.current_process.stdout.readline, ''):
                if not line:
                    continue
                line = self._clean_ansi(line)
                if not line:
                    continue
                if '%' in line and (line.count('─') >= 2 or 'it/s' in line or 's/it' in line):
                    if last_progress_line is not None:
                        self.log_replace(line)
                    else:
                        self.log(line)
                    last_progress_line = line
                else:
                    if last_progress_line is not None:
                        self.log("")
                        last_progress_line = None
                    self.log(line)
            self.current_process.stdout.close()
            self.current_process.wait()
            if self.current_process.returncode == 0:
                self.log("\n✅ Process completed successfully!")
                self._auto_export_log("training")
            elif self.current_process.returncode == -15 or self.current_process.returncode == 1:
                self._auto_export_log("training_stopped")
            else:
                self.log("\n❌ Process exited with code " + str(self.current_process.returncode))
                self._auto_export_log("training_failed")
        except Exception as e:
            self.log("\n❌ Error: " + str(e))
            self._auto_export_log("training_error")
        finally:
            self.current_process = None

    def _run_inference_thread(self, cmd, save_dir, export_mode="top1"):
        try:
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace'
            )
            last_progress_line = None
            for line in iter(self.current_process.stdout.readline, ''):
                if not line:
                    continue
                line = self._clean_ansi(line)
                if not line:
                    continue
                if '%' in line and (line.count('─') >= 2 or 'it/s' in line or 's/it' in line):
                    if last_progress_line is not None:
                        self.log_replace(line)
                    else:
                        self.log(line)
                    last_progress_line = line
                else:
                    if last_progress_line is not None:
                        self.log("")
                        last_progress_line = None
                    self.log(line)
            self.current_process.stdout.close()
            self.current_process.wait()
            if self.current_process.returncode == 0:
                self.log("\n✅ Process completed successfully!")
                self._auto_export_log("inference")

                try:
                    from ultralytics import YOLO
                    import shutil
                    model = YOLO(self.selected_model_path.get())
                    source = self.source_path.get()
                    kwargs = {
                        "source": source,
                        "device": self.device_var.get(),
                        "save": True,
                        "project": save_dir,
                        "name": "predictions",
                        "exist_ok": True,
                    }
                    if self.adv_inf_open:
                        if hasattr(self, 'inf_imgsz') and self.inf_imgsz.get():
                            kwargs["imgsz"] = int(self.inf_imgsz.get())
                        if hasattr(self, 'half_var') and self.half_var.get():
                            kwargs["half"] = True
                    results = model.predict(**kwargs)

                    pred_dir = os.path.join(save_dir, "predictions")
                    class_counters = {}
                    for i, r in enumerate(results):
                        cls_name = r.names[r.probs.top1]
                        src_path = getattr(r, 'path', None)
                        if src_path and os.path.exists(src_path):
                            ext = os.path.splitext(src_path)[1]
                        else:
                            ext = ".jpg"
                        class_counters[cls_name] = class_counters.get(cls_name, 0) + 1
                        count = class_counters[cls_name]
                        new_name = f"{cls_name} {count}{ext}"
                        orig_name = os.path.basename(src_path) if src_path else f"image_{i}"
                        old_path = os.path.join(pred_dir, orig_name)
                        if not os.path.exists(old_path):
                            for f in os.listdir(pred_dir):
                                if f.startswith(os.path.splitext(orig_name)[0]):
                                    old_path = os.path.join(pred_dir, f)
                                    break
                        new_path = os.path.join(pred_dir, new_name)
                        if os.path.exists(old_path) and not os.path.exists(new_path):
                            shutil.move(old_path, new_path)
                            self.log(f"  📝 Renamed: {os.path.basename(old_path)} → {new_name}")

                    json_path, csv_path = self._save_results_summary(results, pred_dir, export_mode)
                    self.log(f"\n📄 Summary saved:")
                    self.log(f"   JSON: {json_path}")
                    self.log(f"   CSV:  {csv_path}")
                except Exception as e:
                    self.log(f"\n⚠️ Could not generate summary files: {e}")
                    import traceback
                    self.log(traceback.format_exc())

                if hasattr(self, 'open_results_var') and self.open_results_var.get():
                    self._open_results_folder(save_dir)
            elif self.current_process.returncode == -15 or self.current_process.returncode == 1:
                self._auto_export_log("inference_stopped")
            else:
                self.log("\n❌ Process exited with code " + str(self.current_process.returncode))
                self._auto_export_log("inference_failed")
        except Exception as e:
            self.log("\n❌ Error: " + str(e))
            self._auto_export_log("inference_error")
        finally:
            self.current_process = None

    def test_model_screen(self):
        if not self.selected_model_path.get():
            messagebox.showerror("Error", "Please select a model file first!")
            return
        self.clear_frame()
        self.build_header("Test Model — Evaluate Accuracy")

        canvas = tk.Canvas(self.root, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=BG_COLOR)
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw", width=980)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(10,0))
        scrollbar.pack(side="right", fill="y")
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        content.pack(pady=20, padx=50)

        tk.Label(content, text="Test Dataset Folder (same structure as training: class subfolders):",
                font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w", pady=(10,5))
        test_frame = tk.Frame(content, bg=BG_COLOR)
        test_frame.pack(fill="x", pady=5)
        self.test_dataset_path = tk.StringVar()
        tk.Entry(test_frame, textvariable=self.test_dataset_path,
                font=("Segoe UI", 11), bg=ENTRY_BG, fg=ENTRY_FG,
                insertbackground=FG_COLOR, width=55).pack(side="left", padx=(0,10))
        tk.Button(test_frame, text="Browse", font=("Segoe UI", 10),
                 bg=BUTTON_BG, fg=BUTTON_FG, cursor="hand2",
                 command=lambda: self.test_dataset_path.set(
                     filedialog.askdirectory(title="Select Test Dataset Folder") or "")).pack(side="left")

        # Confusion matrix save location
        tk.Label(content, text="Save Confusion Matrix & Results To:",
                font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w", pady=(20,5))
        cm_frame = tk.Frame(content, bg=BG_COLOR)
        cm_frame.pack(fill="x", pady=5)
        self.cm_save_path = tk.StringVar()
        tk.Entry(cm_frame, textvariable=self.cm_save_path,
                font=("Segoe UI", 11), bg=ENTRY_BG, fg=ENTRY_FG,
                insertbackground=FG_COLOR, width=55).pack(side="left", padx=(0,10))
        tk.Button(cm_frame, text="Browse", font=("Segoe UI", 10),
                 bg=BUTTON_BG, fg=BUTTON_FG, cursor="hand2",
                 command=lambda: self.cm_save_path.set(
                     filedialog.askdirectory(title="Select Folder to Save Confusion Matrix & Results") or "")).pack(side="left")

        tk.Label(content, text="Image Size (imgsz):",
                font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w", pady=(20,5))
        self.test_imgsz = tk.Entry(content, font=("Segoe UI", 11),
                                  bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=18)
        self.test_imgsz.insert(0, "224")
        self.test_imgsz.pack(anchor="w", padx=5)

        tk.Label(content, text="Device:",
                font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w", pady=(20,5))
        self.test_device = tk.StringVar(value=self.device_var.get())
        test_dev_combo = ttk.Combobox(content, textvariable=self.test_device,
                    values=["0", "0,1","cpu"], state="readonly", width=18)
        test_dev_combo.pack(anchor="w", padx=5)
        self._style_combo(test_dev_combo)

        btn_frame = tk.Frame(content, bg=BG_COLOR)
        btn_frame.pack(pady=30)
        tk.Button(btn_frame, text="← Back", font=("Segoe UI", 12),
                 bg=BUTTON_BG, fg=BUTTON_FG, width=14, cursor="hand2",
                 command=self.use_model_screen).pack(side="left", padx=10)
        tk.Button(btn_frame, text="▶ Run Test / Get Accuracy", font=("Segoe UI", 15, "bold"),
                 bg=SUCCESS_COLOR, fg=BG_COLOR, width=24, cursor="hand2",
                 command=self._do_run_test).pack(side="left", padx=10)

        # Results area with scroll
        self.test_results_frame = tk.LabelFrame(content, text=" Test Results ",
                                               font=("Segoe UI", 13, "bold"),
                                               bg=BG_COLOR, fg=ACCENT_COLOR,
                                               bd=2, relief="groove")
        self.test_results_frame.pack(fill="both", expand=True, pady=20)
        
        # Add scrollbar to results text
        results_scroll = ttk.Scrollbar(self.test_results_frame)
        results_scroll.pack(side="right", fill="y")
        
        self.test_results_text = tk.Text(self.test_results_frame, height=16, width=80,
                                        bg="#11111b", fg="#a6e3a1",
                                        font=("Consolas", 10),
                                        state="disabled",
                                        yscrollcommand=results_scroll.set)
        self.test_results_text.pack(fill="both", expand=True, padx=10, pady=10)
        results_scroll.config(command=self.test_results_text.yview)


    def _do_run_test(self):
        test_dataset = self.test_dataset_path.get()
        if not test_dataset or not os.path.exists(test_dataset):
            messagebox.showerror("Error", "Please select a valid test dataset folder!")
            return

        model_path = self.selected_model_path.get()
        imgsz = int(self.test_imgsz.get() or 224)
        device = self.test_device.get()

        self.test_results_text.configure(state="normal")
        self.test_results_text.delete(1.0, "end")
        self.test_results_text.insert("end", "Running validation... please wait.\n")
        self.test_results_text.configure(state="disabled")
        self.root.update()

        thread = threading.Thread(target=self._run_test_thread, args=(model_path, test_dataset, imgsz, device))
        thread.daemon = True
        thread.start()

    def _run_test_thread(self, model_path, test_dataset, imgsz, device):
        try:
            from ultralytics import YOLO
            import json

            model = YOLO(model_path)

            # Get save location
            cm_save_dir = self.cm_save_path.get() or self.save_folder.get() or os.path.expanduser("~")
            os.makedirs(cm_save_dir, exist_ok=True)

            # Build a temporary data.yaml
            data_yaml_path = os.path.join(test_dataset, "_temp_data.yaml")
            class_names = sorted([d.name for d in Path(test_dataset).iterdir() if d.is_dir()])
            
            yaml_content = f"""path: {test_dataset}
train: {test_dataset}
val: {test_dataset}
test: {test_dataset}

nc: {len(class_names)}
names: {class_names}
"""
            with open(data_yaml_path, 'w') as f:
                f.write(yaml_content)

            # Run validation
            results = model.val(
                data=data_yaml_path,
                imgsz=imgsz,
                device=device,
                split='test',
                save_json=True,
                save_conf=True,
                project=cm_save_dir,
                name="test_results",
                exist_ok=True,
            )

            # Clean up temp yaml
            try:
                os.remove(data_yaml_path)
            except Exception:
                pass

            # Extract overall metrics
            metrics = {
                "top1_accuracy": float(results.top1) if hasattr(results, 'top1') else None,
                "top5_accuracy": float(results.top5) if hasattr(results, 'top5') else None,
                "speed_ms_preprocess": float(results.speed['preprocess']) if hasattr(results, 'speed') else None,
                "speed_ms_inference": float(results.speed['inference']) if hasattr(results, 'speed') else None,
                "speed_ms_postprocess": float(results.speed['postprocess']) if hasattr(results, 'speed') else None,
            }

            # Per-image analysis: right vs wrong
            correct_count = 0
            wrong_count = 0
            total_images = 0
            per_image_log = []
            wrong_predictions = []

            names = results.names if hasattr(results, 'names') else {i: f"class_{i}" for i in range(len(class_names))}

            for class_idx, class_name in enumerate(class_names):
                class_dir = Path(test_dataset) / class_name
                if not class_dir.exists():
                    continue
                    
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'):
                        continue
                    
                    total_images += 1
                    
                    pred_result = model.predict(
                        source=str(img_file),
                        imgsz=imgsz,
                        device=device,
                        verbose=False
                    )[0]
                    
                    pred_class_idx = pred_result.probs.top1
                    pred_class = names.get(pred_class_idx, f"class_{pred_class_idx}")
                    confidence = round(float(pred_result.probs.top1conf) * 100, 2)
                    
                    true_class = class_name
                    
                    if pred_class_idx == class_idx:
                        correct_count += 1
                        status = "✅ CORRECT"
                    else:
                        wrong_count += 1
                        status = "❌ WRONG"
                        wrong_predictions.append({
                            "image": str(img_file),
                            "true": true_class,
                            "predicted": pred_class,
                            "confidence": confidence
                        })
                    
                    per_image_log.append(f"{status} | {img_file.name} → True: {true_class} | Pred: {pred_class} ({confidence}%)")

            output = []
            output.append("=" * 70)
            output.append("TEST RESULTS — PER-IMAGE BREAKDOWN")
            output.append("=" * 70)
            output.append(f"Model: {model_path}")
            output.append(f"Test Dataset: {test_dataset}")
            output.append(f"Image Size: {imgsz}")
            output.append(f"Results Folder: {os.path.join(cm_save_dir, 'test_results')}")
            output.append(f"Confusion Matrix: {os.path.join(cm_save_dir, 'test_results', 'confusion_matrix.png')}")
            output.append("-" * 70)

            if per_image_log:
                output.append("Per-Image Results:")
                for line in per_image_log:
                    output.append(f"  {line}")
                output.append("-" * 70)

            output.append("SUMMARY:")
            output.append(f"  Total Images Tested:  {total_images}")
            if total_images > 0:
                output.append(f"  ✅ Correct:           {correct_count} ({(correct_count/total_images*100):.2f}%)")
                output.append(f"  ❌ Wrong:             {wrong_count} ({(wrong_count/total_images*100):.2f}%)")
            else:
                output.append(f"  ✅ Correct:           {correct_count}")
                output.append(f"  ❌ Wrong:             {wrong_count}")
            output.append("-" * 70)

            output.append("YOLO VALIDATION METRICS:")
            output.append(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%" if metrics['top1_accuracy'] else "  Top-1: N/A")
            output.append(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%" if metrics['top5_accuracy'] else "  Top-5: N/A")
            output.append("-" * 70)
            output.append("Speed (ms/image):")
            output.append(f"  Preprocess:  {metrics['speed_ms_preprocess']:.2f}" if metrics['speed_ms_preprocess'] else "  Preprocess: N/A")
            output.append(f"  Inference:   {metrics['speed_ms_inference']:.2f}" if metrics['speed_ms_inference'] else "  Inference: N/A")
            output.append(f"  Postprocess: {metrics['speed_ms_postprocess']:.2f}" if metrics['speed_ms_postprocess'] else "  Postprocess: N/A")
            
            # if wrong_predictions:
            #     output.append("-" * 70)
            #     output.append("WRONG PREDICTIONS DETAIL:")
            #     for wp in wrong_predictions:
            #         output.append(f"  ❌ {wp['image']}")
            #         output.append(f"     True: {wp['true']} | Predicted: {wp['predicted']} ({wp['confidence']}%)")
            
            # output.append("=" * 70)

            # Save JSON results
            save_dir = self.save_folder.get() or os.path.expanduser("~")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(save_dir, f"test_accuracy_{timestamp}.json")
            
            save_data = {
                "model": model_path,
                "test_dataset": test_dataset,
                "timestamp": timestamp,
                "results_folder": os.path.join(cm_save_dir, "test_results"),
                "summary": {
                    "total_images": total_images,
                    "correct": correct_count,
                    "wrong": wrong_count,
                    "accuracy_top1_percent": (correct_count/total_images*100) if total_images > 0 else 0,
                    "yolo_top1_accuracy": metrics['top1_accuracy'],
                    "yolo_top5_accuracy": metrics['top5_accuracy']
                },
                "metrics": metrics,
                "per_image_results": per_image_log,
                "wrong_predictions": wrong_predictions
            }
            
            with open(result_file, 'w') as f:
                json.dump(save_data, f, indent=2)

            output.append(f"\n📄 Results saved to: {result_file}")

            # Auto-export console log to test results folder
            results_folder = os.path.join(cm_save_dir, "test_results")
            log_file = os.path.join(results_folder, f"test_console_log_{timestamp}.txt")
            try:
                self.test_results_text.configure(state="normal")
                console_content = self.test_results_text.get(1.0, "end")
                self.test_results_text.configure(state="disabled")
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(console_content)
                output.append(f"📝 Console log saved to: {log_file}")
            except Exception as e:
                output.append(f"⚠️ Could not save console log: {e}")

            self.test_results_text.configure(state="normal")
            self.test_results_text.delete(1.0, "end")
            for line in output:
                self.test_results_text.insert("end", line + "\n")
            self.test_results_text.configure(state="disabled")
            self.test_results_text.see("end")

        except Exception as e:
            self.test_results_text.configure(state="normal")
            self.test_results_text.delete(1.0, "end")
            self.test_results_text.insert("end", f"Error during testing:\n{str(e)}\n")
            import traceback
            self.test_results_text.insert("end", traceback.format_exc())
            self.test_results_text.configure(state="disabled")
            self.test_results_text.see("end")
    
    def _do_run_inference(self):
        model = self.selected_model_path.get()
        source = self.source_path.get()
        save_dir = self.save_folder.get()
        os.makedirs(save_dir, exist_ok=True)

        export_mode = getattr(self, 'export_mode_var', tk.StringVar(value="top1")).get()

        kwargs_lines = [
            '    source=r"' + source + '",',
            '    device="' + self.device_var.get() + '",',
            '    save=True,',
            '    project=r"' + save_dir + '",',
            '    name="predictions",',
            '    exist_ok=True,',
        ]
        if self.adv_inf_open:
            if hasattr(self, 'inf_imgsz') and self.inf_imgsz.get():
                kwargs_lines.append('    imgsz=' + self.inf_imgsz.get() + ',')
            if hasattr(self, 'half_var') and self.half_var.get():
                kwargs_lines.append('    half=True,')
        kwargs_str = "\n".join(kwargs_lines)

        script_lines = [
            'import sys',
            'sys.path.insert(0, ".")',
            'from ultralytics import YOLO',
            'import os',
            'model = YOLO(r"' + model + '")',
            'results = model.predict(',
            kwargs_str,
            ')',
            'print("\\nClassification complete!")',
            'export_mode = "' + export_mode + '"',
            'for r in results:',
            '    cls_name = r.names[r.probs.top1]',
            '    conf = float(r.probs.top1conf)',
            '    print(f"  Top-1: {cls_name} ({round(conf*100,2)}%)")',
            '    if export_mode in ("top5", "all"):',
            '        top5 = [(r.names[i], float(c)) for i, c in zip(r.probs.top5, r.probs.top5conf)]',
            '        print("  Top-5:", ", ".join([f"{n} ({round(c*100,1)}%)" for n, c in top5]))',
            '    print()',
            'print("\\nResults saved to: " + os.path.join(r"' + save_dir + '", "predictions"))',
        ]
        script = "\n".join(script_lines)

        cmd = [sys.executable, "-c", script]
        self.log("=" * 60)
        self.log("Starting classification inference...")
        self.log("Model: " + model)
        self.log("Source: " + source)
        self.log("Export mode: " + export_mode)
        self.log("Save to: " + save_dir)
        self.log("-" * 60)
        thread = threading.Thread(target=self._run_inference_thread, args=(cmd, save_dir, export_mode))
        thread.daemon = True
        thread.start()

    def _open_results_folder(self, save_dir):
        results_path = os.path.join(save_dir, "predictions")
        if sys.platform == "win32":
            os.startfile(results_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", results_path])
        else:
            subprocess.Popen(["xdg-open", results_path])

    def run_training(self, mode):
        pass

    def execute_command(self, cmd):
        try:
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace'
            )
            last_progress_line = None
            for line in iter(self.current_process.stdout.readline, ''):
                if not line:
                    continue
                line = self._clean_ansi(line)
                if not line:
                    continue
                if '%' in line and (line.count('─') >= 2 or 'it/s' in line or 's/it' in line):
                    if last_progress_line is not None:
                        self.log_replace(line)
                    else:
                        self.log(line)
                    last_progress_line = line
                else:
                    if last_progress_line is not None:
                        self.log("")
                        last_progress_line = None
                    self.log(line)
            self.current_process.stdout.close()
            self.current_process.wait()
            if self.current_process.returncode == 0:
                self.log("\n✅ Process completed successfully!")
            elif self.current_process.returncode == -15 or self.current_process.returncode == 1:
                pass
            else:
                self.log("\n❌ Process exited with code " + str(self.current_process.returncode))
        except Exception as e:
            self.log("\n❌ Error: " + str(e))
        finally:
            self.current_process = None

    def build_log_area(self, parent):
        log_header = tk.Frame(parent, bg=BG_COLOR)
        log_header.pack(anchor="w", pady=(25,5), padx=5, fill="x")
        tk.Label(log_header, text="Console Output:",
                font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(side="left")
        tk.Button(log_header, text="💾 Export Log", font=("Segoe UI", 10),
                 bg=BUTTON_BG, fg=BUTTON_FG, cursor="hand2",
                 command=self.export_log).pack(side="right", padx=5)
        self.log_text = scrolledtext.ScrolledText(
            parent, height=14, width=110,
            bg="#11111b", fg="#a6e3a1",
            font=("Consolas", 10),
            insertbackground=FG_COLOR,
            state="disabled"
        )
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        self._user_scrolled = False
        self.log_text.bind("<MouseWheel>", self._on_log_scroll)
        self.log_text.bind("<Button-4>", self._on_log_scroll)
        self.log_text.bind("<Button-5>", self._on_log_scroll)
        for child in self.log_text.winfo_children():
            if isinstance(child, tk.Scrollbar):
                child.bind("<B1-Motion>", self._on_log_scroll)

    def _on_log_scroll(self, event=None):
        self._user_scrolled = True
        self.root.after(100, self._check_scroll_position)

    def _check_scroll_position(self):
        try:
            bottom = self.log_text.yview()[1]
            if bottom >= 0.98:
                self._user_scrolled = False
        except Exception:
            pass

    def export_log(self):
        path = filedialog.asksaveasfilename(
            title="Export Console Log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Log files", "*.log"), ("All files", "*.*")]
        )
        if path:
            try:
                self.log_text.configure(state="normal")
                content = self.log_text.get(1.0, "end")
                self.log_text.configure(state="disabled")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Export Complete", f"Log saved to: {path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save log: {str(e)}")

    def log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        if not self._user_scrolled:
            self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def log_replace(self, message):
        self.log_text.configure(state="normal")
        end_idx = self.log_text.index("end-1c")
        last_line_start = self.log_text.index("end-1c linestart")
        self.log_text.delete(last_line_start, end_idx)
        self.log_text.insert(last_line_start, message)
        if not self._user_scrolled:
            self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def build_header(self, title):
        header = tk.Frame(self.root, bg=BG_COLOR)
        header.pack(fill="x", pady=20)
        tk.Label(header, text=title, font=("Segoe UI", 26, "bold"),
                bg=BG_COLOR, fg=ACCENT_COLOR).pack()
        tk.Button(header, text="🏠 Home", font=("Segoe UI", 11),
                 bg=BUTTON_BG, fg=BUTTON_FG, cursor="hand2",
                 command=self.build_main_screen).pack(pady=10)

    def _style_combo(self, combo):
        combo.bind("<Map>", lambda e: self._fix_combo_listbox(combo))
        self.root.after(100, lambda: self._fix_combo_listbox(combo))

    def _fix_combo_listbox(self, combo):
        try:
            for child in self.root.winfo_children():
                if isinstance(child, tk.Toplevel):
                    for lb in child.winfo_children():
                        if isinstance(lb, tk.Listbox):
                            lb.configure(bg="white", fg="black", selectbackground="#89b4fa", selectforeground="black")
        except Exception:
            pass

    def toggle_advanced_training(self):
        if not self.adv_train_open:
            self.adv_train_frame.pack(fill="x", pady=15, padx=25, before=self.adv_train_btn)
            self.build_advanced_training()
            self.adv_train_open = True
            self.adv_train_btn.config(text="▲ Hide Advanced Settings")
        else:
            self.adv_train_frame.pack_forget()
            for w in self.adv_inf_frame.winfo_children():
                w.destroy()
            self.adv_train_open = False
            self.adv_train_btn.config(text="▼ Show Advanced Settings")

    def build_advanced_training(self):
        fields = [
            ("Learning Rate (lr0):", "lr0", "0.01"),
            ("Learning Rate Final (lrf):", "lrf", "0.01"),
            ("Momentum:", "momentum", "0.937"),
                       ("Weight Decay:", "weight_decay", "0.0005"),
            ("Warmup Epochs:", "warmup_epochs", "3.0"),
            ("Warmup Momentum:", "warmup_momentum", "0.8"),
            ("Box Loss Gain:", "box", "7.5"),
            ("Cls Loss Gain:", "cls", "0.5"),
            ("Dfl Loss Gain:", "dfl", "1.5"),
            ("Patience (early stopping):", "patience", "50"),
            ("Dropout:", "dropout", "0.0"),
            ("Seed:", "seed", "0"),
        ]
        self.adv_entries = {}
        for i, (label, key, default) in enumerate(fields):
            row = i // 2
            col = (i % 2) * 2
            tk.Label(self.adv_train_frame, text=label, font=("Segoe UI", 11),
                    bg=BG_COLOR, fg=FG_COLOR).grid(row=row, column=col, sticky="w", padx=15, pady=6)
            entry = tk.Entry(self.adv_train_frame, font=("Segoe UI", 11),
                            bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR, width=18)
            entry.insert(0, default)
            entry.grid(row=row, column=col+1, sticky="w", padx=15, pady=6)
            self.adv_entries[key] = entry
        row = (len(fields) // 2) + 1
        tk.Label(self.adv_train_frame, text="Optimizer:", font=("Segoe UI", 11),
                bg=BG_COLOR, fg=FG_COLOR).grid(row=row, column=0, sticky="w", padx=15, pady=6)
        self.optimizer = tk.StringVar(value="auto")
        ttk.Combobox(self.adv_train_frame, textvariable=self.optimizer,
                    values=["auto", "SGD", "Adam", "AdamW", "RMSProp"],
                    state="readonly", width=18).grid(row=row, column=1, sticky="w", padx=15, pady=6)
        row += 1
        self.cache_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.adv_train_frame, text="Cache Dataset (RAM/disk)",
                      variable=self.cache_var,
                      font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR,
                      selectcolor=BG_COLOR).grid(row=row, column=0, columnspan=2, sticky="w", padx=15, pady=6)
        self.amp_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.adv_train_frame, text="AMP (Automatic Mixed Precision)",
                      variable=self.amp_var,
                      font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR,
                      selectcolor=BG_COLOR).grid(row=row+1, column=0, columnspan=2, sticky="w", padx=15, pady=6)
        self.exist_ok_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.adv_train_frame, text="Exist OK (overwrite existing)",
                      variable=self.exist_ok_var,
                      font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR,
                      selectcolor=BG_COLOR).grid(row=row+2, column=0, columnspan=2, sticky="w", padx=15, pady=6)

    def browse_dataset_folder(self):
        path = filedialog.askdirectory(title="Select Dataset Root Folder")
        if path:
            self.dataset_path.set(path)

    def browse_save_folder(self):
        path = filedialog.askdirectory(title="Select Save Folder")
        if path:
            self.save_folder.set(path)


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TCombobox", fieldbackground=ENTRY_BG, background=BUTTON_BG, foreground=FG_COLOR)
    style.configure("TScrollbar", background=BUTTON_BG, troughcolor=BG_COLOR)
    root.state('zoomed')
    app = YoloTrainerApp(root)
    root.mainloop()
