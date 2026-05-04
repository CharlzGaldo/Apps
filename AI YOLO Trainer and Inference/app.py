import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import yaml
import os
import sys
import threading
import subprocess
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
        
        # Detect if CUDA is available
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False
        
        suffix = " [CUDA]" if cuda_available else " [CPU]"
        self.root.title("AI Trash - YOLO Model Manager" + suffix)
        
        # Set default device based on CUDA detection
        default_device = "0" if cuda_available else "cpu"
        
        self.current_process = None
        self.root.geometry("1000x800")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(900, 700)
        self.selected_model_path = tk.StringVar()
        self.config_path = tk.StringVar()
        self.dataset_path = tk.StringVar()
        self.save_folder = tk.StringVar()
        self.source_path = tk.StringVar()
        self.device_var = tk.StringVar(value=default_device)      # Inference device
        self.train_device = tk.StringVar(value=default_device)    # Training device
        self.current_mode = "new"
        self.adv_inf_open = False
        self.adv_train_open = False
        self.build_main_screen()

    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def build_main_screen(self):
        # Kill any running process first
        if self.current_process is not None:
            try:
                self.current_process.terminate()
            except Exception:
                pass
            self.current_process = None
        
        self.clear_frame()
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
        """Remove ANSI escape codes and progress bar artifacts."""
        import re
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text = ansi_escape.sub('', text)
        # Remove carriage returns (progress bar overwrites)
        text = text.replace('\r', '')
        # Remove the [K clear-line sequence leftover
        text = text.replace('[K', '')
        return text.strip()


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
        tk.Label(settings_frame, text="Confidence Threshold:",
                font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR).grid(row=0, column=0, sticky="w", padx=15, pady=10)
        self.conf_scale = tk.Scale(settings_frame, from_=0.01, to=1.0, resolution=0.01,
                                  orient="horizontal", length=250,
                                  bg=BG_COLOR, fg=FG_COLOR, highlightthickness=0)
        self.conf_scale.set(0.25)
        self.conf_scale.grid(row=0, column=1, sticky="w", padx=15, pady=10)
        tk.Label(settings_frame, text="IoU Threshold:",
                font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR).grid(row=1, column=0, sticky="w", padx=15, pady=10)
        self.iou_scale = tk.Scale(settings_frame, from_=0.01, to=1.0, resolution=0.01,
                                 orient="horizontal", length=250,
                                 bg=BG_COLOR, fg=FG_COLOR, highlightthickness=0)
        self.iou_scale.set(0.45)
        self.iou_scale.grid(row=1, column=1, sticky="w", padx=15, pady=10)
        tk.Label(settings_frame, text="Device:",
                font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR).grid(row=2, column=0, sticky="w", padx=15, pady=10)

        inf_dev_combo = ttk.Combobox(settings_frame, textvariable=self.device_var,
                    values=["0", "0,1","cpu"],
                    state="readonly", width=18)
        inf_dev_combo.grid(row=2, column=1, sticky="w", padx=15, pady=10)
        self._style_combo(inf_dev_combo)
        self.save_txt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(settings_frame, text="Save detection labels (.txt)",
                      variable=self.save_txt_var,
                      font=("Segoe UI", 11), bg=BG_COLOR, fg=FG_COLOR,
                      selectcolor=BG_COLOR, activebackground=BG_COLOR,
                      activeforeground=FG_COLOR).grid(row=3, column=0, columnspan=2, sticky="w", padx=15, pady=10)
        
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
        self.inf_imgsz.insert(0, "640")
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
        
        # --- SETTINGS VIEW ---
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
            self.arch_var = tk.StringVar(value="yolov8s.pt")
            architectures = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
                           "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
                           "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt", "yolov10l.pt", "yolov10x.pt"]
            arch_combo = ttk.Combobox(arch_frame, textvariable=self.arch_var,
                        values=architectures, state="readonly", width=22)
            arch_combo.pack(side="left")
            # Fix dropdown colors
            self._style_combo(arch_combo)
        
        tk.Label(content, text="Select Dataset Config (config.yaml):",
                font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w", pady=(25,5), padx=25)
        config_frame = tk.Frame(content, bg=BG_COLOR)
        config_frame.pack(fill="x", pady=5, padx=25)
        tk.Entry(config_frame, textvariable=self.config_path,
                font=("Segoe UI", 11), bg=ENTRY_BG, fg=ENTRY_FG,
                insertbackground=FG_COLOR, width=55).pack(side="left", padx=(0,10))
        tk.Button(config_frame, text="Browse", font=("Segoe UI", 10),
                 bg=BUTTON_BG, fg=BUTTON_FG, cursor="hand2",
                 command=self.browse_config).pack(side="left")
        
        tk.Label(content, text="Dataset Root Folder (updates config.yaml 'path:'):",
                font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w", pady=(20,5), padx=25)
        ds_frame = tk.Frame(content, bg=BG_COLOR)
        ds_frame.pack(fill="x", pady=5, padx=25)
        tk.Entry(ds_frame, textvariable=self.dataset_path,
                font=("Segoe UI", 11), bg=ENTRY_BG, fg=ENTRY_FG,
                insertbackground=FG_COLOR, width=55).pack(side="left", padx=(0,10))
        tk.Button(ds_frame, text="Browse", font=("Segoe UI", 10),
                 bg=BUTTON_BG, fg=BUTTON_FG, cursor="hand2",
                 command=self.browse_dataset_folder).pack(side="left")
        tk.Button(ds_frame, text="Auto-load from Config", font=("Segoe UI", 10),
                 bg=ACCENT_COLOR, fg=BG_COLOR, cursor="hand2",
                 command=self.auto_load_dataset_path).pack(side="left", padx=10)
        
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
        self.imgsz.insert(0, "640")
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

        # --- CONSOLE VIEW (hidden) ---
        self.train_console_frame = tk.Frame(self.train_main_frame, bg=BG_COLOR)
        
        console_header = tk.Frame(self.train_console_frame, bg=BG_COLOR)
        console_header.pack(fill="x", pady=20)
        tk.Label(console_header, text="Training in Progress...", font=("Segoe UI", 22, "bold"),
                bg=BG_COLOR, fg=SUCCESS_COLOR).pack()
        
        self.build_log_area(self.train_console_frame)
        self.log_text.configure(height=28)
        
        console_btn_frame = tk.Frame(self.train_console_frame, bg=BG_COLOR)
        console_btn_frame.pack(pady=20)
        tk.Button(console_btn_frame, text="← Back to Settings", font=("Segoe UI", 12),
                 bg=BUTTON_BG, fg=BUTTON_FG, width=18, cursor="hand2",
                 command=self.show_train_settings_view).pack(side="left", padx=10)
        
        self.current_mode = mode


    def show_train_console_view(self):
        config = self.config_path.get()
        dataset = self.dataset_path.get()
        save_dir = self.save_folder.get()
        if not config or not os.path.exists(config):
            messagebox.showerror("Error", "Please select a valid config.yaml file!")
            return
        if not dataset:
            messagebox.showerror("Error", "Please select a dataset folder!")
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
        if self.current_process is not None:
            try:
                self.current_process.terminate()
                self.log("\n⏹ Training stopped by user.")
            except Exception:
                pass
            self.current_process = None
        self.train_console_frame.pack_forget()
        self.train_settings_frame.pack(fill="both", expand=True)


    def _do_run_training(self):
        mode = self.current_mode
        config = self.config_path.get()
        dataset = self.dataset_path.get()
        save_dir = self.save_folder.get()
        
        success, msg = self.update_config_path()
        if not success:
            messagebox.showerror("Config Error", "Failed to update config.yaml: " + msg)
            self.show_train_settings_view()
            return
        self.log("Config updated: " + msg)
        
        if mode == "improve":
            model_path = self.selected_model_path.get()
            if not model_path:
                messagebox.showerror("Error", "No base model selected for fine-tuning!")
                self.show_train_settings_view()
                return
        else:
            model_path = self.arch_var.get()
            
        kwargs = {
            "data": config,
            "epochs": int(self.epochs.get() or 100),
            "imgsz": int(self.imgsz.get() or 640),
            "batch": int(self.batch.get() or 16),
            "device": self.train_device.get(),
            "workers": int(self.workers.get() or 8),
            "project": os.path.join(save_dir, self.project_name.get() or "AI_Trash_Training"),
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
        script = 'import sys\n'
        script += 'sys.path.insert(0, ".")\n'
        script += 'from ultralytics import YOLO\n'
        script += 'import warnings\n'
        script += 'warnings.filterwarnings("ignore")\n'
        script += 'print("Loading model...")\n'
        script += 'model = YOLO(r"' + model_path + '")\n'
        script += 'print("Model loaded: ' + model_path + '")\n'
        script += 'print("Starting training with config: ' + config + '")\n'
        script += 'print("Dataset path: ' + dataset + '")\n'
        script += 'print("=" * 60)\n'
        script += 'results = model.train(\n'
        script += '    ' + kwargs_str + '\n'
        script += ')\n'
        script += 'print("=" * 60)\n'
        script += 'print("\\n✅ Training complete!")\n'
        script += 'print("Best model saved to: " + str(results.best))\n'
        cmd = [sys.executable, "-c", script]
        self.log("=" * 60)
        self.log("Starting " + ("fine-tuning" if mode == "improve" else "training") + "...")
        self.log("Model: " + model_path)
        self.log("Config: " + config)
        self.log("Dataset: " + dataset)
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
                # Detect progress bars (contains % and looks like epoch progress)
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


    def log_replace(self, message):
        """Overwrite the last line in the log (for progress bars)."""
        self.log_text.configure(state="normal")
        # Get current end position
        end_idx = self.log_text.index("end-1c")
        # Find start of last line
        last_line_start = self.log_text.index("end-1c linestart")
        # Delete from start of last line to end
        self.log_text.delete(last_line_start, end_idx)
        # Insert new message
        self.log_text.insert(last_line_start, message)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _style_combo(self, combo):
        # Force white bg / black text on the dropdown listbox
        combo.bind("<Map>", lambda e: self._fix_combo_listbox(combo))
        # Also try immediate fix
        self.root.after(100, lambda: self._fix_combo_listbox(combo))

    def _fix_combo_listbox(self, combo):
        try:
            # The listbox is a child Toplevel of the root
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
            for w in self.adv_train_frame.winfo_children():
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

    def browse_config(self):
        path = filedialog.askopenfilename(
            title="Select config.yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if path:
            self.config_path.set(path)
            self.auto_load_dataset_path()

    def browse_dataset_folder(self):
        path = filedialog.askdirectory(title="Select Dataset Root Folder")
        if path:
            self.dataset_path.set(path)

    def browse_save_folder(self):
        path = filedialog.askdirectory(title="Select Save Folder")
        if path:
            self.save_folder.set(path)

    def auto_load_dataset_path(self):
        config_file = self.config_path.get()
        if not config_file or not os.path.exists(config_file):
            return
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if config and 'path' in config:
                self.dataset_path.set(config['path'])
        except Exception:
            pass


    def update_config_path(self):
        config_file = self.config_path.get()
        dataset = self.dataset_path.get()
        if not config_file or not os.path.exists(config_file):
            return False, "Config file not found"
        if not dataset:
            return False, "Dataset folder not selected"
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            config['path'] = dataset
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            return True, "Updated path to: " + dataset
        except Exception as e:
            return False, str(e)

    def _do_run_inference(self):
        model = self.selected_model_path.get()
        source = self.source_path.get()
        save_dir = self.save_folder.get()
        os.makedirs(save_dir, exist_ok=True)
        kwargs_lines = [
            '    source=r"' + source + '",',
            '    conf=' + str(self.conf_scale.get()) + ',',
            '    iou=' + str(self.iou_scale.get()) + ',',
            '    device="' + self.device_var.get() + '",',
            '    save=True,',
            '    save_txt=' + str(self.save_txt_var.get()) + ',',
            '    project=r"' + save_dir + '",',
            '    name="predictions",',
            '    exist_ok=True,',
        ]
        if self.adv_inf_open:
            if hasattr(self, 'max_det') and self.max_det.get():
                kwargs_lines.append('    max_det=' + self.max_det.get() + ',')
            if hasattr(self, 'classes_filter') and self.classes_filter.get():
                kwargs_lines.append('    classes=[' + self.classes_filter.get() + '],')
            if hasattr(self, 'inf_imgsz') and self.inf_imgsz.get():
                kwargs_lines.append('    imgsz=' + self.inf_imgsz.get() + ',')
            if hasattr(self, 'half_var') and self.half_var.get():
                kwargs_lines.append('    half=True,')
        kwargs_str = "\n".join(kwargs_lines)
        script = 'import sys\n'
        script += 'sys.path.insert(0, ".")\n'
        script += 'from ultralytics import YOLO\n'
        script += 'import os\n'
        script += 'model = YOLO(r"' + model + '")\n'
        script += 'results = model.predict(\n'
        script += kwargs_str + '\n'
        script += ')\n'
        script += 'print("\\n✅ Predictions saved to: " + os.path.join(r"' + save_dir + '", "predictions"))\n'
        cmd = [sys.executable, "-c", script]
        self.log("=" * 60)
        self.log("Starting inference...")
        self.log("Model: " + model)
        self.log("Source: " + source)
        self.log("Save to: " + save_dir)
        self.log("-" * 60)
        thread = threading.Thread(target=self._run_inference_thread, args=(cmd, save_dir))
        thread.daemon = True
        thread.start()


    def _run_inference_thread(self, cmd, save_dir):
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
                if hasattr(self, 'open_results_var') and self.open_results_var.get():
                    self._open_results_folder(save_dir)
            elif self.current_process.returncode == -15 or self.current_process.returncode == 1:
                pass
            else:
                self.log("\n❌ Process exited with code " + str(self.current_process.returncode))
        except Exception as e:
            self.log("\n❌ Error: " + str(e))
        finally:
            self.current_process = None


    def _open_results_folder(self, save_dir):
        results_path = os.path.join(save_dir, "predictions")
        if sys.platform == "win32":
            os.startfile(results_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", results_path])
        else:
            subprocess.Popen(["xdg-open", results_path])

    def run_training(self, mode):
        # Replaced by _do_run_training — called via show_train_console_view
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
        tk.Label(parent, text="Console Output:",
                font=("Segoe UI", 12, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(anchor="w", pady=(25,5), padx=5)
        self.log_text = scrolledtext.ScrolledText(
            parent, height=14, width=110,
            bg="#11111b", fg="#a6e3a1",
            font=("Consolas", 10),
            insertbackground=FG_COLOR,
            state="disabled"
        )
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    def log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
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


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TCombobox", fieldbackground=ENTRY_BG, background=BUTTON_BG, foreground=FG_COLOR)
    style.configure("TScrollbar", background=BUTTON_BG, troughcolor=BG_COLOR)
    root.state('zoomed')
    app = YoloTrainerApp(root)
    root.mainloop()
    