# -*- coding: utf-8 -*-
import os
import sys
import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import ocr as ocr_module


class IdCardOcrApp:
    def __init__(self, root):
        self.root = root
        self.root.title("身份证识别系统")
        self.root.geometry("900x650")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")

        self.image_path = None
        self.photo = None

        self._build_ui()

    def _build_ui(self):
        # 顶部标题
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=50)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        tk.Label(
            title_frame, text="身份证识别系统",
            font=("Microsoft YaHei", 18, "bold"),
            fg="white", bg="#2c3e50"
        ).pack(expand=True)

        # 主体区域
        body = tk.Frame(self.root, bg="#f0f0f0")
        body.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # 左侧：图片预览
        left_frame = tk.LabelFrame(body, text="图片预览", font=("Microsoft YaHei", 10), bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.canvas = tk.Canvas(left_frame, bg="#e0e0e0", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # 右侧：识别结果
        right_frame = tk.LabelFrame(body, text="识别结果", font=("Microsoft YaHei", 10), bg="#f0f0f0")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.result_frame = tk.Frame(right_frame, bg="#f0f0f0")
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 结果字段
        self.fields = {}
        field_defs = [
            ("姓名", "name"),
            ("性别", "sex"),
            ("民族", "ethnic"),
            ("出生年", "year"),
            ("月", "month"),
            ("日", "day"),
            ("住址", "addr"),
            ("身份证号码", "idnum"),
        ]

        for i, (label_text, key) in enumerate(field_defs):
            row = tk.Frame(self.result_frame, bg="#f0f0f0")
            row.pack(fill=tk.X, pady=3)

            tk.Label(
                row, text=label_text + "：", font=("Microsoft YaHei", 10),
                width=10, anchor="e", bg="#f0f0f0"
            ).pack(side=tk.LEFT)

            var = tk.StringVar(value="")
            entry = tk.Entry(
                row, textvariable=var, font=("Microsoft YaHei", 10),
                state="readonly", readonlybackground="white", relief=tk.GROOVE
            )
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
            self.fields[key] = var

        # 底部按钮区域
        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, padx=15, pady=(5, 15))

        self.btn_select = tk.Button(
            btn_frame, text="选择图片", font=("Microsoft YaHei", 11),
            command=self._select_image, bg="#3498db", fg="white",
            activebackground="#2980b9", activeforeground="white",
            relief=tk.FLAT, padx=20, pady=5, cursor="hand2"
        )
        self.btn_select.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_recognize = tk.Button(
            btn_frame, text="开始识别", font=("Microsoft YaHei", 11),
            command=self._recognize, bg="#27ae60", fg="white",
            activebackground="#219a52", activeforeground="white",
            relief=tk.FLAT, padx=20, pady=5, cursor="hand2", state=tk.DISABLED
        )
        self.btn_recognize.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_clear = tk.Button(
            btn_frame, text="清空", font=("Microsoft YaHei", 11),
            command=self._clear, bg="#e74c3c", fg="white",
            activebackground="#c0392b", activeforeground="white",
            relief=tk.FLAT, padx=20, pady=5, cursor="hand2"
        )
        self.btn_clear.pack(side=tk.LEFT)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = tk.Label(
            self.root, textvariable=self.status_var,
            font=("Microsoft YaHei", 9), anchor="w",
            bg="#bdc3c7", relief=tk.SUNKEN, padx=10
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _select_image(self):
        filetypes = [
            ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
            ("所有文件", "*.*")
        ]
        path = filedialog.askopenfilename(
            title="选择身份证图片", filetypes=filetypes
        )
        if not path:
            return

        self.image_path = path
        self._display_image(path)
        self.btn_recognize.config(state=tk.NORMAL)
        self.status_var.set(f"已选择: {os.path.basename(path)}")

    def _display_image(self, path):
        try:
            img = Image.open(path)
            self._current_pil_img = img
            self._fit_image_to_canvas()
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图片: {e}")

    def _fit_image_to_canvas(self):
        if not hasattr(self, '_current_pil_img') or self._current_pil_img is None:
            return
        img = self._current_pil_img
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 400, 300

        iw, ih = img.size
        scale = min(cw / iw, ch / ih, 1.0)
        new_w = int(iw * scale)
        new_h = int(ih * scale)

        resized = img.resize((new_w, new_h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)

    def _on_canvas_resize(self, event):
        self._fit_image_to_canvas()

    def _recognize(self):
        if not self.image_path:
            messagebox.showwarning("提示", "请先选择图片")
            return

        self.btn_recognize.config(state=tk.DISABLED)
        self.btn_select.config(state=tk.DISABLED)
        self.status_var.set("正在识别中，请稍候...")

        # 在子线程中运行识别，避免界面卡死
        thread = threading.Thread(target=self._do_recognize, daemon=True)
        thread.start()

    def _do_recognize(self):
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                self.root.after(0, lambda: messagebox.showerror("错误", "无法读取图片文件"))
                return

            ret, msg, path = ocr_module.detect(img)

            if path != '':
                img = cv2.imread(path)
                ret, msg, _ = ocr_module.detect(img)
                try:
                    os.unlink(path)
                except:
                    pass

            if ret and isinstance(msg, list) and len(msg) >= 8:
                field_keys = ["name", "sex", "ethnic", "year", "month", "day", "addr", "idnum"]
                result = dict(zip(field_keys, msg))
                self.root.after(0, lambda: self._show_result(result))
            else:
                error_msg = msg if isinstance(msg, str) else "无法识别，请换一张清晰度更高的照片"
                self.root.after(0, lambda: self._show_error(error_msg))

        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))

    def _show_result(self, result):
        for key, var in self.fields.items():
            var.set(result.get(key, ""))
        self.btn_recognize.config(state=tk.NORMAL)
        self.btn_select.config(state=tk.NORMAL)
        self.status_var.set("识别完成")

    def _show_error(self, msg):
        for var in self.fields.values():
            var.set("")
        self.fields["name"].set(f"识别失败: {msg}")
        self.btn_recognize.config(state=tk.NORMAL)
        self.btn_select.config(state=tk.NORMAL)
        self.status_var.set("识别失败")

    def _clear(self):
        self.image_path = None
        self._current_pil_img = None
        self.photo = None
        self.canvas.delete("all")
        for var in self.fields.values():
            var.set("")
        self.btn_recognize.config(state=tk.DISABLED)
        self.status_var.set("就绪")


def main():
    root = tk.Tk()
    app = IdCardOcrApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
