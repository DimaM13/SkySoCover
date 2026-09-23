"""SkySoCover GUI - RVC каверы/даббинг. Запуск: .\\.venv\\Scripts\\python.exe cover\\gui.py"""
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CoverGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("SkySoCover (RVC)")
        root.geometry("620x560")
        self.job_q = queue.Queue()
        self.result_path = None

        from cover import voices
        self.vmap = voices.load()
        vnames = sorted(self.vmap.keys())

        f = ttk.Frame(root, padding=10)
        f.pack(fill="both", expand=True)

        row = 0
        ttk.Label(f, text="Видео:").grid(row=row, column=0, sticky="w")
        self.video_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.video_var, width=55).grid(row=row, column=1, sticky="ew")
        ttk.Button(f, text="...", width=3, command=self.pick_video).grid(row=row, column=2)

        row = 1
        ttk.Label(f, text="Голос (RVC):").grid(row=row, column=0, sticky="w")
        self.voice_var = tk.StringVar(value=vnames[0] if vnames else "")
        ttk.Combobox(f, textvariable=self.voice_var, values=vnames, state="readonly", width=25).grid(
            row=row, column=1, sticky="w")
        self.vtype_lbl = ttk.Label(f, text="")
        self.vtype_lbl.grid(row=row, column=2, sticky="w")
        self.voice_var.trace_add("write", lambda *a: self.show_vtype())

        row = 2
        ttk.Label(f, text="Тон (полутоны):").grid(row=row, column=0, sticky="w")
        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="авто", variable=self.auto_var).grid(row=row, column=1, sticky="w")
        self.tr_var = tk.IntVar(value=0)
        ttk.Scale(f, from_=-12, to=12, variable=self.tr_var, length=200).grid(row=row, column=1, sticky="e")
        self.tr_lbl = ttk.Label(f, text="auto")
        self.tr_lbl.grid(row=row, column=2, sticky="w")
        self.tr_var.trace_add("write", lambda *a: self.tr_lbl.config(text=f"{self.tr_var.get():+d}"))
        self.auto_var.trace_add("write", lambda *a: self.tr_lbl.config(
            text="auto" if self.auto_var.get() else f"{self.tr_var.get():+d}"))

        row = 3
        self.autop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Параметры от Gemma (авто)", variable=self.autop_var,
                        command=self._toggle_manual).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=2)
        row = 4
        ttk.Label(f, text="Index rate (вручную):").grid(row=row, column=0, sticky="w")
        self.idx_var = tk.DoubleVar(value=0.75)
        self.idx_scale = ttk.Scale(f, from_=0.0, to=1.0, variable=self.idx_var, length=200)
        self.idx_scale.grid(row=row, column=1, sticky="w")
        self.idx_lbl = ttk.Label(f, text="0.75")
        self.idx_lbl.grid(row=row, column=2, sticky="w")
        self.idx_var.trace_add("write", lambda *a: self.idx_lbl.config(text=f"{self.idx_var.get():.2f}"))

        row = 5
        ttk.Label(f, text="Protect (вручную):").grid(row=row, column=0, sticky="w")
        self.prot_var = tk.DoubleVar(value=0.33)
        self.prot_scale = ttk.Scale(f, from_=0.0, to=0.5, variable=self.prot_var, length=200)
        self.prot_scale.grid(row=row, column=1, sticky="w")
        self.prot_lbl = ttk.Label(f, text="0.33")
        self.prot_lbl.grid(row=row, column=2, sticky="w")
        self.prot_var.trace_add("write", lambda *a: self.prot_lbl.config(text=f"{self.prot_var.get():.2f}"))

        row = 5
        self.music_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Музыка из оригинала после озвучки", variable=self.music_var).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=2)
        row = 6
        ttk.Label(f, text="Громкость музыки:").grid(row=row, column=0, sticky="w")
        self.mvol_var = tk.DoubleVar(value=1.0)
        ttk.Scale(f, from_=0.0, to=1.0, variable=self.mvol_var, length=200).grid(row=row, column=1, sticky="w")

        row = 7
        ttk.Label(f, text="Сохранить как:").grid(row=row, column=0, sticky="w")
        self.out_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.out_var, width=55).grid(row=row, column=1, sticky="ew")
        ttk.Button(f, text="...", width=3, command=self.pick_out).grid(row=row, column=2)

        row = 8
        self.start_btn = ttk.Button(f, text="▶ Сделать кавер", command=self.start)
        self.start_btn.grid(row=row, column=0, columnspan=2, pady=8, sticky="ew")
        self.open_btn = ttk.Button(f, text="Открыть результат", command=self.open_result, state="disabled")
        self.open_btn.grid(row=row, column=2, pady=8)

        row = 9
        self.log = tk.Text(f, height=12, state="disabled")
        self.log.grid(row=row, column=0, columnspan=3, sticky="nsew")
        f.rowconfigure(9, weight=1)
        f.columnconfigure(1, weight=1)
        self.show_vtype()
        self._manual_widgets = [self.idx_scale, self.prot_scale]
        self._toggle_manual()
        self.root.after(200, self.poll_log)

    def show_vtype(self):
        v = self.vmap.get(self.voice_var.get(), {})
        self.vtype_lbl.config(text=f"тип: {v.get('type', '?')}")

    def _toggle_manual(self):
        # ручные слайдеры активны только когда авто выкл - чтобы не смущали
        st = "disabled" if self.autop_var.get() else "normal"
        for w in self._manual_widgets:
            try:
                w.config(state=st)
            except Exception:
                pass

    def pick_video(self):
        p = filedialog.askopenfilename(filetypes=[("Видео", "*.mp4 *.mkv *.avi *.mov *.webm"), ("Все", "*.*")])
        if p:
            self.video_var.set(p)
            if not self.out_var.get():
                base, _ = os.path.splitext(p)
                self.out_var.set(base + "_cover.mp4")

    def pick_out(self):
        p = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if p:
            self.out_var.set(p)

    def append_log(self, msg: str):
        self.log.config(state="normal")
        self.log.insert("end", str(msg).encode("cp1251", "backslashreplace").decode("cp1251") + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def poll_log(self):
        try:
            while True:
                kind, payload = self.job_q.get_nowait()
                if kind == "log":
                    self.append_log(payload)
                elif kind == "done":
                    self.result_path = payload
                    self.start_btn.config(state="normal")
                    self.open_btn.config(state="normal" if payload else "disabled")
                    if payload:
                        messagebox.showinfo("SkySoCover", f"Готово:\n{payload}")
                    else:
                        messagebox.showwarning("SkySoCover", "Не получилось - смотри лог")
        except queue.Empty:
            pass
        self.root.after(200, self.poll_log)

    def open_result(self):
        if self.result_path and os.path.exists(self.result_path):
            os.startfile(self.result_path)

    def start(self):
        video = self.video_var.get().strip()
        if not video or not os.path.exists(video):
            messagebox.showwarning("SkySoCover", "Выбери видео")
            return
        if not self.voice_var.get():
            messagebox.showwarning("SkySoCover", "Выбери голос")
            return
        if self.autop_var.get():
            params = None  # Gemma подберет в пайплайне
        else:
            params = {"index_rate": round(self.idx_var.get(), 2),
                      "rms_mix_rate": 1.0, "protect": round(self.prot_var.get(), 2)}
        job = {
            "input": video,
            "voice": self.voice_var.get(),
            "transpose": None if self.auto_var.get() else int(self.tr_var.get()),
            "params": params,
            "out": self.out_var.get().strip() or None,
            "music": bool(self.music_var.get()),
            "music_vol": round(self.mvol_var.get(), 2),
        }
        self.start_btn.config(state="disabled")
        self.open_btn.config(state="disabled")
        self.result_path = None
        q = self.job_q

        def worker():
            try:
                from cover import cli as cover_cli
                q.put(("log", f"[gui] старт: {os.path.basename(video)} голос {job['voice']}"))
                res = cover_cli.run_job(job, log=lambda m: q.put(("log", m)))
                q.put(("done", res))
            except Exception as e:  # noqa: BLE001
                import traceback
                q.put(("log", f"[gui] ОШИБКА: {type(e).__name__}: {e}"))
                q.put(("log", traceback.format_exc(limit=3)))
                q.put(("done", None))

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    CoverGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
