"""Headless smoke test of the GUI logic using stub tkinter/customtkinter modules.
Exercises App construction, settings <-> widgets, validation dialogs, and a full run via the event queue.
Usage: python tests/test_gui_logic.py <media_dir> <out_dir>
"""
import os, sys, tempfile, time, types
os.environ["HOME"] = tempfile.mkdtemp(prefix="amvhome_")
os.environ["USERPROFILE"] = os.environ["HOME"]
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class W:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return lambda *a, **k: None
class Var:
    def __init__(self, master=None, value=None, **k): self.v = value if value is not None else self._d()
    def _d(self): return ""
    def get(self): return self.v
    def set(self, v): self.v = v
    def trace_add(self, *a, **k): pass
class SVar(Var): pass
class BVar(Var):
    def _d(self): return False
class DVar(Var):
    def _d(self): return 0.0
shown = []
tk = types.ModuleType("tkinter"); tk.StringVar, tk.BooleanVar, tk.DoubleVar, tk.Listbox = SVar, BVar, DVar, W
mb = types.SimpleNamespace(showinfo=lambda *a, **k: shown.append(("info", a)), showwarning=lambda *a, **k: shown.append(("warn", a)),
                           showerror=lambda *a, **k: shown.append(("error", a)), askyesno=lambda *a, **k: True)
fd = types.SimpleNamespace()
tk.messagebox, tk.filedialog = mb, fd
ctk = types.ModuleType("customtkinter")
for n in ("CTkFrame CTkLabel CTkButton CTkEntry CTkOptionMenu CTkSlider CTkSwitch CTkCheckBox CTkProgressBar "
          "CTkTextbox CTkScrollableFrame CTkFont CTkTabview CTkToplevel CTkInputDialog CTkSegmentedButton CTkRadioButton").split():
    setattr(ctk, n, W)
ctk.set_appearance_mode = ctk.set_default_color_theme = lambda *a, **k: None
ctk.CTk = W
class TV(W):
    def add(self, name): return W()
ctk.CTkTabview = TV
ctk.set_widget_scaling = ctk.set_window_scaling = lambda *a, **k: None
sys.modules.update({"tkinter": tk, "tkinter.messagebox": mb, "tkinter.filedialog": fd, "customtkinter": ctk})
tk.messagebox = mb

from amv_studio.gui import app as A   # noqa: E402
A.messagebox = mb

media, out = Path(sys.argv[1]), Path(sys.argv[2])
class FakeLB:
    def __init__(self): self.items = []
    def insert(self, _, x): self.items.append(x)
    def get(self, a, b=None): return tuple(self.items)
    def delete(self, *a): self.items.clear()
    def curselection(self): return ()
    def configure(self, **k): pass
a = A.App()
a.lb, a.lbm = FakeLB(), FakeLB()
VID, MUS = str(media / "test_gameplay.mp4"), str(media / "test_track.wav")
# 1) validation
a._start(); assert shown[-1][0] == "warn" and not a.running, shown
a.lb.insert("end", VID); a._start(); assert shown[-1][0] == "warn" and not a.running      # no music yet
a.lbm.insert("end", MUS)
a.v_custom.set("banana"); a._start(); assert shown[-1][0] == "warn" and not a.running
a.v_custom.set(""); a.v_size.set("12x"); a._start(); assert shown[-1][0] == "warn" and not a.running
a.v_size.set("")
print("validation OK")
# 2) presets / platforms apply to widgets
a._on_preset("Cinematic Trailer")
assert abs(a.v_cut.get() - 0.30) < 1e-9 and a.v_look.get() == "Cinematic"
a._on_preset("GTA Action (Recommended)")
a._on_platform("YouTube (16:9)"); assert a.v_aspect.get() == "16:9", a.v_aspect.get()
a._on_platform("TikTok"); assert a.v_aspect.get() == "9:16"
# 3) settings <-> widgets round trip incl. new options
s0 = a._collect(validate=False)
s0.fx = ["grain", "vhs", "glow"]; s0.transitions = ["glitch", "rgb", "dip"]; s0.sync_nudge_ms = 30; s0.game_audio = 0.4
s0.title_text = "HELLO"; s0.base_zoom = 1.15; s0.framing = "blur_fit"; s0.pulse_mode = "off"
a._apply_settings_to_ui(s0)
s1 = a._collect(validate=False)
for k in ("fx", "transitions", "sync_nudge_ms", "game_audio", "title_text", "base_zoom", "framing", "pulse_mode"):
    g1, g0 = getattr(s1, k), getattr(s0, k)
    assert (sorted(g1) == sorted(g0)) if isinstance(g0, list) else g1 == g0, (k, g1, g0)
print("settings round trip OK")
# 4) preview run through the real worker process, then final render re-uses the seed
for secs, var in a.v_dur.items(): var.set(secs == 15)
a.v_res.set("720p"); a.v_quality.set("Fast"); a.v_out.set(str(out)); a.v_size.set("")
a._open_path = lambda p: None
def wait():
    t = time.time()
    while a.running and time.time() - t < 400:
        a._poll(); time.sleep(0.2)
    a._poll()
    assert not a.running, "run did not finish"
a._start(preview=True); assert a.running; wait()
assert a.preview_info, "no preview info"
prev_seed = a.preview_info[1]
assert shown[-1][0] != "error", shown[-1]
a._start(); assert a.running
assert a.settings.seed == prev_seed, (a.settings.seed, prev_seed)
wait()
assert shown[-1][0] == "info", shown[-1]
assert len(a.outputs) >= 1 and all(Path(p).exists() for p in a.outputs), a.outputs
assert all(r.state == "done" for r in a.rows.values()), {k: r.state for k, r in a.rows.items()}
assert a.last_report, "edit report missing"
print("GUI run OK; outputs:", [Path(p).name for p in a.outputs])
# 5) cancel path
a._start(); assert a.running; time.sleep(1.0); a._cancel(); wait()
assert not list(Path(out).glob(".amv_work_*"))
print("GUI cancel OK")
# 6) error path
a.lbm.items = [str(media / "nope.wav")]; a._start(); assert shown[-1][0] == "warn" and not a.running
print("GUI error handling OK")
