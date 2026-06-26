"""Offscreen test: load Johnson's documented geometry (patent 4,151,431) and
read the real computed work/revolution straight off the live counter."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from magnetic_motor_ui import MainWindow, JOHNSON_PRESET

app = QApplication(sys.argv)
win = MainWindow()
win.resize(1260, 820); win.show()

print("Johnson preset ->", {k: JOHNSON_PRESET[k] for k in
      ("grade_stator", "grade_rotor", "gap", "skew_deg", "pattern", "n_stator", "R_track")})


def snap(label):
    f = win.worker.latest
    print(f"  {label}: t={f.t:6.2f}s rpm={f.rpm:7.1f} rev={f.revolutions:4d} "
          f"prop/rev={f.prop_last*1e3:+9.3f}mJ switch/rev={f.switch_last*1e3:+9.3f}mJ "
          f"net/rev={f.net_last*1e3:+9.3f}mJ field={f.field_energy_pct:5.1f}% "
          f"ke={f.ke*1e3:8.2f}mJ")


def s0():
    win.controls.apply_preset(dict(JOHNSON_PRESET))      # rebuild gate with Johnson dims
    QTimer.singleShot(200, s1)


def s1():
    win.worker.push(8.0)                                 # hand-start
    QTimer.singleShot(1200, s2)


def s2():
    snap("Johnson, +1.2s after push"); QTimer.singleShot(1500, s3)


def s3():
    snap("Johnson, +1.5s more")
    pix = win.grab(); pix.save("figs/mag_johnson.png")
    print("saved figs/mag_johnson.png", pix.width(), "x", pix.height())
    win.worker.stop(); win.thread.quit(); win.thread.wait(2000); app.quit()


QTimer.singleShot(300, s0)
sys.exit(app.exec())
