"""Offscreen smoke test for magnetic_motor_ui.py (v2 fields)."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from magnetic_motor_ui import MainWindow, REF

app = QApplication(sys.argv)
win = MainWindow()
win.resize(1260, 820); win.show()

def snap(label):
    f = win.worker.latest
    print(f"  {label}: t={f.t:6.2f}s rpm={f.rpm:6.1f} rev={f.revolutions:4d} "
          f"prop/rev={f.prop_last*1e3:+8.3f}mJ switch/rev={f.switch_last*1e3:+8.3f}mJ "
          f"field={f.field_energy_pct:5.1f}% ke={f.ke*1e3:6.2f}mJ")

def s1(): win.worker.push(8.0); QTimer.singleShot(1000, s2)
def s2(): snap("after push +1.0s"); QTimer.singleShot(1500, s3)
def s3(): snap("after +1.5s more"); QTimer.singleShot(1500, s4)
def s4():
    snap("after +1.5s more")
    pix = win.grab(); pix.save("figs/mag_screenshot.png")
    print("saved figs/mag_screenshot.png", pix.width(), "x", pix.height())
    win.worker.stop(); win.thread.quit(); win.thread.wait(2000); app.quit()

print(f"switching = {REF['switching']}  engage_arc={REF['engage_arc_deg']}deg  "
      f"retract={REF['retract_mm']}mm  friction={REF['friction']}")
QTimer.singleShot(300, s1)
sys.exit(app.exec())
