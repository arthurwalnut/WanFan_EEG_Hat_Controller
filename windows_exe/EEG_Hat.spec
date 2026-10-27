# Build with build.py so the scientific libraries match the saved models.
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_submodules, copy_metadata, collect_dynamic_libs

project = Path(SPECPATH).parent
app = project / "EEG_Realtime"
dependencies = app / ".packages"
hiddenimports = collect_submodules("eeg_realtime") + collect_submodules("bci")
for package in ("sklearn", "narwhals"):
    hiddenimports += collect_submodules(package, filter=lambda name: ".tests" not in name)
datas = [(str(app / "eeg_realtime" / "static"), "eeg_realtime/static")]
for distribution in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "narwhals", "cloudpickle"):
    datas += copy_metadata(distribution)
python_license = Path(sys.base_prefix) / "LICENSE.txt"
if python_license.is_file():
    datas.append((str(python_license), "licenses/python"))
binaries = collect_dynamic_libs("sklearn")

a = Analysis(
    [str(project / "windows_exe" / "launcher.py")],
    pathex=[str(app), str(project / "EEG_Signal_to_Decision" / "src"), str(dependencies)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "_pytest", "IPython", "matplotlib", "pandas", "torch", "tensorflow", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="WanFanEEGHatController", debug=False, strip=False, upx=False, console=True,
    disable_windowed_traceback=False,
)
