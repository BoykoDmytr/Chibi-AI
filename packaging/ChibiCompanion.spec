# PyInstaller build of Chibi Companion (one folder, windowed).
#
#     pip install -e ".[voice]" pyinstaller
#     pyinstaller packaging/ChibiCompanion.spec --noconfirm
#
# Output: dist/ChibiCompanion/ChibiCompanion.exe (+ _internal/).  One folder
# rather than one file: starts in ~1 s instead of unpacking ~300 MB to %TEMP%
# on every launch, and antivirus software trusts it more.
# Voice models are NOT bundled; they are downloaded from the Voice settings.

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent  # noqa: F821 - provided by PyInstaller
SRC = ROOT / "src"

# Windows "Details" tab (version, product name), generated from chibi.__version__.
_ns: dict = {}
exec((SRC / "chibi" / "__init__.py").read_text("utf-8"), _ns)
VERSION = _ns["__version__"]
_parts = tuple(int(p) for p in (VERSION.split(".") + ["0"] * 4)[:4])
VERSION_FILE = Path(workpath) / "version_info.txt"  # noqa: F821
VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
VERSION_FILE.write_text(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={_parts}, prodvers={_parts}, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('042204B0', [
      StringStruct('CompanyName', 'Chibi Companion'),
      StringStruct('FileDescription', 'Chibi Companion - desktop companion'),
      StringStruct('FileVersion', '{VERSION}'),
      StringStruct('InternalName', 'ChibiCompanion'),
      StringStruct('OriginalFilename', 'ChibiCompanion.exe'),
      StringStruct('ProductName', 'Chibi Companion'),
      StringStruct('ProductVersion', '{VERSION}')])]),
        VarFileInfo([VarStruct('Translation', [0x0422, 1200])])]
)
""", encoding="utf-8")

datas = collect_data_files("chibi", includes=[
    "assets/characters/*/manifest.json",
    "assets/characters/*/frames/*.png",
    "assets/characters/*/portraits/*.png",
    "assets/characters/*/ASSETS.md",
    "*/data/*.json",
])
# chibi imports some modules lazily (voice, vision, providers): list them all.
hiddenimports = collect_submodules("chibi")

# The voice extra is optional: bundle it only when it is installed.
for package in ("faster_whisper", "ctranslate2", "onnxruntime", "sounddevice", "tokenizers"):
    if importlib.util.find_spec(package) is not None:
        hiddenimports += collect_submodules(package) if package == "faster_whisper" else [package]
        datas += collect_data_files(package)

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(SRC)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        # Never used; keeps the bundle small.
        "tkinter", "unittest", "pydoc", "pytest", "scipy", "PIL",
        "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.QtPdf",
        "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.Qt3DCore", "PySide6.QtCharts",
        "PySide6.QtDataVisualization", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
        "PySide6.QtMultimedia", "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtXml",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ChibiCompanion",
    icon=str(ROOT / "packaging" / "chibi.ico"),
    console=False,
    version=str(VERSION_FILE),
    upx=False,  # UPX-packed executables are a classic antivirus false positive
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ChibiCompanion",
)
