# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas_ctk, binaries_ctk, hiddenimports_ctk = collect_all('customtkinter')
datas_ct, binaries_ct, hiddenimports_ct = collect_all('ctranslate2')
datas_onnx, binaries_onnx, hiddenimports_onnx = collect_all('onnxruntime')

a = Analysis(
    ['gui_ctk.py'],
    pathex=[],
    binaries=binaries_ctk + binaries_ct + binaries_onnx,
    datas=datas_ctk + datas_ct + datas_onnx,
    hiddenimports=hiddenimports_ctk + hiddenimports_ct + hiddenimports_onnx + [
        'faster_whisper', 'tokenizers', 'huggingface_hub', 'av'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='VoiceToText_CTK',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
