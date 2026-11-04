"""仅打包已验证的桌面识别运行文件，拒绝混入历史构建残留。"""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tianruoocr-master/bin/x64/Release/net472'
VERSION = '1.4.2'
NAME = f'天若OCR-Paddle精简版-{VERSION}-win-x64'
ARCHIVE_NAME = f'TianruoOCR-Paddle-{VERSION}-win-x64.zip'

# 保留截图、二维码、翻译、资源反序列化及 OCR 的实际依赖。
# cvextern 与 ONNX 原生库不可按文件名猜测删除。
RUNTIME_FILES = (
    '天若OCR文字识别.exe', '天若OCR文字识别.exe.config', 'OcrLib.dll',
    'clipper_library.dll', 'cvextern.dll', 'Emgu.CV.Bitmap.dll',
    'Emgu.CV.Platform.NetStandard.dll', 'System.Drawing.Common.dll',
    'Microsoft.ML.OnnxRuntime.dll', 'onnxruntime.dll', 'onnxruntime_providers_shared.dll',
    'Newtonsoft.Json.dll', 'zxing.dll', 'ShareX.HelpersLib.dll', 'ShareX.ScreenCaptureLib.dll',
    'System.Buffers.dll', 'System.Memory.dll', 'System.Numerics.Vectors.dll',
    'System.Runtime.CompilerServices.Unsafe.dll', 'System.Resources.Extensions.dll',
    'TencentCloudCommon.dll', 'TencentCloudTmt.dll',
    'Data/screenshot.wav',
    'models/paddle-ocr-v6/det_infer.onnx', 'models/paddle-ocr-v6/cls_infer.onnx',
    'models/paddle-ocr-v6/rec_infer.onnx', 'models/paddle-ocr-v6/ppocr_keys.txt',
)


def main():
    for name in RUNTIME_FILES:
        if not (SOURCE / name).is_file():
            raise SystemExit(f'缺少运行文件，请先构建 Release x64：{name}')
    dist = ROOT / 'dist'
    destination = dist / NAME
    if destination.exists():
        raise SystemExit('输出目录已存在，请移走旧目录后重新打包。')
    for name in RUNTIME_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, target)
    shutil.copy2(ROOT / 'LICENSE', destination / 'LICENSE')
    shutil.copy2(ROOT / 'docs/精简版使用说明.md', destination / '使用说明.md')
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    manifest = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (destination / 'SHA256.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    archive = dist / ARCHIVE_NAME
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(dist))
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (dist / f'SHA256SUMS-{VERSION}.txt').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    print(archive.name, archive.stat().st_size, 'bytes', flush=True)
    print('Files:', len(files) + 1, 'unpacked bytes:', sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()), flush=True)
    print('SHA-256:', digest, flush=True)


if __name__ == '__main__':
    main()
