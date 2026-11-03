"""从 Release 输出制作不含用户配置的便携 ZIP。"""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tianruoocr-master/bin/x64/Release/net472'
NAME = '天若OCR-Paddle单引擎版-1.4.1-win-x64'
ARCHIVE_NAME = 'TianruoOCR-Paddle-1.4.1-win-x64.zip'


def main():
    if not (SOURCE / '天若OCR文字识别.exe').is_file():
        raise SystemExit('请先按 README 构建 Release x64。')
    dist = ROOT / 'dist'
    destination = dist / NAME
    if destination.exists():
        raise SystemExit('输出目录已存在，请移走旧目录后重新打包。')
    shutil.copytree(SOURCE, destination, ignore=shutil.ignore_patterns(
        'cl-ocr', 'OcrLiteLib.*', '*.pdb', '*.lib', '*.log', 'Data',
        'my.Config', 'Data-my.Config', 'msvcp140*.dll', 'vcruntime140*.dll', 'concrt140.dll'))
    for name in ['README.md', '升级与优化说明.md', 'LICENSE']:
        shutil.copy2(ROOT / name, destination / name)
    shutil.copytree(ROOT / 'docs', destination / 'docs')
    screenshot = ROOT / 'tests/PaddleOnlyUiSmoke/bin/x64/Release/net472/paddle-settings.png'
    if screenshot.exists():
        shutil.copy2(screenshot, destination / 'Paddle设置界面.png')
    files = [p for p in destination.rglob('*') if p.is_file()]
    assert not any(p.name == 'my.Config' or p.name.startswith('OcrLiteLib') for p in files)
    manifest = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (destination / 'SHA256.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    archive = dist / ARCHIVE_NAME
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(dist))
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (dist / 'SHA256SUMS.txt').write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    print(archive.name, archive.stat().st_size, 'bytes')
    print('SHA-256:', digest)


if __name__ == '__main__':
    main()
