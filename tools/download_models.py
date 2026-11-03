"""Download the verified CPU OCR model set with Python's standard library."""
import hashlib
import pathlib
import urllib.request

BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/"
FILES = {
    "det_infer.onnx": ("onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx", "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
    "rec_infer.onnx": ("onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx", "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
    "cls_infer.onnx": ("onnx/PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx", "54379ae5174d026780215fc748a7f31910dee36818e63d49e17dc598ecc82df7"),
    "ppocr_keys.txt": ("paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt", "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d"),
}


def main():
    destination = pathlib.Path(__file__).resolve().parents[1] / "tianruoocr-master/models/paddle-ocr-v6"
    destination.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        target = destination / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            print(f"Verified {name}")
            continue
        temporary = target.with_suffix(target.suffix + ".download")
        try:
            print(f"Downloading {name}")
            with urllib.request.urlopen(BASE + url, timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"SHA-256 mismatch: {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
