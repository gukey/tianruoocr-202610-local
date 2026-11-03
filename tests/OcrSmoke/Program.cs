using System;
using System.Drawing;
using System.IO;
using Emgu.CV;
using Emgu.CV.Structure;
using OcrLib;

internal static class Program
{
    private static Ocr Load(string models, int threads = 2)
    {
        var engine = new Ocr();
        engine.InitModels(Path.Combine(models, "det_infer.onnx"), Path.Combine(models, "cls_infer.onnx"),
            Path.Combine(models, "rec_infer.onnx"), Path.Combine(models, "ppocr_keys.txt"), threads);
        return engine;
    }

    private static string Read(Ocr engine, Bitmap bitmap, bool angle = true, int padding = 20)
    {
        using (var input = bitmap.ToImage<Bgr, byte>())
        using (var result = engine.Detect(input.Mat, padding, 1024, 0.5F, 0.3F, 1.6F, angle, false))
        {
            if (result.BoxImg == null || result.BoxImg.IsEmpty) throw new Exception("Missing result image");
            return result.StrRes;
        }
    }

    private static Bitmap Sample()
    {
        var bitmap = new Bitmap(900, 180);
        using (var graphics = Graphics.FromImage(bitmap))
        using (var font = new Font("Microsoft YaHei", 30))
        {
            graphics.Clear(Color.White);
            graphics.DrawString("天若文字识别 升级测试", font, Brushes.Black, 20, 20);
            graphics.DrawString("Paddle OCR 1234567890", font, Brushes.Black, 20, 90);
        }
        return bitmap;
    }

    private static int Main(string[] args)
    {
        try
        {
            using (var sample = Sample())
            using (var engine = Load(args[0]))
            {
                string text = Read(engine, sample);
                Console.WriteLine("RECOGNIZED: " + text);
                if (!text.Contains("天若文字识别") || !text.Contains("1234567890"))
                    throw new Exception("Chinese/English sample recognition failed");
                using (var rotated = (Bitmap)sample.Clone())
                {
                    rotated.RotateFlip(RotateFlipType.Rotate180FlipNone);
                    if (!Read(engine, rotated).Contains("天若文字识别"))
                        throw new Exception("Upside-down text recognition failed");
                }
                using (var blank = new Bitmap(64, 64))
                {
                    using (var graphics = Graphics.FromImage(blank)) graphics.Clear(Color.White);
                    if (!string.IsNullOrWhiteSpace(Read(engine, blank, false, 0)))
                        throw new Exception("Blank image should produce no text");
                }
                string baseline = Read(engine, sample, false);
                for (int i = 0; i < 10; i++)
                    if (Read(engine, sample, false) != baseline) throw new Exception("Repeated result changed");
                Console.WriteLine("PASS: v6 Chinese/English, upside-down text, blank image, padding=0, repeated inference");
            }
            // Fresh and failed engines must both be safe to dispose repeatedly.
            var failed = new Ocr();
            try { failed.InitModels("missing.onnx", "missing.onnx", "missing.onnx", "missing.txt", 0); }
            catch (Exception) { }
            failed.Dispose();
            failed.Dispose();
            using (var sample = Sample())
            using (var engine = Load(args[0], 0))
                if (!Read(engine, sample).Contains("天若文字识别")) throw new Exception("Reload failed");
            Console.WriteLine("PASS: missing model cleanup, repeated disposal, reload, invalid thread count");
            if (args.Length > 1)
            {
                using (var sample = Sample())
                using (var engine = Load(args[1]))
                    if (!Read(engine, sample).Contains("天若")) throw new Exception("Legacy model regression");
                var badKeys = Path.GetTempFileName();
                try
                {
                    File.WriteAllText(badKeys, "wrong\n");
                    using (var engine = new Ocr())
                    {
                        bool rejected = false;
                        try
                        {
                            engine.InitModels(Path.Combine(args[1], "det_infer.onnx"), Path.Combine(args[1], "cls_infer.onnx"),
                                Path.Combine(args[1], "rec_infer.onnx"), badKeys, 2);
                        }
                        catch (InvalidDataException) { rejected = true; }
                        if (!rejected) throw new Exception("Mismatched dictionary was accepted");
                    }
                }
                finally { File.Delete(badKeys); }
                Console.WriteLine("PASS: legacy models, mismatched dictionary rejected");
            }
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error.GetType().FullName);
            Console.Error.WriteLine(error.Message);
            if (error.InnerException != null) Console.Error.WriteLine(error.InnerException.Message);
            return 1;
        }
    }
}
