using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using System.Xml;
using TrOCR;
using TrOCR.Helper;

internal static class Program
{
    private static readonly BindingFlags Flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;
    private static object Field(object instance, string name) => instance.GetType().GetField(name, Flags).GetValue(instance);
    private static void Require(bool condition, string message) { if (!condition) throw new Exception(message); }

    [STAThread]
    private static int Main()
    {
        string config = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "my.Config");
        byte[] previous = File.Exists(config) ? File.ReadAllBytes(config) : null;
        try
        {
            Application.EnableVisualStyles();
            var program = typeof(FmMain).Assembly.GetType("TrOCR.Program");
            program.GetMethod("InitConfig", BindingFlags.NonPublic | BindingFlags.Static).Invoke(null, null);
            program.GetMethod("DealErrorConfig", BindingFlags.NonPublic | BindingFlags.Static).Invoke(null, null);
            IniHelper.SetValue("OCR2", "boxThresh", "0.427");
            IniHelper.SetValue("配置", "接口", "搜狗");
            using (var main = new FmMain())
            {
                Require(IniHelper.GetValue("配置", "接口") == "Paddle", "Legacy interface was not migrated");
                Require(Field(main, "paddleEngine") != null, "Paddle was not initialized");
                Require(Math.Abs((float)Field(main, "paddleBoxThresh") - 0.427F) < 0.001, "Paddle boxThresh was ignored");
                var menu = (ToolStripMenuItem)Field(main, "Main_jiekou");
                Require(menu.DropDownItems.Count == 1 && menu.DropDownItems[0].Text.Contains("Paddle"), "OCR menu is not Paddle only");
                Require(typeof(FmMain).GetField("sougou", Flags) == null, "CL menu field still exists");
                using (var bitmap = new Bitmap(900, 160))
                using (var graphics = Graphics.FromImage(bitmap))
                using (var font = new Font("Microsoft YaHei", 30))
                {
                    graphics.Clear(Color.White);
                    graphics.DrawString("天若文字识别 1234567890", font, Brushes.Black, 20, 35);
                    typeof(FmMain).GetField("image_screen", Flags).SetValue(main, bitmap);
                    main.OCR_Paddle();
                    var body = Field(main, "RichBoxBody");
                    string result = (string)body.GetType().GetProperty("Text").GetValue(body);
                    Require(result.Contains("天若文字识别") && result.Contains("1234567890"), "Main form OCR failed");
                }
                using (var settings = new FmSetting())
                {
                    settings.StartPosition = FormStartPosition.Manual;
                    settings.Location = new Point(-30000, -30000);
                    settings.ShowInTaskbar = false;
                    settings.Show();
                    Application.DoEvents();
                    var tabs = (TabControl)Field(settings, "tab_标签");
                    Require(!tabs.TabPages.Cast<TabPage>().Any(p => p.Text.Contains("CL") || p.Name == "Page_密钥"), "Obsolete OCR settings are visible");
                    Require(tabs.TabPages.Cast<TabPage>().Count(p => p.Text.Contains("OCR")) == 1, "Expected one Paddle settings page");
                    Require(typeof(FmSetting).GetField("ocrNum1", Flags) == null, "CL settings controls still exist");
                    var threshold = (NumericUpDown)Field(settings, "ocrNum10");
                    Require(threshold.Value == 0.427M, "Settings did not read Paddle threshold");
                    tabs.SelectedTab = tabs.TabPages.Cast<TabPage>().Single(p => p.Text.Contains("OCR"));
                    using (var preview = new Bitmap(settings.Width, settings.Height))
                    {
                        settings.DrawToBitmap(preview, new Rectangle(Point.Empty, preview.Size));
                        preview.Save(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "paddle-settings.png"));
                    }
                    threshold.Value = 0.321M;
                    typeof(FmSetting).GetMethod("Form1_FormClosed", Flags).Invoke(settings,
                        new object[] { settings, new FormClosedEventArgs(CloseReason.UserClosing) });
                    Require(IniHelper.GetValue("OCR2", "boxThresh") == "0.321", "Settings did not save Paddle threshold");
                }
                typeof(FmMain).GetMethod("InitializePaddle", Flags).Invoke(main, null);
                Require(Math.Abs((float)Field(main, "paddleBoxThresh") - 0.321F) < 0.001, "Saved settings did not reach Paddle");
            }
            var document = new XmlDocument();
            document.Load(config);
            Require(document.SelectSingleNode("//OCR") == null, "Fresh configuration still creates CL settings");
            Require(!typeof(FmMain).Assembly.GetReferencedAssemblies().Any(a => a.Name == "OcrLiteLib"), "CL assembly is still referenced");
            Require(HanToPinyin.GetFullPinyin("天若").Length > 0, "Embedded pinyin dictionary is unavailable");
            using (var code = new ZXing.BarcodeWriter
            {
                Format = ZXing.BarcodeFormat.QR_CODE,
                Options = new ZXing.Common.EncodingOptions { Width = 240, Height = 240, Margin = 4 }
            }.Write("Paddle slim 123456"))
            {
                var decoded = new ZXing.BarcodeReader().Decode(code);
                Require(decoded != null && decoded.Text == "Paddle slim 123456", "QR code round trip failed");
            }
            string sound = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "screenshot.wav");
            using (var player = new System.Media.SoundPlayer(sound)) player.Load();
            Console.WriteLine("PASS: embedded pinyin, QR encode/decode, default screenshot sound");
            Console.WriteLine("PASS: CL config migration, Paddle-only menu, main-form OCR, settings read/save/reload, no CL fields or assembly reference");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 1;
        }
        finally
        {
            if (previous == null) File.Delete(config); else File.WriteAllBytes(config, previous);
        }
    }
}
