using Emgu.CV;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;

namespace OcrLib
{
    class CrnnNet : IDisposable
    {
        private readonly float[] MeanValues = { 127.5F, 127.5F, 127.5F };
        private readonly float[] NormValues = { 1.0F / 127.5F, 1.0F / 127.5F, 1.0F / 127.5F };
        private int crnnDstHeight = 32;
        private int fixedWidth;
        private int minimumWidth = 1;

        private InferenceSession crnnNet;
        private List<string> keys;
        private List<string> inputNames;

        public CrnnNet() { }

        public void Dispose()
        {
            crnnNet?.Dispose();
            crnnNet = null;
        }

        public void InitModel(string path, string keysPath, int numThread)
        {
            try
            {
                using (SessionOptions op = new SessionOptions())
                {
                    op.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
                    op.InterOpNumThreads = 1;
                    op.IntraOpNumThreads = Math.Max(1, Math.Min(numThread, Environment.ProcessorCount));
                    var nextSession = new InferenceSession(path, op);
                    crnnNet?.Dispose();
                    crnnNet = nextSession;
                }
                inputNames = crnnNet.InputMetadata.Keys.ToList();
                var shape = crnnNet.InputMetadata[inputNames[0]].Dimensions;
                crnnDstHeight = shape.Length == 4 && shape[2] > 0 ? shape[2] : 32;
                fixedWidth = shape.Length == 4 && shape[3] > 0 ? shape[3] : 0;
                string characters;
                if (crnnNet.ModelMetadata.CustomMetadataMap.TryGetValue("character", out characters))
                {
                    keys = new List<string> { "#" };
                    keys.AddRange(characters.Replace("\r\n", "\n").TrimEnd('\n').Split('\n'));
                    keys.Add(" ");
                    minimumWidth = 320;
                }
                else
                {
                    keys = InitKeys(keysPath);
                    minimumWidth = 1;
                }
                var outputShape = crnnNet.OutputMetadata.First().Value.Dimensions;
                if (outputShape.Length != 3 || (outputShape[2] > 0 && outputShape[2] != keys.Count))
                    throw new InvalidDataException("识别模型输出与字典不匹配，请使用配套的模型和字典。");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
                throw;
            }
        }
        private List<string> InitKeys(string path)
        {
            var result = new List<string> { "#" };
            result.AddRange(File.ReadAllLines(path, Encoding.UTF8));
            result.Add(" ");
            return result;
        }

        public List<TextLine> GetTextLines(List<Mat> partImgs)
        {
            List<TextLine> textLines = new List<TextLine>();
            for (int i = 0; i < partImgs.Count; i++)
            {
                var startTicks = DateTime.Now.Ticks;
                var textLine = GetTextLine(partImgs[i]);
                var endTicks = DateTime.Now.Ticks;
                var crnnTime = (endTicks - startTicks) / 10000F;
                textLine.Time = crnnTime;
                textLines.Add(textLine);
            }
            return textLines;
        }

        private TextLine GetTextLine(Mat src)
        {
            float scale = (float)crnnDstHeight / (float)src.Rows;
            int resizedWidth = Math.Max(1, (int)Math.Ceiling(src.Cols * scale));
            int dstWidth = fixedWidth > 0 ? fixedWidth : Math.Max(minimumWidth, resizedWidth);
            resizedWidth = Math.Min(resizedWidth, dstWidth);

            using (Mat srcResize = new Mat())
            {
                CvInvoke.Resize(src, srcResize, new Size(resizedWidth, crnnDstHeight));
                Tensor<float> inputTensors = OcrUtils.SubstractMeanNormalize(srcResize, MeanValues, NormValues);
                if (resizedWidth < dstWidth)
                {
                    var padded = new DenseTensor<float>(new[] { 1, 3, crnnDstHeight, dstWidth });
                    for (int c = 0; c < 3; c++)
                        for (int y = 0; y < crnnDstHeight; y++)
                            for (int x = 0; x < resizedWidth; x++)
                                padded[0, c, y, x] = inputTensors[0, c, y, x];
                    inputTensors = padded;
                }
                var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensors)
            };
                try
                {
                    using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = crnnNet.Run(inputs))
                    {
                        var resultsArray = results.ToArray();
                        var dimensions = resultsArray[0].AsTensor<float>().Dimensions;
                        float[] outputData = resultsArray[0].AsEnumerable<float>().ToArray();

                        if (dimensions.Length != 3 || dimensions[2] != keys.Count)
                            throw new InvalidDataException("识别模型输出类别数与字典不匹配。");
                        return ScoreToTextLine(outputData, dimensions[1], dimensions[2]);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message + ex.StackTrace);
                    throw;
                }

            }
        }

        private TextLine ScoreToTextLine(float[] srcData, int h, int w)
        {
            StringBuilder sb = new StringBuilder();
            TextLine textLine = new TextLine();

            int lastIndex = 0;
            List<float> scores = new List<float>();

            for (int i = 0; i < h; i++)
            {
                int maxIndex = 0;
                float maxValue = -1000F;
                for (int j = 0; j < w; j++)
                {
                    int idx = i * w + j;
                    if (srcData[idx] > maxValue)
                    {
                        maxIndex = j;
                        maxValue = srcData[idx];
                    }
                }

                if (maxIndex > 0 && maxIndex < keys.Count && (!(i > 0 && maxIndex == lastIndex)))
                {
                    scores.Add(maxValue);
                    sb.Append(keys[maxIndex]);
                }
                lastIndex = maxIndex;
            }
            textLine.Text = sb.ToString();
            textLine.CharScores = scores;
            return textLine;
        }

    }
}