using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace OcrLib
{
    class AngleNet : IDisposable
    {
        private readonly float[] MeanValues = { 127.5F, 127.5F, 127.5F };
        private readonly float[] NormValues = { 1.0F / 127.5F, 1.0F / 127.5F, 1.0F / 127.5F };
        private int angleDstWidth = 192;
        private int angleDstHeight = 48;
        private const int angleCols = 2;
        private InferenceSession angleNet;
        private List<string> inputNames;

        public AngleNet() { }

        public void Dispose()
        {
            angleNet?.Dispose();
            angleNet = null;
        }

        public void InitModel(string path, int numThread)
        {
            try
            {
                using (SessionOptions op = new SessionOptions())
                {
                    op.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
                    op.InterOpNumThreads = 1;
                    op.IntraOpNumThreads = Math.Max(1, Math.Min(numThread, Environment.ProcessorCount));
                    angleNet = new InferenceSession(path, op);
                }
                inputNames = angleNet.InputMetadata.Keys.ToList();
                var shape = angleNet.InputMetadata[inputNames[0]].Dimensions;
                angleDstHeight = shape[2] > 0 ? shape[2] : 48;
                angleDstWidth = shape[3] > 0 ? shape[3] : 192;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
                throw;
            }
        }

        public List<Angle> GetAngles(List<Mat> partImgs, bool doAngle, bool mostAngle)
        {
            List<Angle> angles = new List<Angle>();
            if (doAngle)
            {
                for (int i = 0; i < partImgs.Count; i++)
                {
                    var startTicks = DateTime.Now.Ticks;
                    var angle = GetAngle(partImgs[i]);
                    var endTicks = DateTime.Now.Ticks;
                    var angleTime = (endTicks - startTicks) / 10000F;
                    angle.Time = angleTime;
                    angles.Add(angle);
                }
            }
            else
            {
                for (int i = 0; i < partImgs.Count; i++)
                {
                    var angle = new Angle();
                    angle.Index = -1;
                    angle.Score = 0F;
                    angles.Add(angle);
                }
            }
            //Most Possible AngleIndex
            if (doAngle && mostAngle)
            {
                List<int> angleIndexes = new List<int>();
                angles.ForEach(x => angleIndexes.Add(x.Index));

                double sum = angleIndexes.Sum();
                double halfPercent = angles.Count / 2.0f;
                int mostAngleIndex;
                if (sum < halfPercent)
                {//all angle set to 0
                    mostAngleIndex = 0;
                }
                else
                {//all angle set to 1
                    mostAngleIndex = 1;
                }
                Console.WriteLine($"Set All Angle to mostAngleIndex({mostAngleIndex})");
                for (int i = 0; i < angles.Count; ++i)
                {
                    Angle angle = angles[i];
                    angle.Index = mostAngleIndex;
                    angles[i] = angle;
                }
            }
            return angles;
        }

        private Angle GetAngle(Mat src)
        {
            using (Mat angleImg = new Mat())
            {
                int resizedWidth = angleDstHeight == 80
                    ? Math.Max(1, Math.Min(angleDstWidth, (int)Math.Ceiling(src.Cols * angleDstHeight / (double)src.Rows)))
                    : angleDstWidth;
                CvInvoke.Resize(src, angleImg, new Size(resizedWidth, angleDstHeight));
                Tensor<float> inputTensors = OcrUtils.SubstractMeanNormalize(angleImg, MeanValues, NormValues);
                if (resizedWidth < angleDstWidth)
                {
                    var padded = new DenseTensor<float>(new[] { 1, 3, angleDstHeight, angleDstWidth });
                    for (int c = 0; c < 3; c++)
                        for (int y = 0; y < angleDstHeight; y++)
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
                    using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = angleNet.Run(inputs))
                    {
                        var resultsArray = results.ToArray();
                        Console.WriteLine(resultsArray);
                        float[] outputData = resultsArray[0].AsEnumerable<float>().ToArray();
                        return ScoreToAngle(outputData, angleCols);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message + ex.StackTrace);
                    throw;
                }
            }
        }

        private Angle ScoreToAngle(float[] srcData, int angleCols)
        {
            Angle angle = new Angle();
            int angleIndex = 0;
            float maxValue = -1000.0F;
            for (int i = 0; i < angleCols; i++)
            {
                if (i == 0) maxValue = srcData[i];
                else if (srcData[i] > maxValue)
                {
                    angleIndex = i;
                    maxValue = srcData[i];
                }
            }
            angle.Index = angleIndex;
            angle.Score = maxValue;
            return angle;
        }

        private Mat AdjustTargetImg(Mat src, int dstWidth, int dstHeight)
        {
            Mat srcResize = new Mat();
            float scale = (float)dstHeight / (float)src.Rows;
            int angleWidth = (int)((float)src.Cols * scale);
            CvInvoke.Resize(src, srcResize, new Size(angleWidth, dstHeight));
            Mat srcFit = new Mat(dstHeight, dstWidth, DepthType.Cv8U, 3);
            //srcFit.SetTo(new MCvScalar(255,255,255));
            if (angleWidth < dstWidth)
            {
                CvInvoke.CopyMakeBorder(srcResize, srcFit, 0, 0, 0, dstWidth - angleWidth, BorderType.Isolated, new MCvScalar(255, 255, 255));
            }
            else
            {
                Rectangle rect = new Rectangle(0, 0, dstWidth, dstHeight);
                Mat partAngle = new Mat(srcResize, rect);
                partAngle.CopyTo(srcFit);
            }
            return srcFit;
        }

    }
}
