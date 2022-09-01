using HarmfulContentDetectionRealTime.Models;
using HarmfulContentDetectionRealTimeApp;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace HarmfulContentDetectionRealTime
{

    public partial class MainWindow : System.Windows.Window, ICanvasHandler
    {
        private VideoCapture videoCameraCapture;
        private CancellationTokenSource captureCancellationTokenSource;
        private readonly ManualResetEvent manualReset = new ManualResetEvent(true);
        bool alcohol = false;
        bool violence = true;
        bool cigarette = true;
        public MainWindow()
        {
            InitializeComponent();
            string[] args = Environment.GetCommandLineArgs();
        }

        protected override void OnActivated(EventArgs e)
        {
            base.OnActivated(e);
            StartCameraCapture();
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            StopCameraCapture();
            base.OnClosing(e);
        }

        private void StartCameraCapture()
        {
            if (captureCancellationTokenSource == null)
            {
                captureCancellationTokenSource = new CancellationTokenSource();
                Task.Run(() => CaptureCamera(captureCancellationTokenSource.Token), captureCancellationTokenSource.Token);
            }
        }

        private void StopCameraCapture()
        {
            captureCancellationTokenSource?.Cancel();
        }

        private async Task CaptureCamera(CancellationToken token)
        {
            if (videoCameraCapture == null)
                videoCameraCapture = new VideoCapture(@"C:\\e.mp4");

            videoCameraCapture.Set(VideoCaptureProperties.BufferSize, 3);
            bool bIsOpen = videoCameraCapture.IsOpened();
            if (bIsOpen)
            {
                int iFrameCount = int.MaxValue;
                while (!token.IsCancellationRequested)
                {
                    Mat orgMatrix = new Mat();
                    if (!videoCameraCapture.Read(orgMatrix))
                    {
                        Thread.Sleep(50);
                        continue;
                    }
                    Mat temp = new Mat();
                    if (videoCameraCapture.Read(temp))
                        temp.Dispose();
                    temp = new Mat();
                    if (videoCameraCapture.Read(temp))
                        temp.Dispose();

                    await Application.Current.Dispatcher.InvokeAsync(() =>
                    {
                        try
                        {
                            BitmapImage displayImageSource = new BitmapImage();
                            displayImageSource.BeginInit();
                            displayImageSource.CacheOption = BitmapCacheOption.OnLoad;
                            displayImageSource.StreamSource = orgMatrix.ToMemoryStream();
                            displayImageSource.EndInit();
                            DisplayImage.Source = displayImageSource;
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($": {ex.Message}");
                        }
                    });

                    if (iFrameCount > 5 && manualReset.WaitOne(0))
                    {
                        manualReset.Reset();
                        iFrameCount = 0;
                        Mat resizedMatrix = orgMatrix.Resize(new OpenCvSharp.Size(640, 640));
                        List<Prediction> predictions = new List<Prediction>();
                        if (alcohol == true || cigarette == true || violence == true)
                        {
                            if (cigarette == true && alcohol == false && violence == false)
                            {
                                using var cigaretteScorer = new Scorer<CigaretteModel>("Assets/cigaretteweight/best.onnx");
                                List<Prediction> cigarettePred = cigaretteScorer.Predict(resizedMatrix.ToBitmap());
                                predictions.AddRange(cigarettePred);
                            }
                            if (cigarette == false && alcohol == true && violence == false)
                            {
                                using var alcoholScorer = new Scorer<AlcoholModel>("Assets/alcoholweight/best.onnx");
                                List<Prediction> alcoholPred = alcoholScorer.Predict(resizedMatrix.ToBitmap());
                                predictions.AddRange(alcoholPred);
                                using var yolosScorer = new Scorer<YoloCocoP5Model>("Assets/Weights/yolov5s.onnx");
                                List<Prediction> yolosPred = yolosScorer.Predict(resizedMatrix.ToBitmap());
                                Parallel.ForEach(yolosPred, pred =>
                                {
                                    if (pred.Label.Id == 40 || pred.Label.Id == 41)
                                    {
                                        predictions.Add(pred);
                                    }
                                });
                            }
                            if (cigarette == false && alcohol == false && violence == true)
                            {
                                using var violenceScorer = new Scorer<ViolenceModel>("Assets/violenceweight/best.onnx");
                                List<Prediction> violencePred = violenceScorer.Predict(resizedMatrix.ToBitmap());
                                predictions.AddRange(violencePred);
                                using var yolosScorer = new Scorer<YoloCocoP5Model>("Assets/Weights/yolov5s.onnx");
                                List<Prediction> yolosPred = yolosScorer.Predict(resizedMatrix.ToBitmap());
                                Parallel.ForEach(yolosPred, pred =>
                                {
                                    if (pred.Label.Id == 44)
                                    {
                                        predictions.Add(pred);
                                    }
                                });
                            }
                            if (cigarette == false && alcohol == true && violence == true)
                            {
                                using var alcoholviolenceScorer = new Scorer<AlcoholViolenceModel>("Assets/alcoholviolenceweight/best.onnx");
                                List<Prediction> alcoholviolencePred = alcoholviolenceScorer.Predict(resizedMatrix.ToBitmap());
                                predictions.AddRange(alcoholviolencePred);
                            }
                            if (cigarette == true && alcohol == false && violence == true)
                            {
                                using var cigaretteviolenceScorer = new Scorer<CigaretteViolenceModel>("Assets/cigaretteviolenceweight/best.onnx");
                                List<Prediction> cigaretteviolencePred = cigaretteviolenceScorer.Predict(resizedMatrix.ToBitmap());
                                predictions.AddRange(cigaretteviolencePred);
                            }
                            if (cigarette == true && alcohol == true && violence == false)
                            {
                                using var alcoholcigaretteScorer = new Scorer<AlcoholCigaretteModel>("Assets/alcoholcigaretteweight/best.onnx");
                                List<Prediction> alcoholcigarette = alcoholcigaretteScorer.Predict(resizedMatrix.ToBitmap());
                                predictions.AddRange(alcoholcigarette);
                            }
                            if (cigarette == true && alcohol == true && violence == true)
                            {
                                using var allScorer = new Scorer<AlcoholCigaretteViolenceModel>("Assets/alcoholcigaretteviolenceweight/best.onnx");
                                List<Prediction> allPred = allScorer.Predict(resizedMatrix.ToBitmap());
                                predictions.AddRange(allPred);
                            }
                        }
                        var size = new OpenCvSharp.Size(orgMatrix.Width, orgMatrix.Height);
                        _ = Task.Run(() => DetectAsync(predictions, resizedMatrix, size));
                    }
                    iFrameCount++;
                }
                videoCameraCapture.Release();
            }
            videoCameraCapture.Dispose();
            videoCameraCapture = null;
            captureCancellationTokenSource = null;
        }

        private async Task DetectAsync(List<Prediction> predictions, Mat imageMatrix, OpenCvSharp.Size size)
        {
            try
            {
                List<BoundingBox> boundingBoxes = new List<BoundingBox>();
                if (predictions.Count > 0)
                {
                    foreach (var item in predictions)
                    {
                        BoundingBox a = new BoundingBox();
                        a.Label = item.Label.Name;
                        a.Confidence = item.Score;
                        a.BoxColor = item.Label.Color;
                        a.Dimensions = new BoundingBoxDimensions
                        {
                            X = item.Rectangle.X,
                            Y = item.Rectangle.Y,
                            Height = item.Rectangle.Height,
                            Width = item.Rectangle.Width
                        };
                        boundingBoxes.Add(a);
                    }
                }
                else
                {
                    boundingBoxes = new List<BoundingBox>();
                }

                imageMatrix.Dispose();

                await Application.Current.Dispatcher.InvokeAsync(() =>
                {
                    OverlayFrame.DrawOverlays(this, boundingBoxes, DisplayImage.ActualHeight, DisplayImage.ActualWidth);
                });

            }
            catch (Exception ex)
            {
                Console.WriteLine($": {ex.Message}");
            }
            finally
            {
                manualReset.Set();
            }

        }
        public void AddToCanvas(object control)
        {
            if (control is Rectangle rectangle)
                DisplayImageCanvas.Children.Add(rectangle);
            if (control is TextBlock textBlock)
                DisplayImageCanvas.Children.Add(textBlock);
            else
                Console.WriteLine("Object type not supported..");
        }

        public void Clear()
        {
            DisplayImageCanvas.Children.Clear();
        }
    }
}
