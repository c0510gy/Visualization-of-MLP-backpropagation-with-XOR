using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Text;
using System.Threading.Tasks;
using System.Drawing.Drawing2D;
using System.Threading;
using System.Runtime.InteropServices; //DLLImport

namespace ANN
{
    class Program
    {
        [DllImport("kernel32.dll", EntryPoint = "GetConsoleWindow", SetLastError = true)]
        private static extern IntPtr GetConsoleHandle();
        static IntPtr handler = GetConsoleHandle();
        
        private static ANN ann;

        private static Bitmap gp;

        private static int frames = 0;

        static void Main(string[] args)
        {
            gp = new Bitmap(1000 + 100 + 400, 1000 + 100);
            Graphics g = Graphics.FromImage(gp);
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            g.Clear(Color.White);

            Pen p = new Pen(Color.Black, 5);
            p.StartCap = LineCap.Round;
            p.EndCap = LineCap.Round;

            g.DrawLine(p, 49, 1051, 1051, 1051);
            g.DrawLine(p, 49, 1051, 49, 49);

            p = new Pen(Color.Black, 2);
            p.StartCap = LineCap.Round;
            p.EndCap = LineCap.Round;

            g.DrawLine(p, 49, 49, 1051, 49);
            g.DrawLine(p, 1051, 49, 1051, 1051);

            Font font = new Font("나눔고딕", 25);
            g.DrawString("0", font, new SolidBrush(Color.Black), 20, 1050);
            g.DrawString("0.5", new Font("나눔고딕", 20), new SolidBrush(Color.Black), 525, 1055);
            g.DrawString("0.5", new Font("나눔고딕", 20), new SolidBrush(Color.Black), 0, 535);
            g.DrawString("1", font, new SolidBrush(Color.Black), 20, 50);
            g.DrawString("1", font, new SolidBrush(Color.Black), 1050, 1050);

            //g.DrawString("정확도 : 00.00 %", font, new SolidBrush(Color.Black), 1060, 50);
            
            ann = new ANN(2, 1, 1, new int[]{2});
            string learningData = System.IO.File.ReadAllText("Learning_data.txt");
            
            while (true)
            {
                double acc = ann.MachineLearning(learningData, false, handler);
                Console.Write("\r" + acc.ToString() + "%");

                //시각화
                //ann.DrawThings(true, handler);

                Bitmap gp_ = new Bitmap(gp);
                g = Graphics.FromImage(gp_);
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;

                for (double y = 0; y <= 1; y += 0.001)
                {
                    ann.Per[0, 1].v = y;
                    for (double x = 0; x <= 1; x += 0.001)
                    {
                        
                        ann.Per[0, 0].v = x;

                        double[] re = ann.solve();
                        if(0.5 <= re[0] && re[0] < 0.51) //중간값
                        {
                            gp_.SetPixel(50 + Convert.ToInt32(x * 1000), 1049 - Convert.ToInt32(y * 1000), Color.Red);
                        }
                        else
                        {
                            gp_.SetPixel(50 + Convert.ToInt32(x * 1000), 1049 - Convert.ToInt32(y * 1000), Color.FromArgb(255, Convert.ToInt32(re[0] * 255), Convert.ToInt32(re[0] * 255), Convert.ToInt32(re[0] * 255)));
                        }
                    }
                }
                g.DrawString("정확도 : " + String.Format("{0:00.00} %", acc.ToString()), new Font("나눔고딕", 30), new SolidBrush(Color.Black), 1060, 50);

                g.DrawLine(p, 550, 49, 550, 1051);
                g.DrawLine(p, 49, 550, 1051, 550);

                frames++;
                System.IO.File.WriteAllText(@"datas\info_" + frames + ".txt", acc.ToString());
                gp_.Save(@"datas\gp_" + frames + ".png", System.Drawing.Imaging.ImageFormat.Png);
                
                if (acc > 99)
                {
                    break;
                }
            }
            ann.MachineLearning(learningData, true, handler);
            ann.visual.Save("tmp.png", System.Drawing.Imaging.ImageFormat.Png);

            Console.WriteLine("\n계산할 parameter 입력");
            while (true)
            {
                for (int j = 0; j < ann.count_P[0]; j++)
                {
                    ann.Per[0, j].v = double.Parse(Console.ReadLine());
                }
                double[] re = ann.solve();
                for(int j = 0; j < re.Count(); j++)
                {
                    Console.Write(re[j] + ",");
                }
                Console.WriteLine();
            }
        }
    }

    class ANN
    {
        public struct PerceptronNet //각 레이어 사이 노드간 관계
        {
            public double w; //가중치
            public bool connect; //링크 확인
            public int colorCode; //색(기본값 0)
        }

        public struct Perceptron //각 퍼셉트론
        {
            public double i; //퍼셉트론 input
            public double v; //퍼셉트론 output
            public double d; //이상적 값
            public double error; //오차
            public int colorCode; //색(기본값 0)
        }
        int numberOfLayers = 5;

        public PerceptronNet[,,] P = new PerceptronNet[5, 100, 100]; //퍼셉트론 [layer, node, n] = Wn, 연결 체크
        public Perceptron[,] Per = new Perceptron[5, 100]; //각 퍼셉트론이 가지는 값
        public int[] count_P = new int[5]; //퍼셉트론의 개수
        double[,] bias = new double[5, 100]; //aX + bias

        int[] ap1 = new int[2]; //활성화된 회로1 [layer, node]
        int[] ap2 = new int[2]; //활성화된 회로2

        int radius = 10;
        int xGap = 100, yGap = 70;
        int maxN = 0; //레이어 당 최대 퍼셉트론 개수

        int testN = 0, nowtestN = 0; //학습 데이터 양, 현재 학습한 양
        public bool learningnow = false; //현재 학습중인가?

        double learningRate = 1; //0.25;

        int cx = 0, cy = 0; //현재 카메라 위치

        public Bitmap visual; //ANN 시각화

        int inputP; int layersN; int outputP; int[] layersP;

        // 입력 퍼셉트론 수, 레이어 수, 출력 퍼셉트론 수, 각 레이어 퍼셉트론 수
        public ANN(int inputP, int layersN, int outputP, int[] layersP){
            this.inputP = inputP; this.layersN = layersN; this.outputP = outputP;
            numberOfLayers = this.layersN + 2; //총 레이어 수
            count_P = new int[numberOfLayers];
            
            maxN = 0;

            count_P[0] = this.inputP;
            if (maxN < count_P[0]) maxN = count_P[0];

            this.layersP = new int[layersP.Count()];
            //레이어당 퍼셉트론 개수 정보
            for (int j = 0; j < this.layersN; j++)
            {
                this.layersP[j] = layersP[j];
                count_P[1 + j] = layersP[j];
                if (maxN < count_P[1 + j]) maxN = count_P[1 + j];
            }

            count_P[this.layersN + 1] = this.outputP;
            if (maxN < count_P[this.layersN + 1]) maxN = count_P[this.layersN + 1];

            P = new PerceptronNet[numberOfLayers, maxN, maxN];
            Per = new Perceptron[numberOfLayers, maxN];
            bias = new double[numberOfLayers, maxN];

            var random = new Random((int)DateTime.Now.Ticks); //초기값
            //link 연결
            for (int j = 1; j < numberOfLayers; j++)
            {
                for (int k = 0; k < count_P[j]; k++)
                {
                    for (int q = 0; q < count_P[j - 1]; q++)
                    {
                        if (random.Next(0, 2) == 0)
                        {
                            P[j, k, q].w = -random.NextDouble(); //랜덤 가중치
                        }
                        else
                        {
                            P[j, k, q].w = random.NextDouble(); //랜덤 가중치
                        }

                        P[j, k, q].connect = true;
                    }
                }
            }
        }

        //학습된 가중치 저장
        public void saveWeight(string path)
        {
            string weightStr = "";

            //ANN에 대한 기본 정보 기록
            weightStr += this.inputP + "," + this.layersN + "," + this.outputP + ",";
            for(int j = 0; j < this.layersN; j++)
            {
                weightStr += this.layersP[j] + ",";
            }
            weightStr += "/";

            //weight값 저장
            for (int j = 1; j < numberOfLayers; j++)
            {
                for (int k = 0; k < count_P[j]; k++)
                {
                    weightStr += j + "," + k + "," + bias[j, k] + "/"; //bias
                    for (int q = 0; q < count_P[j - 1]; q++)
                    {
                        if(P[j, k, q].connect)
                        {
                            weightStr += j + "," + k + "," + q + "," + P[j, k, q].w + "/"; //weight
                        }
                    }
                }
            }

            System.IO.File.WriteAllText(path, weightStr);
        }

        //저장된 가중치 불러오기
        public void loadWeight(string path)
        {
            string weightStr = System.IO.File.ReadAllText(path);

            string[] weights = weightStr.Split('/');

            //ANN에 대한 기본 정보 불러오기
            string[] subs = weights[0].Split(',');
            int L_inputP = int.Parse(subs[0]);
            int L_layersN = int.Parse(subs[1]);
            int L_outputP = int.Parse(subs[2]);
            int[] L_layersP = new int[L_layersN];
            for(int j = 0; j < L_layersN; j++)
            {
                L_layersP[j] = int.Parse(subs[3 + j]);
            }

            
            //현재 ANN에 적용하기

            inputP = L_inputP; layersN = L_layersN; outputP = L_outputP;
            numberOfLayers = layersN + 2; //총 레이어 수\
            count_P = new int[numberOfLayers];
            maxN = 0;

            count_P[0] = this.inputP;
            if (maxN < count_P[0]) maxN = count_P[0];

            this.layersP = new int[L_layersP.Count()];
            //레이어당 퍼셉트론 개수 정보
            for (int j = 0; j < this.layersN; j++)
            {
                this.layersP[j] = L_layersP[j];
                count_P[1 + j] = L_layersP[j];
                if (maxN < count_P[1 + j]) maxN = count_P[1 + j];
            }

            count_P[this.layersN + 1] = this.outputP;
            if (maxN < count_P[this.layersN + 1]) maxN = count_P[this.layersN + 1];

            P = new PerceptronNet[numberOfLayers, maxN, maxN];
            Per = new Perceptron[numberOfLayers, maxN];
            bias = new double[numberOfLayers, maxN];

            //wieght 값 불러오기
            for (int j = 1; j < weights.Count() - 1; j++)
            {
                subs = weights[j].Split(',');

                if(subs.Count() == 4) //weight value
                {
                    int a1 = int.Parse(subs[0]);
                    int a2 = int.Parse(subs[1]);
                    int a3 = int.Parse(subs[2]);
                    double w = double.Parse(subs[3]);
                    P[a1, a2, a3].w = w;
                    P[a1, a2, a3].connect = true;
                }
                else //bias value
                {
                    int a1 = int.Parse(subs[0]);
                    int a2 = int.Parse(subs[1]);
                    double b = double.Parse(subs[2]);
                    bias[a1, a2] = b;
                }
                
            }
        }

        public void DrawThings(bool draw, [Optional] IntPtr handler)
        {
            Font font = new Font("맑은 고딕", 16);
            Brush brush = new SolidBrush(Color.Black);

            visual = new Bitmap(xGap * (numberOfLayers + 1), yGap * (maxN + 1)); //ANN 시각화 Bitmap
            Graphics g = Graphics.FromImage(visual);
            g.Clear(Color.White); // 배경 설정.

            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;

            if (learningnow)
            {
                //e.Graphics.DrawString("SG 인공신경네트워크|" + "인공지능 자가 학습 중..", font, brush, 5, 25);
            }
            else
            {
                //e.Graphics.DrawString("SG 인공신경네트워크" + "", font, brush, 5, 25);
            }

            font = new Font("맑은 고딕", 8);
            brush = new SolidBrush(Color.DarkGray);

            for (int j = 0; j < numberOfLayers; j++)
            {
                for (int k = 0; k < count_P[j]; k++)
                {
                    int h = yGap * (maxN - count_P[j]) / 2; //퍼셉트론 위치를 중앙으로 맞추어 주기 위한 값.
                    Point a = new Point(xGap * (j + 1) - cx, h + yGap * (k + 1) - cy);

                    if (Per[j, k].colorCode == 0)
                    {
                        g.FillEllipse(brush, a.X - radius, a.Y - radius, radius + radius, radius + radius);
                    }
                    else if (Per[j, k].colorCode == 1)
                    {
                        brush = new SolidBrush(Color.LightBlue);
                        g.FillEllipse(brush, a.X - radius, a.Y - radius, radius + radius, radius + radius);
                        brush = new SolidBrush(Color.DarkGray);
                    }

                    g.DrawString("P" + j + k, font, new SolidBrush(Color.Black), a);
                    g.DrawString("" + Math.Round(Per[j, k].v, 2), font, new SolidBrush(Color.Black), xGap * (j + 1) - radius - cx, h + yGap * (k + 1) - radius - cy);
                }
            }

            Pen pen = new Pen(Color.Green, 2); //Pen 객체 생성
            pen.StartCap = LineCap.RoundAnchor; //Line의 시작점 모양 변경 
            pen.EndCap = LineCap.RoundAnchor; //Line의 끝점 모양 변경

            for (int j = 0; j < numberOfLayers - 1; j++)
            {
                for (int k = 0; k < count_P[j]; k++)
                {
                    for (int q = 0; q < count_P[j + 1]; q++)
                    {
                        if (P[j + 1, q, k].connect)
                        { //연결
                            int h1 = yGap * (maxN - count_P[j]) / 2;
                            int h2 = yGap * (maxN - count_P[j + 1]) / 2;
                            Point p1 = new Point(xGap * (j + 1) + radius - cx, h1 + yGap * (k + 1) - cy);
                            Point p2 = new Point(xGap * (j + 2) - radius - cx, h2 + yGap * (q + 1) - cy);

                            if (ap1[0] == j && ap1[1] == k && ap2[0] == j + 1 && ap2[1] == q)
                            {
                                pen = new Pen(Color.Red, 2);
                                g.DrawLine(pen, p1, p2);
                                pen = new Pen(Color.Green, 2);
                            }
                            else
                            {
                                if (P[j + 1, q, k].colorCode == 0)
                                {
                                    g.DrawLine(pen, p1, p2);
                                }
                                else if (P[j + 1, q, k].colorCode == 1)
                                {
                                    pen = new Pen(Color.Blue, 2);
                                    g.DrawLine(pen, p1, p2);
                                    pen = new Pen(Color.Green, 2);
                                }
                            }
                        }
                    }
                }
            }

            visual.RotateFlip(RotateFlipType.Rotate90FlipNone); //시계방향 90도 회전

            //brush = new SolidBrush(Color.Red);
            //e.Graphics.FillEllipse(brush, gole.X - radius, gole.Y - radius, radius + radius, radius + radius); //목적지
            
            //시각화
            /*
            if(draw == true)
            {
                using (var graphics = Graphics.FromHwnd(handler))
                using (var image = (Image)visual)
                    graphics.DrawImage(image, 50, 100, image.Width, image.Height);
            }
            */
        }


        public double[] solve() //인공신경망 계산 => 결과 array리턴
        {
            double[] r = new double[count_P[numberOfLayers - 1]];
            for (int j = 1; j < numberOfLayers; j++)
            {
                for (int k = 0; k < count_P[j]; k++)
                {
                    double v = bias[j, k];
                    for (int q = 0; q < count_P[j - 1]; q++)
                    {
                        if (P[j, k, q].connect)
                        {
                            v += Per[j - 1, q].v * P[j, k, q].w;

                            ap2[0] = j; ap2[1] = k;
                            ap1[0] = j - 1; ap1[1] = q;
                            //Thread.Sleep(100);
                        }
                    }
                    Per[j, k].i = v;
                    Per[j, k].v = fy(v);
                }
            }

            //계산 결과
            //string output = "", upperoutput = "";
            for (int j = 0; j < count_P[numberOfLayers - 1]; j++)
            {
                r[j] = Per[numberOfLayers - 1, j].v;
                //output += Per[numberOfLayers - 1, j].v.ToString() + (j == count_P[numberOfLayers - 1] - 1 ? "" : ",");
                //upperoutput += Math.Round(Per[numberOfLayers - 1, j].v).ToString() + (j == count_P[numberOfLayers - 1] - 1 ? "" : ",");
            }
            return r;
        }

        private double run() //학습 중 인공신경망 계산
        {
            for (int j = 1; j < numberOfLayers; j++)
            {
                for (int k = 0; k < count_P[j]; k++)
                {
                    double v = bias[j, k];
                    for (int q = 0; q < count_P[j - 1]; q++)
                    {
                        if (P[j, k, q].connect)
                        {
                            v += Per[j - 1, q].v * P[j, k, q].w;

                            ap2[0] = j; ap2[1] = k;
                            ap1[0] = j - 1; ap1[1] = q;
                            //Thread.Sleep(1);
                        }
                    }
                    Per[j, k].i = v;
                    Per[j, k].v = fy(v);
                }
            }

            // 계산 정확도 측정
            double ac = 0;
            for (int j = 0; j < count_P[numberOfLayers - 1]; j++)
            {
                ac += Math.Round(100 * (1 - (Math.Max(Per[numberOfLayers - 1, j].v, Per[numberOfLayers - 1, j].d) - Math.Min(Per[numberOfLayers - 1, j].v, Per[numberOfLayers - 1, j].d))), 2);
            }
            ac /= count_P[numberOfLayers - 1];

            return ac; //정확도 반환
        }

        public double MachineLearning(string data, bool draw, [Optional] IntPtr handler) //학습
        {
            learningnow = true;

            double acc = 0;

            string[] datas = data.Split('&');
            testN = datas.Count();
            nowtestN = 0;

            /*
            //학습 진행 상황 출력
            this.Invoke(new Action(delegate () {
                progressBar1.Value = nowtestN;
                progressBar1.Maximum = testN;
                label11.Text = "학습 진행 = " + Math.Round(100 * ((double)nowtestN / (double)testN), 2).ToString() + "% (" + testN + ")";
                //textBox4.Text = Math.Round(Per[3, 0].v).ToString();
            }));
            */

            for (int t = 0; t < testN; t++)
            {
                //set input value
                for (int j = 0; j < datas[t].Split(',').Count() - count_P[numberOfLayers - 1]; j++)
                {
                    Per[0, j].v = double.Parse(datas[t].Split(',')[j]);
                }
                //이상 수치 설정
                for (int j = 0; j < count_P[numberOfLayers - 1]; j++)
                {
                    Per[numberOfLayers - 1, j].d = double.Parse(datas[t].Split(',')[datas[t].Split(',').Count() - count_P[numberOfLayers - 1] + j]);
                }

                acc += run(); //가동

                //역전파
                for (int j = numberOfLayers - 1; j > 0; j--)
                {
                    for (int k = 0; k < count_P[j]; k++)
                    {
                        if (j == numberOfLayers - 1) Per[j, k].error = fprime(Per[j, k].v) * (Per[j, k].d - Per[j, k].v);
                        else
                        {
                            double Errorsum = 0; int cnt = 0;
                            for (int q = 0; q < count_P[j + 1]; q++)
                                if (P[j + 1, q, k].connect)
                                {
                                    Errorsum += Per[j + 1, q].error * P[j + 1, q, k].w;
                                    cnt++;
                                }
                            Per[j, k].error = fprime(Per[j, k].v) * Errorsum / cnt;
                        }
                        for (int q = 0; q < count_P[j - 1]; q++)
                        {
                            if (P[j, k, q].connect)
                            {
                                double delta = Per[j, k].error;//Per[j, k].error * Per[j - 1, q].v * P[j, k, q].w * Math.Exp(-Per[j, k].i) / Math.Pow(1 + Math.Exp(-Per[j, k].i), 2);
                                //if (Per[j, k].d != Math.Round(Per[j, k].v)) { }

                                //가중치 갱신. w = w - E*delta(dE/dw)

                                if (Per[j, k].v >= 0.5)
                                {
                                    P[j, k, q].colorCode = 1;
                                    Per[j, k].colorCode = 1;
                                    Per[j - 1, q].colorCode = 1;
                                }
                                else
                                {
                                    P[j, k, q].colorCode = 0;
                                    Per[j, k].colorCode = 0;
                                    Per[j - 1, q].colorCode = 0;
                                }

                                P[j, k, q].w += Per[j - 1, q].v * delta * learningRate; //learningRate //갱신

                                //j-1, q의 이상수치, 오차 값 계산 필요함.
                                //MessageBox.Show(delta + "");

                                bias[j, k] += delta;
                                Per[j - 1, q].d = Per[j, k].d;
                            }
                        }
                    }
                }
                nowtestN++;
                /*
                //진행상황 갱신
                this.Invoke(new Action(delegate () {
                    progressBar1.Value = nowtestN;
                    label11.Text = "학습 진행 = " + Math.Round(100 * ((double)nowtestN / (double)testN), 2).ToString() + "% (" + testN + ")";
                    //textBox4.Text = Math.Round(Per[3, 0].v).ToString();
                }));
                */

                if(draw == true)
                {
                    DrawThings(true, handler);
                }
            }

            acc /= testN;
            learningnow = false;
            return acc; //평균 정확도 반환
        }

        private double fprime(double x)
        {
            return x * (1 - x);
        }

        private double fy(double X) //Sigmoid function
        {
            return (1 / (1 + Math.Exp(-X)));
        }
    }
}
