#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

Mat src;
Mat hsv;
Mat hue;
int bins = 25;

void Hist_and_Backproj (int, void*) {
  MatND hist;
  int histSize = MAX(bins, 2);
  float hue_range[] = {0, 180};
  const float* ranges = {hue_range};

  /// Get the Histogram and normalize it
  calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
  normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

  /// Get Backprojection
  MatND backproj;
  calcBackProject(&hue, 1, 0, hist, backproj, &ranges, 1, true);

  /// Draw the backproj
  imshow("BackProj", backproj);

  /// Draw the histogram
  int w = 400;
  int h = 400;
  int bin_w = cvRound((double) w / histSize);
  Mat histImg = Mat::zeros(w, h, CV_8UC3);

  for (int i = 0; i < bins; i ++) {
    rectangle(
        histImg,
        Point(i*bin_w, h),
        Point( 
            (i+1)*bin_w, h - cvRound( hist.at<float>(i)*h/255.0 )
        ),
        Scalar(0, 0, 255),
        -1
    );
  }

  imshow( "Histogram", histImg);
}

int main (int argc, char** argv) {
  /// Read the image
  std::string pattern("pattern.png");
  /// Load source image and convert it to gray
  if (argc < 1) {
    std::cout << " Usage : ./backprojection.out image_name" << std::endl;
    std::cout << " Loading default image ! " << std::endl;
  }
  else
    pattern = std::string(argv[1]);

  src = cv::imread( pattern , 1 );

  if (!src.data) {
    std::cout << "Failed to read the image!" << std::endl;
    return -1;
  }

  /// Transform it to HSV
  cvtColor(src, hsv, CV_BGR2HSV);

  /// Use only the Hue value
  hue.create(hsv.size(), hsv.depth());
  int ch[] = {0, 0};
  mixChannels(&hsv, 1, &hue, 1, ch, 1);

  /// Create Trackbar to enter the number of bins
  
  createTrackbar("* Hue  bins: ", "Source image", &bins, 180, Hist_and_Backproj);
  Hist_and_Backproj(0, 0);

  /// Show the image
  imshow("Source image", src);

  /// Wait until user exits the program
  waitKey(0);
  return 0;
}


