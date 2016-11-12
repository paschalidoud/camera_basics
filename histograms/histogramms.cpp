#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

int main (int argc, char** argv) {
  cv::Mat src, dst;

  /// Load image
  std::string pattern("pattern.png");
  /// Load source image and convert it to gray
  if (argc < 1) {
    std::cout << " Usage : ./histogramms.out image_name" << std::endl;
    std::cout << " Loading default image ! " << std::endl;
  }
  else
    pattern = std::string(argv[1]);

  src = cv::imread(pattern , 1);

  if (!src.data) {
    std::cout << "Failed to read the image!" << std::endl;
    return -1;
  }

  /// Separate the image in 3 places (B,G,R) one for each channel
  std::vector<cv::Mat> bgr_planes;
  split(src, bgr_planes);

  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) between 0 and 255
  float range[] = {0, 256} ;
  const float* histRange = {range};

  bool uniform = true; 
  bool accumulate = false;

  /// Creation of 3 Mat objects to save each histogramm
  cv::Mat b_hist, g_hist, r_hist;

  /// Compute the histograms:
  cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
  cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
  cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

  /// Draw the histograms for B, G and R
  int hist_w = 512; 
  int hist_h = 400;
  int bin_w = cvRound((double) hist_w/histSize);

  cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0));

  /// Normalize the result to [ 0, histImage.rows ]
  cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
  cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
  cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

  /// Draw for each channel
  for ( int i = 1; i < histSize; i++) {
      cv::line(
        histImage,
        cv::Point( bin_w*(i-1),
        hist_h - cvRound(b_hist.at<float>(i-1))),
        cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
        cv::Scalar(255, 0, 0),
        2,
        8,
        0
      );
      cv::line(
        histImage,
        cv::Point( bin_w*(i-1),
        hist_h - cvRound(b_hist.at<float>(i-1))),
        cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
        cv::Scalar(0, 255, 0),
        2,
        8,
        0
      );
      cv::line(
        histImage,
        cv::Point( bin_w*(i-1),
        hist_h - cvRound(b_hist.at<float>(i-1))),
        cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
        cv::Scalar(0, 0, 255),
        2,
        8,
        0
      );
  }

  /// Display
  cv::imshow("Pattern",src);
  cv::imshow("calcHist Demo", histImage);
  cv::waitKey(0);
}
