#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

cv::Mat src, src_gray;
int maxCorners = 23;

/**
 * @function goodFeaturesToTrack_Demo.cpp
 * @brief Apply Shi-Tomasi corner detector
 */
void goodFeaturesToTrack_Demo(int, void*) {

  if (maxCorners < 1)
    maxCorners = 1;

  /// Parameters for Shi-Tomasi algorithm
  std::vector< cv::Point2f > corners;
  double qualityLevel = 0.01;
  double minDistance = 10;
  int blockSize = 3;
  bool useHarrisDetector = false;
  double k = 0.04;

  /// Copy the source image
  cv::Mat copy;
  copy = src.clone();

  /// Apply corner detection
  cv::goodFeaturesToTrack(
    src_gray,
    corners,
    maxCorners,
    qualityLevel,
    minDistance,
    cv::Mat(),
    blockSize,
    useHarrisDetector,
    k
  );

  /// Draw corners detected
  std::cout << "** Number of corners detected: " << corners.size() << std::endl;
  int r = 4;
  for (int i = 0; i < corners.size(); i++)
    circle( copy, corners[i], r, cv::Scalar(0,0,255), -1, 8, 0 );

  /// Show what you got
  cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);
  cv::imshow("Image", copy);
}

int main (int argc, char** argv) {
  /// Load source image and convert it to gray
  std::string pattern("pattern.png");
  /// Load source image and convert it to gray
  if (argc < 1) {
    std::cout << " Usage : ./shi_tomashi image_name" << std::endl;
    std::cout << " Loading default image ! " << std::endl;
  }
  else
    pattern = std::string(argv[1]);

  src = cv::imread( pattern , 1 );

  if (!src.data) {
    std::cout << "Failed to load image ! " << std::endl;
    std::cout << "The program cannot continue ! " << std::endl;
    return -1;
  }

  cvtColor(src, src_gray, CV_BGR2GRAY);

  /// Create Window
  cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);

  /// Create Trackbar to set the number of corners
  cv::createTrackbar("Max corners:", "Image", &maxCorners, 100, goodFeaturesToTrack_Demo);

  cv::imshow("Image", src);

  goodFeaturesToTrack_Demo(0, 0);

  cv::waitKey(0);
}
