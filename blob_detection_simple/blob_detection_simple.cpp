#include <iostream>
#include <stdio.h>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>

class ComputerVisionSimple
{
  public:
    ComputerVisionSimple() : HISTOGRAM_BINS(25)
    {
    }
    void applyTransformations(std::string image);
  private:
    void histAndBackProj();
    void applyCanny(const cv::Mat& inImage, cv::Mat* outImage);
    void applySobel(const cv::Mat inImage, cv::Mat* outImage);
    void blobDetect(cv::Mat srcimage, cv::Mat* out);
    void getFrameHSV(int cameraMode, cv::Mat* frame);
    void readImageFromWall(std::string imageName, cv::Mat* frame);
  private:
    cv::Mat originalRGBImage_, originalHSVImage_, originalHueImage_, hueImage2_;
    int HISTOGRAM_BINS;
};

void ComputerVisionSimple::applyTransformations(std::string image)
{
  originalRGBImage_ = cv::imread(image, 1);
  if(!originalRGBImage_.data)
  {
    std::cout << "Could not open or find the image" << std::endl;
    return;
  }

  cvtColor(originalRGBImage_, originalHSVImage_, CV_BGR2HSV);
  
  originalHueImage_.create(originalHSVImage_.size(), originalHSVImage_.depth());
  int ch[] = {0, 0};
  mixChannels(&originalHSVImage_, 1, &originalHueImage_, 1, ch, 1);

  // Create Trackbar to enter the number of bins
  std::string window_image = "Source image";
  cv::namedWindow(window_image, CV_WINDOW_AUTOSIZE);
  //cv::createTrackbar("* originalHueImage  bins: ", window_image, &HISTOGRAM_BINS, 180, histAndBackProj);
  histAndBackProj();

  /// Show the image
  cv::imshow(window_image, originalHSVImage_);

  /// Wait until user exits the program
  cvWaitKey(0);
}

void ComputerVisionSimple::applySobel(const cv::Mat inImage, cv::Mat* outImage)
{
  //!< appropriate values for scale, delta and ddepth
  int scale = 1;
  int delta = 0; //!< the value for the non-edges
  int ddepth = CV_16S;
  cv::Mat edges;
  inImage.copyTo(edges);
  cvtColor(edges, edges, CV_RGB2GRAY);
  //!< Generate grad_x and grad_y
  cv::Mat grad_x, grad_y;
  cv::Mat abs_grad_x, abs_grad_y;
  //!< Gradient X
  cv::Sobel(edges, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
  cv::convertScaleAbs(grad_x, abs_grad_x);
  //!< Gradient Y
  cv::Sobel(edges, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
  cv::convertScaleAbs(grad_y, abs_grad_y);
  // Total Gradient (approximate)
  cv::Mat grad_g;
  cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad_g);
  *outImage = grad_g;
}

void ComputerVisionSimple::applyCanny(const cv::Mat& inImage, cv::Mat* outImage)
{
  inImage.copyTo(*outImage);
  cv::Mat detected_edges;
  cv::Mat dst;
  int ratio = 3;
  int kernel_size = 3;
  int lowThreshold = 100;
  //!< Reduce noise with a kernel 3x3
  cv::blur(*outImage, detected_edges, cv::Size(kernel_size, kernel_size));
  //!< Canny detector
  cv::Canny(detected_edges, detected_edges, lowThreshold,
  lowThreshold * ratio, kernel_size);
  //!< Using Canny's output as a mask, we display our result
  dst = cv::Scalar::all(0);
  outImage->copyTo(dst, detected_edges);
  *outImage = dst;
}

void ComputerVisionSimple::blobDetect(cv::Mat srcimage, cv::Mat* out)
{
  std::vector<cv::KeyPoint> keyPoints;
  cv::Mat circularBlobs;
  std::vector< std::vector <cv::Point> > contours;
  std::vector< std::vector <cv::Point> > approxContours;
  cv::SimpleBlobDetector::Params params;
  params.minThreshold = 40;
  params.maxThreshold = 60;
  params.thresholdStep = 5;
  params.minArea = 550;
  params.minConvexity = 0.6;
  params.minInertiaRatio = 0.5;
  params.maxArea = 8000;
  params.maxConvexity = 10;
  params.maxCircularity=1;
  params.minCircularity=0.3;
  params.filterByColor = false;
  params.filterByCircularity = true;
  cv::SimpleBlobDetector blobDetector( params );
  blobDetector.create("SimpleBlob");
  blobDetector.detect( srcimage, keyPoints );
  //blobDetector.detectEx( srcimage, keyPoints, contours );
  cv::drawKeypoints( srcimage, keyPoints, *out, CV_RGB(0,255,255), cv::DrawMatchesFlags::DEFAULT);
  approxContours.resize( contours.size() );
  for( int i = 0; i < contours.size(); ++i )
  {
    cv::approxPolyDP( cv::Mat(contours[i]), approxContours[i], 4, 1 );
    cv::drawContours( *out, contours, i, CV_RGB(255,0,255));
    cv::drawContours( *out, approxContours, i, CV_RGB(255,0,255));
  }
}

void ComputerVisionSimple::histAndBackProj()
{
  cv::MatND hist;
  int histSize = MAX(HISTOGRAM_BINS, 2);
  float hue_range[] = {0, 180};
  const float* ranges = {hue_range};

  // Get the Histogram and normalize it
  calcHist(&originalHueImage_, 1, 0, cv::Mat(), hist, 1, &histSize, &ranges, true, false);
  normalize(hist, hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());

  // Draw the histogram
  int w = 400;
  int h = 400;
  int bin_w = cvRound((double) w / histSize);
  cv::Mat histImg = cv::Mat::zeros(w, h, CV_8UC3);

  for(int i = 0; i < HISTOGRAM_BINS; i++)
  { 
    rectangle(histImg, cv::Point(i * bin_w, h), cv::Point((i+1) * bin_w, h - cvRound(hist.at<float>(i) * h / 255.0)),
      cv::Scalar(0, 0, 255), -1);
  }

  cv::imshow("Histogram", histImg);
  
  // Open Camera
  int cameraMode = 0;
  cv::Mat frame;
  // getFrameHSV(cameraMode, &frame);
  readImageFromWall("/home/miltos/camera_basics/walls/2.png", &frame);
  
  hueImage2_.create(frame.size(), frame.depth());
  int ch[] = {0, 0};
  mixChannels(&frame, 1, &hueImage2_, 1, ch, 1);

  // Get Backprojection
  cv::MatND backproj;
  calcBackProject(&hueImage2_, 1, 0, hist, backproj, &ranges, 1, true);
  // Draw the backproj
  cv::imshow("BackProj", backproj);
  
  // Apply Canny
  cv::Mat cannyImage;
  applyCanny(hueImage2_, &cannyImage);
  cv::imshow("Image after Canny", cannyImage);
  
  cv::Mat blobs=cv::Mat::zeros(cannyImage.size(), cannyImage.depth());
  blobDetect(cannyImage, &blobs);
  cv::imshow("Blobs frame", blobs);
}

void ComputerVisionSimple::getFrameHSV(int cameraMode, cv::Mat* frame)
{
  cv::VideoCapture camera(cameraMode);
  
  if (!camera.isOpened())  
  {
    std::cout << "Cannot open the video file" << std::endl;
    return;
  }

  camera.grab();
  camera.retrieve(*frame);
  // cvtColor(*frame, *frame, CV_BGR2HSV);
  imshow("Camera Frame", *frame);
}

void ComputerVisionSimple::readImageFromWall(std::string imageName, cv::Mat* frame)
{
  cv::Mat frameTemp;
  frameTemp = cv::imread(imageName, 1);
  cvtColor(frameTemp, *frame, CV_BGR2HSV);
  imshow("File Frame", *frame);
}

int main(int argc, char* argv[])
{
  // Default image
  std::string image("/home/miltos/camera_basics/walls/0.png");
  if(argc < 2)
  {
    std::cout << "Loading Default Image" << std::endl;
  }
  else
  {
    image = std::string(argv[1]);
  }

  ComputerVisionSimple findBlobsSimple;
  findBlobsSimple.applyTransformations(image);

  return 0;
}
