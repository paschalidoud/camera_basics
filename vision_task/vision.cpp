#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

//sobel edge detection algorithm
void applySobel(const cv::Mat inImage, cv::Mat* outImage)
{
  //sobel edge detection
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  
  cv::Mat grad;
  cv::Mat edges;
  inImage.copyTo(edges);
  
  /// Generate grad_x and grad_y
  cv::Mat grad_x, grad_y;
  cv::Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  cv::Sobel( edges, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  cv::Sobel( edges, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
  cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  *outImage = grad;
}

int main( int argc, char** argv )
{
  cv::Mat src;
  
  std::string source;
  //give the name of our input file into source variable
  source = std::string(argv[1]);
  
  //load image as it is(3-channel)
  src = cv::imread( argv[1] , 1 );

  //check if loaded correctly
  if(!src.data)
  {
    std::cout<<"no image data\n";
    return -1;
  }

  /// Separate the image in 3 places (B,G,R) one for each channel
  std::vector<cv::Mat> bgr_planes;
  split( src, bgr_planes );

  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) between 0 and 255
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; 
  bool accumulate = false;

  /// Creation of 3 Mat objects to save each histogramm
  cv::Mat b_hist, g_hist, r_hist;

  /// Compute the histograms:
  cv::calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  cv::calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  cv::calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

  /// Draw the histograms for B, G and R
  int hist_w = 512; 
  int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
  cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
  cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                       cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                       cv::Scalar( 255, 0, 0), 2, 8, 0  );
      cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                       cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                       cv::Scalar( 0, 255, 0), 2, 8, 0  );
      cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                       cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                       cv::Scalar( 0, 0, 255), 2, 8, 0  );
  }
  //enable camera
  cv::Mat frame;
  cv::VideoCapture camera(0);
  
  if( !camera.isOpened() )
  {
    std::cout <<"cannot open camera\n";
    return -1;
  }
  
  while( true )
  {
    camera.grab();
    camera.retrieve(frame);
    
    
    //close video with key pressed
    int keyPressed=cv::waitKey(10);
    
    if(keyPressed>0)
    {
     return 0;
    }

  /// Separate the video image in 3 places (B,G,R) one for each channel
  std::vector<cv::Mat> bgr_frame;
  split( frame, bgr_frame );
   
    
  




  //backprojection

  
  cv::MatND backproj1,backproj2,backproj3;

  //backprojection with blue channel
  cv::calcBackProject(&bgr_frame[0], 1, 0, b_hist, backproj1, &histRange, 1, true);

  //backprojection with green channel
  cv::calcBackProject(&bgr_frame[1], 1, 0, g_hist, backproj2, &histRange, 1, true);

  //backprojection with red channel
  cv::calcBackProject(&bgr_frame[2], 1, 0, r_hist, backproj3, &histRange, 1, true);

  //backprojection from all three colors merged for better view and sobel
  cv::Mat dst;
  
  cv::Mat bgr[]={backproj1, backproj2, backproj3};
  cv::merge(bgr, 3, dst);
  
  //sobel edge detection, dst is the merged, change if you want(backproj1-3 if not dst)
  cv::Mat sobel;
  applySobel(dst, &sobel);

  


  //show images

  cv::imshow(source, src);
  cv::imshow("histograms",histImage);
  cv::imshow("backprojection_blue",backproj1);
  cv::imshow("backprojection2_green",backproj2);
  cv::imshow("backprojection_red",backproj3);
  cv::imshow("3-colour_backprojection", dst);
  cv::imshow("sobel_edge_detection",sobel);
  cv::imshow("video",frame);
  
}

  cv::waitKey(0);

  return 0;


}
