#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "include/histogram_edge_blobs.h"
#include <stdio.h>
#include <iostream>

int main( int argc, char** argv )
{
    cv::Mat _in_image;
    std::string _input_image("index.jpeg");
    _in_image = cv::imread(_input_image, 1);
    calcHistogram(_in_image);
//    calcBackProjection(_in_image);
//    detectEdges(_in_image);
//    detectBlobs(_in_image);
      
}

void calcHistogram(const cv::Mat& _in_image)
{
    cv::vector<cv::Mat> _bgr_parts;
    cv::split(_in_image, _bgr_parts);
    //set number of bins
    int _hist_sizes = 128;
    //set ranges for b,g,r
    float _range[] = {0, 256};
    const float* _hist_range = {_range};    
    cv::Mat _b_hist, _g_hist, _r_hist;
    cv::calcHist(&_bgr_parts[0], 1, 0, cv::Mat(), _b_hist, 1, &_hist_sizes, &_hist_range, true, false);
    cv::calcHist(&_bgr_parts[1], 1, 0, cv::Mat(), _g_hist, 1, &_hist_sizes, &_hist_range, true, false);
    cv::calcHist(&_bgr_parts[2], 1, 0, cv::Mat(), _r_hist, 1, &_hist_sizes, &_hist_range, true, false);
    //draw the histograms
    cv::namedWindow("hist B", CV_WINDOW_AUTOSIZE);
    cv::imshow("hist B", _b_hist);
    cv::waitKey(0);
}

