#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "include/histogram_edge_blobs.h"
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    cv::Mat _in_image1;
    std::string _input_image1("../walls/0.png");
    _in_image1 = cv::imread(_input_image1, 1);
    cv::Mat _in_image2;
    std::string _input_image2("../walls/1.png");
    _in_image2 = cv::imread(_input_image2, 1);
    //calcHistogram(_in_image);
    calcBackProjection(_in_image1, _in_image2);
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
    int _hist_width = 512, _hist_height = 400;
    int _bin_width = cvRound( (double) _hist_width/_hist_sizes );
    cv::Mat _hist_img ( _hist_height, _hist_width, CV_8UC3, cv::Scalar( 0, 0 ,0) );
    //Normalize to fall into range
    normalize(_b_hist, _b_hist, 0, _hist_img.rows, NORM_MINMAX, -1, cv::Mat() );
    normalize(_g_hist, _g_hist, 0, _hist_img.rows, NORM_MINMAX, -1, cv::Mat() );
    normalize(_r_hist, _r_hist, 0, _hist_img.rows, NORM_MINMAX, -1, cv::Mat() );
    for( int i = 1; i < _hist_sizes; i++ )
    {
        line( _hist_img, Point( _bin_width*(i-1), _hist_height - cvRound(_b_hist.at<float>(i-1)) ) , Point( _bin_width*(i), _hist_height - cvRound(_b_hist.at<float>(i)) ), Scalar( 255, 0, 0), 2, 8, 0  );
        line( _hist_img, Point( _bin_width*(i-1), _hist_height - cvRound(_g_hist.at<float>(i-1)) ) , Point( _bin_width*(i), _hist_height - cvRound(_g_hist.at<float>(i)) ), Scalar( 0, 255, 0), 2, 8, 0  );
        line( _hist_img, Point( _bin_width*(i-1), _hist_height - cvRound(_r_hist.at<float>(i-1)) ) , Point( _bin_width*(i), _hist_height - cvRound(_r_hist.at<float>(i)) ), Scalar( 0, 0, 255), 2, 8, 0  );
     }
    cv::namedWindow("hist", CV_WINDOW_AUTOSIZE);
    cv::imshow("hist", _hist_img);
    cv::waitKey(0);
}


void calcBackProjection( const cv::Mat _in_image1, const cv::Mat _in_image2)
{
    cv::VideoCapture camera( 1 );
    cv::Mat frame;
    cv::Mat _dest_img;
    cv::Mat _hsv_img, _hsv_dest_img;
    cv::Mat _hue_ch, _hue_dest_ch;
    cv::MatND _hist;
    //set number of bins
    int _bins = 30;
    int _hist_sizes = 128;
    float _hue_range[] = {0, 180};
    const float* _ranges = { _hue_range };
    cv::cvtColor( _in_image1, _hsv_img, CV_BGR2HSV);
    //extract and use only the hue value
    _hue_ch.create( _hsv_img.size(), _hsv_img.depth() );
    int ch[] = { 0, 0};
    cv::mixChannels( &_hsv_img, 1, &_hue_ch, 1, ch, 1);
    /// Get the Histogram and normalize it(accumulate)
    cv::calcHist( &_hue_ch, 1, 0, cv::Mat(), _hist, 1, &_hist_sizes, &_ranges, true, true );
    cv::normalize( _hist, _hist, 0, 255, NORM_MINMAX, -1, cv::Mat() );
    cv::cvtColor( _in_image2, _hsv_img, CV_BGR2HSV);
    //extract and use only the hue value
    _hue_ch.create( _hsv_img.size(), _hsv_img.depth() );
    cv::mixChannels( &_hsv_img, 1, &_hue_ch, 1, ch, 1);
    /// Get the Histogram and normalize it (accumulate)
    cv::calcHist( &_hue_ch, 1, 0, cv::Mat(), _hist, 1, &_hist_sizes, &_ranges, true, true );
    cv::normalize( _hist, _hist, 0, 255, NORM_MINMAX, -1, cv::Mat() );

    /// Draw the histogram
    cv::MatND _hist_img;
    int w = 400; int h = 400;
    int _bin_w = cvRound( (double) w / _hist_sizes );
    _hist_img = cv::Mat::zeros( w, h, CV_8UC3 );

    for( int i = 0; i < _bins; i ++ )
    {
         cv::rectangle( _hist_img, Point( i*_bin_w, h ), Point( (i+1)*_bin_w, h - cvRound( _hist.at<float>(i)*h/255.0 ) ), Scalar( 0, 0, 255 ), -1 ); }

    cv::namedWindow("hist", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("BackProj", CV_WINDOW_AUTOSIZE);
    cv::imshow("hist", _hist_img);
    //Get Backprojection
    cv::MatND _backproj;
    while( true )
    {
        // Get the next frame.
        camera.grab();
        camera.retrieve(frame);
        _dest_img = frame;
        cv::cvtColor( _dest_img, _hsv_dest_img, CV_BGR2HSV);
        _hue_dest_ch.create( _hsv_dest_img.size(), _hsv_dest_img.depth() );
        cv::mixChannels( &_hsv_dest_img, 1, &_hue_dest_ch, 1, ch, 1);
        cv::calcBackProject( &_hue_dest_ch, 1, 0, _hist, _backproj, &_ranges, 1, true );
       /// Draw the backproj
       cv::imshow( "BackProj", _backproj );
       if( cv::waitKey(10)>10 ) break;
    }

}
