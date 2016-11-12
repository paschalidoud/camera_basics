#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

int main ( int argc , char* argv[]) {
    std::string pattern("pattern.png");
    /// Load source image and convert it to gray
    if (argc < 1) {
      std::cout << " Usage : ./histogramms2D.out image_name" << std::endl;
      std::cout << " Loading default image ! " << std::endl;
    }
    else
      pattern = std::string(argv[1]);

    cv::Mat src, hsv;
    src = cv::imread(pattern , 1);

    if (!src.data) {
      std::cout << "Failed to read the image!" << std::endl;
      return -1;
    }
    cv::cvtColor(src, hsv, CV_BGR2HSV);

    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = {0, 180};
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = {0, 256};
    const float* ranges[] = {hranges, sranges};
    cv::MatND hist;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

    cv::calcHist(
        &hsv,
        1,
        channels,
        cv::Mat(), // do not use mask
        hist,
        2,
        histSize,
        ranges,
        true, // the histogram is uniform
        false
    );
    double maxVal=0;
    cv::minMaxLoc(hist, 0, &maxVal, 0, 0);

    int scale = 10;
    cv::Mat histImg = cv::Mat::zeros(sbins*scale, hbins*10, CV_8UC3);

    for (int h = 0; h < hbins; h++) {
        for ( int s = 0; s < sbins; s++ ) {
            float binVal = hist.at<float>(h, s);
            int intensity = cvRound(binVal*255/maxVal);
            cv::rectangle(
                histImg,
                cv::Point(h*scale, s*scale),
                cv::Point( (h+1)*scale - 1, (s+1)*scale - 1),
                cv::Scalar::all(intensity),
                CV_FILLED
            );
        }
    }

    cv::namedWindow("Source", 1);
    cv::imshow("Source", src);

    cv::namedWindow("H-S Histogram", 1);
    cv::imshow("H-S Histogram", histImg);
    cv::waitKey();
}
