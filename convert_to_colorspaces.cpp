#include <iostream>
#include <string>
#include <sstream>
#include "opencv/cv.h"
#include "opencv/highgui.h"

int main(int argc, char* argv[])
{
  //!< In this variable the current frame is stored
  cv::Mat frame;
  cv::Mat display;

  int cam;
  bool exitFlag = false ;
  std::string windowName("Display Window");
  std::string frameName("Frame");
  std::stringstream ss;

  if (argc <= 1) {
    std::cout << "Usage : ./camera CAM_NUM" << std::endl;
    std::cout << "CAM_NUM is the camera we wish to open!" << std::endl;
    std::cout << "Default Laptop Camera will be opened!" << std::endl;
    cam = 0;
  }
  else
    cam = atoi(argv[1]);

  cv::VideoCapture camera(cam);

  if (!camera.isOpened()) {
    std::cout << "Cannot open the video file" << std::endl;
    return -1;
  }

  char KeyPressed=(char)255; 
  int i=0;

  std::cout << "Press esc if you want to stop the process" << std::endl;
  std::cout << "Press s to save snapshot" << std::endl;
  std::cout << "Press B to convert the image to BGR color Space" << std::endl;
  std::cout << "Press R to convert the image to RGB color Space" << std::endl;
  std::cout << "Press H to convert the image to HSV color Space" << std::endl;
  std::cout << "Press G to convert the image to Grayscale" << std::endl;

  cv::namedWindow(frameName , CV_WINDOW_AUTOSIZE);
  cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);

  display = frame;

  char state;
  while (true) {
    // Get the next frame.
    camera.grab();
    camera.retrieve(frame);

    if (i==0)
      display = frame;

    cv::imshow(frameName,frame);

    KeyPressed=cvWaitKey(10) & 255; 
    //!< wait for 'esc' key press for 10 ms
    //!< different choices
    if (KeyPressed >= 0) {
      switch (KeyPressed) {
        case 's':
        case 'S':
          break;
        default:
          state = KeyPressed;
      }
    }
    
    // Check for image conversion.
    switch (state) {
      // Convert to Gray.
      case 'g':
      case 'G':
        cv::cvtColor(frame, display, CV_BGR2GRAY);
        break;
      // Convert to RGB.
      case 'r':
      case 'R':
        cv::cvtColor(frame, display, CV_BGR2RGB);
        break;
      // Convert to HSV.
      case 'h':
      case 'H':
        cv::cvtColor(frame, display, CV_BGR2HSV);
        break;
      case (char)27:
        exitFlag = true;
        break;        
      case 'b':
      case 'B':
        display = frame;
        break;
      default :
        break;
    }

    // Check if we have to save the image.
    if (KeyPressed == 's' || KeyPressed == 'S') {
      ss << (i+1);
      imwrite( std::string("frame") + ss.str() 
        + std::string(".jpg"), display);
      std::cout << "Frame.jpg " << i << " saved." << std::endl;
      i++;
      ss.str(std::string() );
    }

    cv::imshow(windowName,display);

    if (exitFlag)
      break;
  }
  
  return 0;
}
