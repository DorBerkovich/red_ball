#include <iostream>
#include "rclcpp/rclcpp.hpp"          // Include ROS2 C++ client library
#include "std_msgs/msg/string.hpp"    // Include ROS2 standard message type for string
#include "sensor_msgs/msg/image.hpp"  // Include ROS2 standard message type for image
#include "opencv2/opencv.hpp"         // Include OpenCV library for computer vision
#include "opencv2/imgproc/imgproc.hpp"// Include OpenCV image processing functions

using namespace cv;    // Using the OpenCV namespace for convenience

// Class declaration for RedBallDetector inheriting from rclcpp::Node
class RedBallDetector : public rclcpp::Node 
{
    public:
        // Constructor for RedBallDetector
        RedBallDetector() : Node("red_ball_detector")
        {
            // Start the detection loop
            timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&RedBallDetector::detectBall, this));
        }

    private:
        // Method to detect the red ball
        void detectBall() 
        {
            // Open USB camera
            VideoCapture cap(0, CAP_V4L2);
            
            // Define range for red color in HSV
            Scalar hsvMin = Scalar(0, 96, 0);
            Scalar hsvMax = Scalar(17, 255, 255);

            // Check if the camera is opened successfully
            if (!cap.isOpened()) 
            {
                std::cout << "unable to open the camera" << std::endl;
            }

            // Main loop for ball detection
            while (rclcpp::ok()) 
            {
                // Read frame from camera
                Mat frame;
                cap >> frame;

                // Check if frame is empty
                if (frame.empty()) 
                {
                    std::cout << "empty frame" << std::endl;
                    return;
                }

                // Convert frame to HSV color space
                Mat hsv;
                cvtColor(frame, hsv, COLOR_BGR2HSV);

                // Create mask to detect red color
                Mat mask;
                inRange(hsv, hsvMin, hsvMax, mask);

                // Invert the mask to prepare for blob detection
                // Detector works on black blobs
                Mat reverseMask = 255 - mask;

                // Set parameters for blob detection
                SimpleBlobDetector::Params params;
                params.minThreshold = 0;
                params.maxThreshold = 100;
                params.filterByArea = true;
                params.minArea = 1000;
                params.maxArea = 20000;
                params.filterByCircularity = true;
                params.minCircularity = 0.1;
                params.filterByConvexity = true;
                params.minConvexity = 0.5;
                params.filterByInertia = true;
                params.minInertiaRatio = 0.5;

                // Create blob detector
                Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
                std::vector<KeyPoint> keypoints;
                detector->detect(reverseMask, keypoints);
            
                // Draw circles around detected blobs
                for (const auto& kp : keypoints) 
                {
                    Point center(cvRound(kp.pt.x), cvRound(kp.pt.y));
                    int radius = cvRound(kp.size / 2);
                    // Draw green circle around the detected blob
                    circle(frame, center, radius, Scalar(0, 255, 0), 2);
                    // Draw a cross at the center of the detected blob
                    line(frame, Point(center.x - 10, center.y), Point(center.x + 10, center.y), Scalar(0, 255, 0), 2);
                    line(frame, Point(center.x, center.y - 10), Point(center.x, center.y + 10), Scalar(0, 255, 0), 2);
                }
        
                // Display the frame with detected blobs
                imshow("Blob Detection", frame);
                waitKey(1);
            }
        }

        rclcpp::TimerBase::SharedPtr timer_;  // Timer for detection loop

};

// Main function
int main(int argc, char** argv) 
{   
    rclcpp::init(argc, argv);   // Initialize ROS2 node
    rclcpp::spin(std::make_shared<RedBallDetector>());   // Spin the ROS2 node with RedBallDetector object
    rclcpp::shutdown(); // Shutdown ROS2 node
    return 0;   // Return success
}