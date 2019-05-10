/*

The MIT License

Copyright (c) 2015 Avi Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/
#include <stdlib.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "vo_features.h"



using namespace cv;
using namespace std;

#define MAX_FRAME 2894
#define MIN_NUM_FEAT 2000
#define PNPRANSAC false

// IMP: Change the file directories (4 places) according to where your dataset is saved before running!

void get_Name_and_Scale(int frame_id, char* rgbname, char* depthname, char* time)	{
  string line;
  char listname[200];
  int i = 0;
  char listFileDir[200] = "/home/linaro/dataset/tum/rgbd_dataset_freiburg2_desk/";
  sprintf(listname, "%smatch.txt", listFileDir);
  
  //cout << listname << endl;
  ifstream myfile (listname);
  char tmp[30];
  if (myfile.is_open())
  {
    while (( getline (myfile,line) ) && (i<=frame_id))
    {
      std::istringstream in(line);
      //cout << line << '\n';
      for (int j=0; j<4; j++)  {
        in >> tmp ;
        //cout << "tmp is " << tmp << endl;
        if (j==3) sprintf(depthname, "%s%s", listFileDir, tmp);
        if (j==1) sprintf(rgbname, "%s%s", listFileDir, tmp);
        if (j==0) sprintf(time, "%s", tmp);
        //cout << "rgbname is " << rgbname << endl;
      }
      
      i++;
    }
    myfile.close();
  }
  else {
    cout << "Unable to open file";
  }
}

int main( int argc, char** argv )	{
  DPUKernel *kernel;
  DPUTask *task;
  device_setup(kernel, task);
  
  long t1,t2;

  Mat img_1, img_2;
  Mat R_f, t_f; //the final rotation and tranlation vectors containing the 

  ofstream myfile;
  myfile.open ("results1_1.txt");
  myfile << "# timestamp tx ty tz qx qy qz qw" << endl;

  //double scale = 1.00;
  char time_stamp [30];
  char filename1[200];
  char filename2[200];
  char depthname1[200];
  char depthname2[200];
  // sprintf(filename1, "/home/share/kitti_odometry/dataset/sequences/00/image_0/%06d.png", 0);
  // sprintf(filename2, "/home/share/kitti_odometry/dataset/sequences/00/image_0/%06d.png", 1);
  get_Name_and_Scale(0, filename1, depthname1, time_stamp);
  get_Name_and_Scale(1, filename2, depthname2, time_stamp);

  char text[100];
  int fontFace = FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 1;  
  cv::Point textOrg(10, 50);

  //read the first two frames from the dataset
  Mat img_1_c = imread(filename1);
  Mat img_2_c = imread(filename2);
  Mat d1 = imread ( depthname1, CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
  cout << "d1.type : " << d1.type() << endl;

  if ( !img_1_c.data || !img_2_c.data ) { 
    std::cout<< " --(!) Error reading images " << std::endl; return -1;
  }

  // we work with grayscale images
  cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
  cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);
  cout << "img_1.rows : " << img_1.rows << endl;
  cout << "img_1.cols : " << img_1.cols << endl;
  cout << "img_1.at<uint8_t>(2, 2) is " << int(img_1.at<uint8_t>(479, 639)) << endl;
  //cout << "img_1.isContinuous() is " << img_1.isContinuous() << endl;
  
  

  // feature detection, tracking
  vector<Point3f> points1;
  vector<Point2f> points2;        //vectors to store the coordinates of the feature points
  // featureDetection(img_1, points1);        //detect features in img_1
  // vector<uchar> status;

  //Mat K = ( Mat_<double> ( 3,3 ) << 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1 );//freiburg1
  Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );//freiburg2
  //Mat K = ( Mat_<double> ( 3,3 ) << 535.4, 0, 320.1, 0, 539.2, 247.6, 0, 0, 1 );//freiburg3
  featureTracking_superpoint(task, img_1,img_2,d1,K,points1,points2); //track those features to img_2
  cout<<"2d pairs: "<<points2.size() <<endl;

  //TODO: add a fucntion to load these values directly from KITTI's calib files
  // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
  /* double focal = 718.8560;
  cv::Point2d pp(607.1928, 185.2157);
  //recovering the pose and the essential matrix
  Mat E, R, t, mask;
  E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
  recoverPose(E, points2, points1, R, t, focal, pp, mask); */

  
  Mat r, t;
  if ( PNPRANSAC )
    solvePnPRansac ( points1, points2, K, Mat(), r, t, false, 100, 8.0, 0.99, noArray(), cv::SOLVEPNP_EPNP ); 
  else
    solvePnP ( points1, points2, K, Mat(), r, t, false, cv::SOLVEPNP_EPNP ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
  Mat R;
  cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
  R = R.t();
  t = -R * t;

  Mat prevImage = img_2;
  Mat currImage;
  vector<Point3f> prevFeatures;
  vector<Point2f> currFeatures;
  Mat dpic;

  char filename[200];

  R_f = R.clone();
  t_f = t.clone();

  clock_t begin = clock();

  namedWindow( "Road facing camera", WINDOW_AUTOSIZE );// Create a window for display.
  namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

  Mat traj = Mat::zeros(600, 600, CV_8UC3);

  for(int numFrame=2; numFrame < MAX_FRAME; numFrame++)	{
  	//sprintf(filename, "/home/share/kitti_odometry/dataset/sequences/00/image_0/%06d.png", numFrame);
    dpic = imread ( depthname2, CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    
    get_Name_and_Scale(numFrame, filename, depthname2, time_stamp);

    //cout << "Scale is " << scale << endl;
    
    cout << numFrame << endl;
  	Mat currImage_c = imread(filename);
  	cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
  	// vector<uchar> status;
  	
    t1=clock();//程序段开始前取得系统运行时间(ms)

    featureTracking_superpoint(task, prevImage, currImage, dpic, K, prevFeatures, currFeatures);
    
    t2=clock();//程序段结束后取得系统运行时间(ms)
    //cout << "featureTracking time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;

  	t1=clock();//程序段开始前取得系统运行时间(ms)
    /* E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
  	recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask); */

    if ( PNPRANSAC )
        solvePnPRansac ( prevFeatures, currFeatures, K, Mat(), r, t, false, 100, 8.0, 0.99, noArray(), cv::SOLVEPNP_EPNP ); 
    else
        solvePnP ( prevFeatures, currFeatures, K, Mat(), r, t, false, cv::SOLVEPNP_EPNP ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    R = R.t(); 
    t = -R * t;
    cout<<"R="<<R<<endl;
    cout<<"t="<<t<<endl;
    
    /* Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2,currFeatures.size(), CV_64F);


    for(int i=0;i<prevFeatures.size();i++)	{   //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
  		prevPts.at<double>(0,i) = prevFeatures.at(i).x;
  		prevPts.at<double>(1,i) = prevFeatures.at(i).y;

  		currPts.at<double>(0,i) = currFeatures.at(i).x;
  		currPts.at<double>(1,i) = currFeatures.at(i).y;
    } */
    
    //cout << "t_norm is :" << norm(t) << endl;

    //if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {

      t_f = t_f + (R_f*t);
      R_f = R_f*R; 

    //}
  	
    // else {
     // cout << "scale below 0.1, or incorrect translation" << endl;
    // }
    
   // lines for printing results
   // myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;

  // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
 	  /* if (prevFeatures.size() < MIN_NUM_FEAT)	{
      //cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
      //cout << "trigerring redection" << endl;
 		  featureDetection(prevImage, prevFeatures);
      featureTracking(prevImage,currImage,prevFeatures,currFeatures, status);

 	  } */

    prevImage = currImage.clone();
    //prevFeatures = currFeatures;
    
    t2=clock();//程序段结束后取得系统运行时间(ms)
    //cout << "compute time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;

    t1=clock();//程序段开始前取得系统运行时间(ms)
    int x = int(t_f.at<double>(0)*50) + 300;
    //cout << "int(t_f.at<double>(0)) is :" << int(t_f.at<double>(0)) << endl;
    int y = int(t_f.at<double>(2)*50) + 300;
    circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);

    rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
    putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    //imshow( "Road facing camera", currImage_c );
    imshow( "Trajectory", traj );
    
    Eigen::Matrix3d R_eigen;
    cv2eigen(R_f, R_eigen);
    Eigen::Quaterniond q_f = Eigen::Quaterniond(R_eigen); 
    myfile << time_stamp << " " << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << " ";
    myfile << q_f.x() << " " << q_f.y() << " " << q_f.z() << " " << q_f.w() << endl;
    
    t2=clock();//程序段结束后取得系统运行时间(ms)
    //cout << "output time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;

    waitKey(1);

  }

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Total time taken: " << elapsed_secs << "s" << endl;

  //cout << R_f << endl;
  //cout << t_f << endl;
  
  myfile.close();
  
  device_close(kernel, task);

  return 0;
}