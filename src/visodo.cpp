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

//ZU9

#include <stdlib.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include "vo_features.h"



using namespace cv;
using namespace std;

#define MAX_FRAME 2894
#define MIN_NUM_FEAT 2000
#define PNPRANSAC false
#define K_FREIBURG 2
#define DETECTOR "superpoint"


// IMP: Change the file directories (4 places) according to where your dataset is saved before running!

bool get_Name_and_Scale(int frame_id, char* rgbname, char* depthname, char* time)	{
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
  return i>frame_id;
}

/* void bundleAdjustment (
    const vector< Point3f > points_3d,
    const vector< Point2f > points_2d,
    const Mat& K,
    Mat& R, Mat& t )
{
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat <<
          R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
               R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
               R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
                            R_mat,
                            Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                        ) );
    optimizer.addVertex ( pose );

    int index = 1;
    for ( const Point3f p:points_3d )   // landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId ( index++ );
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized ( true ); // g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex ( point );
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters (
        K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    // edges
    index = 1;
    for ( const Point2f p:points_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId ( index );
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
        edge->setVertex ( 1, pose );
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId ( 0,0 );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization();
    optimizer.optimize ( 100 );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
    
    Mat T;
    eigen2cv(Eigen::Isometry3d ( pose->estimate() ).matrix(), T);
    R = T.rowRange(0,3).colRange(0,3);
    t = T.rowRange(0,3).col(3);
} */

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
  
  myfile << time_stamp << " " << 0 << " " << 0 << " " << 0 << " ";
  myfile << 0 << " " << 0 << " " << 0 << " " << 1 << endl;
  
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

  Mat K;
  if ( K_FREIBURG == 1 )
    K = ( Mat_<double> ( 3,3 ) << 517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1 );//freiburg1
  else if  ( K_FREIBURG == 2 )
    K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );//freiburg2
  else
    K = ( Mat_<double> ( 3,3 ) << 535.4, 0, 320.1, 0, 539.2, 247.6, 0, 0, 1 );//freiburg3
  
  if(DETECTOR=="orb")
    featureTracking_ORB(img_1,img_2,d1,K,points1,points2); //track those features to img_2
  else if(DETECTOR=="sift")
    featureTracking_sift(img_1,img_2,d1,K,points1,points2);
  else
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
  //bundleAdjustment ( points1, points2, K, R, t );
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
  
  Eigen::Matrix3d R_eigen;
  cv2eigen(R_f, R_eigen);
  Eigen::Quaterniond q_f = Eigen::Quaterniond(R_eigen); 
  myfile << time_stamp << " " << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << " ";
  myfile << q_f.x() << " " << q_f.y() << " " << q_f.z() << " " << q_f.w() << endl;

  clock_t begin = clock();

  namedWindow( "Road facing camera", WINDOW_AUTOSIZE );// Create a window for display.
  namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

  Mat traj = Mat::zeros(600, 600, CV_8UC3);

  for(int numFrame=2; numFrame < MAX_FRAME; numFrame++)	{
  	t1=clock();//程序段开始前取得系统运行时间(ms)

    //sprintf(filename, "/home/share/kitti_odometry/dataset/sequences/00/image_0/%06d.png", numFrame);
    dpic = imread ( depthname2, CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    
    if(!get_Name_and_Scale(numFrame, filename, depthname2, time_stamp))
        break;

    //cout << "Scale is " << scale << endl;
    
    cout << numFrame << endl;
  	Mat currImage_c = imread(filename);
  	
    cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
  	// vector<uchar> status;
  	
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "read picture time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;//0.046013s

  	t1=clock();//程序段开始前取得系统运行时间(ms)
    
    if(DETECTOR=="orb")
        featureTracking_ORB(prevImage, currImage, dpic, K, prevFeatures, currFeatures);
    else if(DETECTOR=="sift")
        featureTracking_sift(prevImage, currImage, dpic, K, prevFeatures, currFeatures);
    else
        featureTracking_superpoint(task, prevImage, currImage, dpic, K, prevFeatures, currFeatures);
    
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "featureTracking time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;//0.513374s

  	t1=clock();//程序段开始前取得系统运行时间(ms)
    /* E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
  	recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask); */

    if ( PNPRANSAC )
        solvePnPRansac ( prevFeatures, currFeatures, K, Mat(), r, t, false, 100, 8.0, 0.99, noArray(), cv::SOLVEPNP_EPNP ); 
    else
        solvePnP ( prevFeatures, currFeatures, K, Mat(), r, t, false, cv::SOLVEPNP_EPNP ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    //bundleAdjustment ( prevFeatures, currFeatures, K, R, t );
    R = R.t(); 
    t = -R * t;
    // cout<<"R="<<R<<endl;
    // cout<<"t="<<t<<endl;
    
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
    cout << "compute time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;//0.000971s

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
    
    cv2eigen(R_f, R_eigen);
    q_f = Eigen::Quaterniond(R_eigen); 
    myfile << time_stamp << " " << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << " ";
    myfile << q_f.x() << " " << q_f.y() << " " << q_f.z() << " " << q_f.w() << endl;
    
    cv::waitKey(1); 
    
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "output time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;//0.008223s

    
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