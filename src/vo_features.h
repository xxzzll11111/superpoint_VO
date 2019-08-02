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
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <typeinfo>

#include "top_k.h"

/* header file for DNNDK APIs */
#include <dnndk/dnndk.h>

#define INPUT_NODE "ConvNdBackward1"
#define OUTPUT_NODE_semi "ConvNdBackward22"
#define OUTPUT_NODE_desc "ConvNdBackward25"
#define Width 640
#define Height 480
#define Cell 8
#define Feature_Length 65
#define NMS_Threshold 4
#define D 256
#define KEEP_K_POINTS 200
#define NN_thresh 0.7
#define MATCHER "BF"
#define CONF_thresh 0.15

using namespace cv;
using namespace std;


void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{ 

//this function automatically gets rid of points for which tracking fails

  vector<float> err;					
  Size winSize=Size(21,21);																								
  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  Point2f pt = points2.at(i- indexCorrection);
     	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
     		  if((pt.x<0)||(pt.y<0))	{
     		  	status.at(i) = 0;
     		  }
     		  points1.erase (points1.begin() + (i - indexCorrection));
     		  points2.erase (points2.begin() + (i - indexCorrection));
     		  indexCorrection++;
     	}

     }

}


void featureDetection(Mat img_1, vector<Point2f>& points1)	{   //uses FAST as of now, modify parameters as necessary
  vector<KeyPoint> keypoints_1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
  KeyPoint::convert(keypoints_1, points1, vector<int>());
}


void device_setup(DPUKernel *&kernel, DPUTask *&task)
{
    // Attach to DPU driver and prepare for running
    dpuOpen();

    // Load DPU Kernel for DenseBox neural network
    kernel = dpuLoadKernel("superpoint");


	task = dpuCreateTask(kernel, 0);
}

void device_close(DPUKernel *kernel, DPUTask *task)
{
    dpuDestroyTask(task);

    // Destroy DPU Kernel & free resources
    dpuDestroyKernel(kernel);

    // Dettach from DPU driver & release resources
    dpuClose();
}

void run_superpoint(DPUTask *task, Mat img, vector<Point2f>& points, Mat& desc)
{
    long t1,t2;
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    assert(task);
    points.resize(0,Point2f(0,0));
    
    int num = dpuGetInputTensorSize(task, INPUT_NODE);
    int8_t* input_img = new int8_t[num]();
    uint8_t* data = (uint8_t*)img.data;
    // int input_max = 0;
    // int input_min = 128;
    for(int i=0; i<num; i++) {
        input_img[i] = (int8_t)(data[i]/2);
        // if (input_max<input_img[i]) input_max=input_img[i];
        // if (input_min>input_img[i]) input_min=input_img[i];
    }
    // cout << "input_max:" << input_max << endl;
    // cout << "input_min:" << input_min << endl;
    
    dpuSetInputTensorInHWCInt8(task, INPUT_NODE, (int8_t*)input_img, num);
    
    // cout << "\nRun superpoint ..." << endl;
    dpuRunTask(task);
    
    /* Get DPU execution time (in us) of CONV Task */
    // long long timeProf = dpuGetTaskProfile(task);
    //cout << "  DPU CONV Execution time: " << (timeProf * 1.0f) << "us\n";
    
    int num_semi = dpuGetOutputTensorSize(task, OUTPUT_NODE_semi);
	float* result_semi = new float[num_semi];
    int num_desc = dpuGetOutputTensorSize(task, OUTPUT_NODE_desc);
	float* result_desc = new float[num_desc];
    
    dpuGetOutputTensorInHWCFP32(task, OUTPUT_NODE_semi, result_semi, num_semi);
    dpuGetOutputTensorInHWCFP32(task, OUTPUT_NODE_desc, result_desc, num_desc);
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "       DPU time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    //semi exp
    // cout << "\nRun exp semi ..." << endl;
    for(int i=0; i<num_semi; i++) {
		result_semi[i] = exp(result_semi[i]); //e^x
        // result_semi[i] = pow(2, result_semi[i]); //2^x
        // result_semi[i] = pow(4, result_semi[i]); //4^x
	}
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "       exp time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
    
    t1=clock();//程序段开始前取得系统运行时间(ms)  
    float semi[Height][Width];
    point coarse_semi[Height/Cell][Width/Cell];
    float coarse_desc[Height/Cell][Width/Cell][D];
    
    // cout << "\nRun normalize ..." << endl;
    for(int i=0; i<Height/Cell; i++) {
        for(int j=0; j<Width/Cell; j++) {
            //semi softmax
            float cell_sum = 0;
            for(int k=0; k<Feature_Length; k++) {
                cell_sum = cell_sum + result_semi[k+j*Feature_Length+i*Feature_Length*Width/Cell];
            }
            for(int kh=0; kh<Cell; kh++) {
                for(int kw=0; kw<Cell; kw++) {
                    semi[kh+i*Cell][kw+j*Cell] = result_semi[kw+kh*Cell+j*Feature_Length+i*Feature_Length*Width/Cell]/cell_sum;
                }
            }
            
            //max 1 point
            /* float max_semi=0;
            for(int kh=0; kh<Cell; kh++) {
                for(int kw=0; kw<Cell; kw++) {
                    if(semi[kh+i*Cell][kw+j*Cell] > max_semi) {
                        max_semi = semi[kh+i*Cell][kw+j*Cell];
                        coarse_semi[i][j].H = kh+i*Cell;
                        coarse_semi[i][j].W = kw+j*Cell;
                        coarse_semi[i][j].semi = max_semi;
                    }
                }
            } */
            
            //desc normalize
            float desc_sum_2 = 0;
            for(int k=0; k<D; k++) {
                desc_sum_2 = desc_sum_2 + pow(result_desc[k+j*D+i*D*Width/Cell],2);
            }
            float desc_sum = sqrt(desc_sum_2);
            for(int k=0; k<D; k++) {
                coarse_desc[i][j][k] = result_desc[k+j*D+i*D*Width/Cell]/desc_sum;
                // coarse_desc[i][j][k] = (float)(int)(result_desc[k+j*D+i*D*Width/Cell]/desc_sum*512);
                // coarse_desc[i][j][k] = coarse_desc[i][j][k]>127? 127:coarse_desc[i][j][k];
                // coarse_desc[i][j][k] = coarse_desc[i][j][k]<-128? -128:coarse_desc[i][j][k];
            }
        }
    }
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "       normalize time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    //nms_fast(semi, Height, Width, NMS_Threshold);
    // cout << "\nRun NMS ..." << endl;
    vector<point> tmp_point;
    
    //NMS
    for(int i=0; i<Height; i++) {
        for(int j=0; j<Width; j++) {
            if(semi[i][j] != 0) {
                float tmp_semi = semi[i][j];
                for(int kh=max(0,i-NMS_Threshold); kh<min(Height,i+NMS_Threshold+1); kh++)
                    for(int kw=max(0,j-NMS_Threshold); kw<min(Width,j+NMS_Threshold+1); kw++)
                        if(i!=kh||j!=kw) {
                            if(tmp_semi>=semi[kh][kw])
                                semi[kh][kw] = 0;
                            else
                                semi[i][j] = 0;
                        }
                if(semi[i][j]!=0)
                    tmp_point.push_back(point(i,j,semi[i][j]));
            }
        }
    }
    
    //coarse NMS
    /* for(int i=0; i<Height/Cell; i++) {
        for(int j=0; j<Width/Cell; j++) {
            if(coarse_semi[i][j].semi != 0) {
                float tmp_semi = coarse_semi[i][j].semi;
                for(int kh=max(0,i-1); kh<min(Height/Cell,i+1+1); kh++)
                    for(int kw=max(0,j-1); kw<min(Width/Cell,j+1+1); kw++)
                        if(i!=kh||j!=kw) {
                            if(abs(coarse_semi[i][j].H-coarse_semi[kh][kw].H)<=NMS_Threshold && abs(coarse_semi[i][j].W-coarse_semi[kh][kw].W)<=NMS_Threshold) {
                                if(tmp_semi>=coarse_semi[kh][kw].semi)
                                    coarse_semi[kh][kw].semi = 0;
                                else
                                    coarse_semi[i][j].semi = 0;
                            }
                        }
                if(coarse_semi[i][j].semi!=0)
                    tmp_point.push_back(coarse_semi[i][j]);
            }
        }
    } */
    cout<<"tmp_point.size:"<<tmp_point.size()<<endl;
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "       NMS time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    //rank POINTS
    /* for(int i=0; i<min( KEEP_K_POINTS, int(tmp_point.size()-1) ); i++) {
        for(int j=tmp_point.size()-1; j>i; j--) {
            if(tmp_point[j].semi>tmp_point[j-1].semi) {
                swap(tmp_point[j], tmp_point[j-1]);
            }
        }
    }
    //KEEP K POINTS
    if(tmp_point.size()>KEEP_K_POINTS) tmp_point.resize(KEEP_K_POINTS, point(0,0,0)); */
    
    top_k(tmp_point,tmp_point.size(),KEEP_K_POINTS);
    // top_k_with_NMS(tmp_point,tmp_point.size(),KEEP_K_POINTS,NMS_Threshold);
    
    
    //CONF_thresh
    // for(int i=0; i<tmp_point.size(); i++) {
        // if(tmp_point[i].semi < CONF_thresh) {
            // tmp_point.erase(tmp_point.begin()+i);
            // i--;
        // }
    // }
    
    double min_conf=10000, max_conf=0;
    for ( int i = 0; i < tmp_point.size(); i++ )
    {
        double conf = tmp_point[i].semi;
        if ( conf < min_conf ) min_conf = conf;
        if ( conf > max_conf ) max_conf = conf;
    }
    printf ( "-- Max conf : %f \n", max_conf );
    printf ( "-- Min conf : %f \n", min_conf );
    
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "       rank time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    desc.create( int(tmp_point.size()), D, CV_32FC1);
    
    for(int i=0; i<tmp_point.size(); i++) {
        points.push_back(Point2f(tmp_point[i].W, tmp_point[i].H));
        float* pData = desc.ptr<float>(i);   //第i+1行的所有元素  
        for(int j = 0; j < desc.cols; j++)
            pData[j] = coarse_desc[tmp_point[i].H/Cell][tmp_point[i].W/Cell][j];
    }
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "       output time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
    
    delete[] input_img;
    delete[] result_semi;
    delete[] result_desc;
}


Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}



void featureTracking_superpoint(DPUTask *task, vector<Point2f>& points1, Mat& desc1, Mat& img_2, Mat depth_1, Mat K, vector<Point3f>& points_3d, vector<Point2f>& points_2d)	{ 

//this function automatically gets rid of points for which tracking fails
    long t1,t2;
    vector<Point2f>points2;
    Mat desc2;
    points_3d.resize(0,Point3f(0,0,0));
    points_2d.resize(0,Point2f(0,0));
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    run_superpoint(task, img_2, points2, desc2);
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "   run_superpoint time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    vector<DMatch> matches;
    if( MATCHER == "BF" ) {
        BFMatcher matcher(NORM_L2, true);
        matcher.match(desc1, desc2, matches);
    }
    else {
        FlannBasedMatcher matcher;
        matcher.match(desc1, desc2, matches);
    }
    cout <<  "desc1 size:" << desc1.size() << endl;
    cout <<  "matches size:" << matches.size() << endl;
    // cout <<  "matches[0].distance:" << matches[0].distance << endl;
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "   match time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;//0.231208s
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    
    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;
    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < desc1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }
    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );
    
    vector <Point2f> RAN_KP1, RAN_KP2;
    for(int i=0; i<matches.size(); i++) {
        if ( matches[i].distance>NN_thresh )//360)//
            continue;
        if ( abs(points1[matches[i].queryIdx].x-points2[matches[i].trainIdx].x)> Width/10 )
            continue;
        if ( abs(points1[matches[i].queryIdx].y-points2[matches[i].trainIdx].y)> Height/10 )
            continue;
        RAN_KP1.push_back(points1[matches[i].queryIdx]);
        RAN_KP2.push_back(points2[matches[i].trainIdx]);
        //RAN_KP1是要存储img01中能与img02匹配的点
    }
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "   thresh check time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;//5.3e-05s
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    cout <<  "RAN_KP1 size:" << RAN_KP1.size() << endl;
    vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(RAN_KP1, RAN_KP2, RansacStatus, FM_RANSAC);
	//通过RansacStatus来删除误匹配点
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "   RANSAC time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;//0.001139s
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    for ( int i=0; i<RAN_KP1.size(); i++ )
    {
        //cout<<i<<endl;
        if (RansacStatus[i] == 0)
            continue;
        ushort d = depth_1.ptr<unsigned short> (int ( RAN_KP1[i].y )) [ int ( RAN_KP1[i].x ) ];
        // uint8_t d = depth_1.ptr<uint8_t> (int ( RAN_KP1[i].y )) [ int ( RAN_KP1[i].x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/5000.0;
        Point2d p1 = pixel2cam ( RAN_KP1[i], K );
        points_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
        points_2d.push_back ( RAN_KP2[i] );
        
        line(img_2, RAN_KP1[i], RAN_KP2[i], Scalar(0, 0, 255));
        circle(img_2, RAN_KP2[i], 4, cv::Scalar(0, 0, 255));
    }

    cout<<"3d-2d pairs: "<<points_3d.size() <<endl;
        
    //assert( points_3d.size() >= 50 );
    imshow( "Road facing camera", img_2 );
    
    points1 = points2;
    desc1 = desc2;
    
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "   2d_to_3d time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;//0.000659s
}


//局部极大值抑制，这里利用fast特征点的响应值做比较
void selectMax(int r, std::vector<KeyPoint> & kp){

    //r是局部极大值抑制的窗口半径
    if (r != 0){
        //对kp中的点进行局部极大值筛选
        for (int i = 0; i < kp.size(); i++){
            for (int j = i + 1; j < kp.size(); j++){
                //如果两个点的距离小于半径r，则删除其中响应值较小的点
                if (abs(kp[i].pt.x - kp[j].pt.x)<=r && abs(kp[i].pt.y - kp[j].pt.y)<=r){
                    if (kp[i].response < kp[j].response){
                        std::vector<KeyPoint>::iterator it = kp.begin() + i;
                        kp.erase(it);
                        i--;
                        break;
                    }
                    else{
                        std::vector<KeyPoint>::iterator it = kp.begin() + j;
                        kp.erase(it);
                        j--;
                    }
                }
            }
        }
    }

}

void run_orb ( const Mat& img, vector<Point2f>& point, Mat& descriptors )
{
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create( 2000 );
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    
    //-- 第一步:检测 Oriented FAST 角点位置
    vector<KeyPoint> keypoint;
    detector->detect ( img,keypoint );
    //NMS
    selectMax(NMS_Threshold, keypoint);
    cout << "keypoints size:" << keypoint.size() << endl;
    
    for(int i=0; i<min( KEEP_K_POINTS, int(keypoint.size()-1) ); i++) {
        for(int j=keypoint.size()-1; j>i; j--) {
            if(keypoint[j].response>keypoint[j-1].response) {
                swap(keypoint[j], keypoint[j-1]);
            }
        }
    }
    if(keypoint.size()>KEEP_K_POINTS) keypoint.resize(KEEP_K_POINTS, KeyPoint());

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img, keypoint, descriptors );
    
    for(int i=0; i<keypoint.size(); i++) {
        point.push_back(keypoint[i].pt);
    }
}

void featureTracking_ORB(vector<Point2f>& keypoint1, Mat& desc1, Mat img_2, Mat depth_1, Mat K, vector<Point3f>& points_3d, vector<Point2f>& points_2d)	

{ 
    points_3d.resize(0,Point3f(0,0,0));
    points_2d.resize(0,Point2f(0,0));
    vector<Point2f>keypoint2;
    Mat desc2;
    
    run_orb(img_2, keypoint2, desc2);
    
    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    // BFMatcher matcher ( NORM_HAMMING );
    if( MATCHER == "BF" ) {
        Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
        matcher->match ( desc1, desc2, matches );
    }
    else {
         // the descriptor for FlannBasedMatcher should has matrix element of CV_32F
        if( desc1.type()!=CV_32F ) 
            desc1.convertTo( desc1, CV_32F );
        if( desc2.type()!=CV_32F ) 
            desc2.convertTo( desc2, CV_32F );
        Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "FlannBased" );
        matcher->match ( desc1, desc2, matches );
    }

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < desc1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    vector <Point2f> RAN_KP1, RAN_KP2;
    for(int i=0; i<matches.size(); i++) {
        if ( matches[i].distance> max ( 2*min_dist, 300.0 ) )
            continue;
        if ( abs(keypoint1[matches[i].queryIdx].x-keypoint2[matches[i].trainIdx].x)> Width/10 )
            continue;
        if ( abs(keypoint1[matches[i].queryIdx].y-keypoint2[matches[i].trainIdx].y)> Height/10 )
            continue;
        RAN_KP1.push_back(keypoint1[matches[i].queryIdx]);
        RAN_KP2.push_back(keypoint2[matches[i].trainIdx]);
        //RAN_KP1是要存储img01中能与img02匹配的点
    }
    
    //cout <<  "RAN_KP1 size:" << RAN_KP1.size() << endl;
    vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(RAN_KP1, RAN_KP2, RansacStatus, FM_RANSAC);
	//重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
    
    cout << "RAN_KP1 size:" << RAN_KP1.size() << endl;
    for ( int i=0; i<RAN_KP1.size(); i++ )
    {
        if (RansacStatus[i] == 0)
            continue;
        ushort d = depth_1.ptr<unsigned short> (int ( RAN_KP1[i].y )) [ int ( RAN_KP1[i].x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/5000.0;
        Point2d p1 = pixel2cam ( RAN_KP1[i], K );
        points_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
        points_2d.push_back ( RAN_KP2[i] );
        
        line(img_2, RAN_KP1[i], RAN_KP2[i], Scalar(0, 0, 255));
        circle(img_2, RAN_KP2[i], 4, cv::Scalar(0, 0, 255));
    }

    cout<<"3d-2d pairs: "<<points_3d.size() <<endl;
    imshow( "Road facing camera", img_2 );
    
    keypoint1 = keypoint2;
    //cout << "keypoints size:" << keypoint1.size() << endl;
    desc1 = desc2;
}


void run_sift ( const Mat& img, vector<Point2f>& point, Mat& descriptors )
{
    // used in OpenCV3
    Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create( KEEP_K_POINTS,  3, 0.04, 3, 1.6 );
    Ptr<DescriptorExtractor> descriptor = xfeatures2d::SIFT::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    
    //-- 第一步:检测 keypoint 角点位置
    vector<KeyPoint> keypoint;
    detector->detect ( img,keypoint );
    //NMS
    // selectMax(NMS_Threshold, keypoint);
    //cout << "keypoints size:" << keypoint.size() << endl;

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img, keypoint, descriptors );
    
    for(int i=0; i<keypoint.size(); i++) {
        point.push_back(keypoint[i].pt);
    }
}

void featureTracking_sift(vector<Point2f>& keypoint1, Mat& desc1, Mat img_2, Mat depth_1, Mat K, vector<Point3f>& points_3d, vector<Point2f>& points_2d)	{ 

//this function automatically gets rid of points for which tracking fails
    long t1,t2;
    vector<Point2f>keypoint2;
    Mat desc2;
    points_3d.resize(0,Point3f(0,0,0));
    points_2d.resize(0,Point2f(0,0));
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    run_sift(img_2, keypoint2, desc2);
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "   run_sift time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    vector<DMatch> matches;
    if( MATCHER == "BF" ) {
        BFMatcher matcher(NORM_L2, true);
        matcher.match(desc1, desc2, matches);
    }
    else {
        FlannBasedMatcher matcher;
        matcher.match(desc1, desc2, matches);
    }
    //cout <<  "desc1 size:" << desc1.size() << endl;
    //cout <<  "matches size:" << matches.size() << endl;
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "   BFmatch time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;//0.231208s
    
    //cout << "RAN_KP" << endl;
    vector <Point2f> RAN_KP1, RAN_KP2;
    for(int i=0; i<matches.size(); i++) {
        if ( matches[i].distance> 200 )
            continue;
        if ( abs(keypoint1[matches[i].queryIdx].x-keypoint2[matches[i].trainIdx].x)> Width/10 )
            continue;
        if ( abs(keypoint1[matches[i].queryIdx].y-keypoint2[matches[i].trainIdx].y)> Height/10 )
            continue;
        RAN_KP1.push_back(keypoint1[matches[i].queryIdx]);
        RAN_KP2.push_back(keypoint2[matches[i].trainIdx]);
        //RAN_KP1是要存储img01中能与img02匹配的点
    }
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    //cout <<  "RAN_KP1 size:" << RAN_KP1.size() << endl;
    vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(RAN_KP1, RAN_KP2, RansacStatus, FM_RANSAC);
	//通过RansacStatus来删除误匹配点
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "   RANSAC time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;//0.001139s
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    for ( int i=0; i<RAN_KP1.size(); i++ )
    {
        if (RansacStatus[i] == 0)
            continue;
        ushort d = depth_1.ptr<unsigned short> (int ( RAN_KP1[i].y )) [ int ( RAN_KP1[i].x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/5000.0;
        Point2d p1 = pixel2cam ( RAN_KP1[i], K );
        points_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
        points_2d.push_back ( RAN_KP2[i] );
        
        line(img_2, RAN_KP1[i], RAN_KP2[i], Scalar(0, 0, 255));
        circle(img_2, RAN_KP2[i], 4, cv::Scalar(0, 0, 255));
    }

    // cout<<"3d-2d pairs: "<<points_3d.size() <<endl;
    imshow( "Road facing camera", img_2 );
    
    keypoint1 = keypoint2;
    desc1 = desc2;
    
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "match time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
}