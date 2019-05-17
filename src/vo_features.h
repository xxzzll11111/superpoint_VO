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
#define KEEP_K_POINTS 400
#define NN_thresh 0.7

using namespace cv;
using namespace std;

class point
{
    public:
        int W;   
        int H;  
        float semi;   
        point(int a, int b, float c) {H=a;W=b;semi=c;}
};

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
    assert(task);
    points.resize(0,Point2f(0,0));
    
    int num = dpuGetInputTensorSize(task, INPUT_NODE);
    int8_t* input_img = new int8_t[num]();
    uint8_t* data = (uint8_t*)img.data;
    int input_max = 0;
    int input_min = 128;
    for(int i=0; i<num; i++) {
        input_img[i] = (int8_t)(data[i]/2);
        if (input_max<input_img[i]) input_max=input_img[i];
        if (input_min>input_img[i]) input_min=input_img[i];
    }
    // cout << "input_max:" << input_max << endl;
    // cout << "input_min:" << input_min << endl;
    
    dpuSetInputTensorInHWCInt8(task, INPUT_NODE, (int8_t*)input_img, num);
    
    // cout << "\nRun superpoint ..." << endl;
    dpuRunTask(task);
    
    /* Get DPU execution time (in us) of CONV Task */
    long long timeProf = dpuGetTaskProfile(task);
    //cout << "  DPU CONV Execution time: " << (timeProf * 1.0f) << "us\n";
    
    
    int num_semi = dpuGetOutputTensorSize(task, OUTPUT_NODE_semi);
	float* result_semi = new float[num_semi];
    int num_desc = dpuGetOutputTensorSize(task, OUTPUT_NODE_desc);
	float* result_desc = new float[num_desc];
    
    dpuGetOutputTensorInHWCFP32(task, OUTPUT_NODE_semi, result_semi, num_semi);
    dpuGetOutputTensorInHWCFP32(task, OUTPUT_NODE_desc, result_desc, num_desc);
//    
//    //semi exp
    // cout << "\nRun exp semi ..." << endl;
    for(int i=0; i<num_semi; i++) {
		result_semi[i] = exp(result_semi[i]);
	}
//    
    float semi[Height][Width];
    float coarse_desc[Height/Cell][Width/Cell][D];
    
    // cout << "\nRun normalize ..." << endl;
    for(int i=0; i<Height/Cell; i++) {
        for(int j=0; j<Width/Cell; j++) {
            //semi normalize
            float cell_sum = 0;
            for(int k=0; k<Feature_Length; k++) {
                cell_sum = cell_sum + result_semi[k+j*Feature_Length+i*Feature_Length*Width/Cell];
            }
            for(int kh=0; kh<Cell; kh++) {
                for(int kw=0; kw<Cell; kw++) {
                    semi[kh+i*Cell][kw+j*Cell] = result_semi[kw+kh*Cell+j*Feature_Length+i*Feature_Length*Width/Cell]/cell_sum;
                }
            }
            
            float desc_sum_2 = 0;
            for(int k=0; k<D; k++) {
                desc_sum_2 = desc_sum_2 + pow(result_desc[k+j*D+i*D*Width/Cell],2);
            }
            float desc_sum = sqrt(desc_sum_2);
            for(int k=0; k<D; k++) {
                coarse_desc[i][j][k] = result_desc[k+j*D+i*D*Width/Cell]/desc_sum;
            }
        }
    }
    
    //nms_fast(semi, Height, Width, NMS_Threshold);
    // cout << "\nRun NMS ..." << endl;
    vector<point> tmp_point;
    
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
    
    //rank POINTS
    for(int i=0; i<min( KEEP_K_POINTS, int(tmp_point.size()-1) ); i++) {
        for(int j=tmp_point.size()-1; j>i; j--) {
            if(tmp_point[j].semi>tmp_point[j-1].semi) {
                swap(tmp_point[j], tmp_point[j-1]);
            }
        }
    }
    //KEEP K POINTS
    if(tmp_point.size()>KEEP_K_POINTS) tmp_point.resize(KEEP_K_POINTS, point(0,0,0));
    
    desc.create( int(tmp_point.size()), D, CV_32FC1);
    
    for(int i=0; i<tmp_point.size(); i++) {
        points.push_back(Point2f(tmp_point[i].W, tmp_point[i].H));
        float* pData = desc.ptr<float>(i);   //第i+1行的所有元素  
        for(int j = 0; j < desc.cols; j++)
            pData[j] = coarse_desc[tmp_point[i].H/Cell][tmp_point[i].W/Cell][j];
    }
    
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



void featureTracking_superpoint(DPUTask *task, Mat img_1, Mat img_2, Mat depth_1, Mat K, vector<Point3f>& points1, vector<Point2f>& points2)	{ 

//this function automatically gets rid of points for which tracking fails
    long t1,t2;
    
    Mat desc1;
    Mat desc2;
    vector<Point2f>points1_tmp, points2_tmp;
    points1.resize(0,Point3f(0,0,0));
    points2.resize(0,Point2f(0,0));
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    run_superpoint(task, img_1, points1_tmp, desc1);
    run_superpoint(task, img_2, points2_tmp, desc2);
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "   run_superpoint time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;//0.281702s
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    vector<DMatch> matches;
    BFMatcher matcher(NORM_L2, true);
    matcher.match(desc1, desc2, matches);
    // cout <<  "matches size:" << matches.size() << endl;
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "   BFmatch time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    vector <Point2f> RAN_KP1, RAN_KP2;
    for(int i=0; i<matches.size(); i++) {
        if ( matches[i].distance>NN_thresh )
            continue;
        if ( abs(points1_tmp[matches[i].queryIdx].x-points2_tmp[matches[i].trainIdx].x)> Width/10 )
            continue;
        if ( abs(points1_tmp[matches[i].queryIdx].y-points2_tmp[matches[i].trainIdx].y)> Height/10 )
            continue;
        RAN_KP1.push_back(points1_tmp[matches[i].queryIdx]);
        RAN_KP2.push_back(points2_tmp[matches[i].trainIdx]);
        //RAN_KP1是要存储img01中能与img02匹配的点
    }
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "   thresh check time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    //cout <<  "RAN_KP1 size:" << RAN_KP1.size() << endl;
    vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(RAN_KP1, RAN_KP2, RansacStatus, FM_RANSAC);
	//重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
	vector <Point2f> RR_KP1, RR_KP2;
	for (size_t i = 0; i < RansacStatus.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			RR_KP1.push_back(RAN_KP1[i]);
			RR_KP2.push_back(RAN_KP2[i]);
		}
	}
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "   RANSAC time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    // cout << "RR_KP1 size:" << RR_KP1.size() << endl;
    for ( int i=0; i<RR_KP1.size(); i++ )
    {
        ushort d = depth_1.ptr<unsigned short> (int ( RR_KP1[i].y )) [ int ( RR_KP1[i].x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/5000.0;
        Point2d p1 = pixel2cam ( RR_KP1[i], K );
        points1.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
        points2.push_back ( RR_KP2[i] );
        
        line(img_2, RR_KP1[i], RR_KP2[i], Scalar(0, 0, 255));
        circle(img_2, RR_KP2[i], 4, cv::Scalar(0, 0, 255));
    }

    // cout<<"3d-2d pairs: "<<points1.size() <<endl;
    imshow( "Road facing camera", img_2 );
    
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "   2d_to_3d time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
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

void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<Point2f>& RAN_KP1,
                            std::vector<Point2f>& RAN_KP2 )
{
    //-- 初始化
    vector<KeyPoint> keypoint1, keypoint2;
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create( 2000 );
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoint1 );
    detector->detect ( img_2,keypoint2 );
    //NMS
    selectMax(NMS_Threshold, keypoint1);
    selectMax(NMS_Threshold, keypoint2);
    cout << "keypoints_1 size:" << keypoint1.size() << endl;
    cout << "keypoints_2 size:" << keypoint2.size() << endl;

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoint1, descriptors_1 );
    descriptor->compute ( img_2, keypoint2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, matches );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for(int i=0; i<matches.size(); i++) {
     
        if ( matches[i].distance> max ( 2*min_dist, 30.0 ) )
            continue;
        if ( abs(keypoint1[matches[i].queryIdx].pt.x-keypoint2[matches[i].trainIdx].pt.x)> Width/10 )
            continue;
        if ( abs(keypoint1[matches[i].queryIdx].pt.y-keypoint2[matches[i].trainIdx].pt.y)> Height/10 )
            continue;
        RAN_KP1.push_back(keypoint1[matches[i].queryIdx].pt);
        RAN_KP2.push_back(keypoint2[matches[i].trainIdx].pt);
        //RAN_KP1是要存储img01中能与img02匹配的点
    }
}

void featureTracking_ORB(Mat img_1, Mat img_2, Mat depth_1, Mat K, vector<Point3f>& points1, vector<Point2f>& points2)	
{ 
    points1.resize(0,Point3f(0,0,0));
    points2.resize(0,Point2f(0,0));
    vector <Point2f> RAN_KP1, RAN_KP2;
    vector<DMatch> matches;
    find_feature_matches ( img_1, img_2, RAN_KP1, RAN_KP2 );
    cout<<"match pairs: "<<matches.size() <<endl;
    
    //cout <<  "RAN_KP1 size:" << RAN_KP1.size() << endl;
    vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(RAN_KP1, RAN_KP2, RansacStatus, FM_RANSAC);
	//重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
	vector <Point2f> RR_KP1, RR_KP2;
	for (size_t i = 0; i < RansacStatus.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			RR_KP1.push_back(RAN_KP1[i]);
			RR_KP2.push_back(RAN_KP2[i]);
		}
	}
    
    cout << "RR_KP1 size:" << RR_KP1.size() << endl;
    for ( int i=0; i<RR_KP1.size(); i++ )
    {
        ushort d = depth_1.ptr<unsigned short> (int ( RR_KP1[i].y )) [ int ( RR_KP1[i].x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/5000.0;
        Point2d p1 = pixel2cam ( RR_KP1[i], K );
        points1.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
        points2.push_back ( RR_KP2[i] );
        
        line(img_2, RR_KP1[i], RR_KP2[i], Scalar(0, 0, 255));
        circle(img_2, RR_KP2[i], 4, cv::Scalar(0, 0, 255));
    }

    cout<<"3d-2d pairs: "<<points1.size() <<endl;
    imshow( "Road facing camera", img_2 );
}

void featureTracking_sift(Mat img_1, Mat img_2, Mat depth_1, Mat K, vector<Point3f>& points1, vector<Point2f>& points2)	{ 

//this function automatically gets rid of points for which tracking fails
    long t1,t2;
    Mat desc1;
    Mat desc2;
    vector<KeyPoint> keypoint1,keypoint2;
    points1.resize(0,Point3f(0,0,0));
    points2.resize(0,Point2f(0,0));
    Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create( KEEP_K_POINTS,  3, 0.04, 3, 1.6 );
    Ptr<DescriptorExtractor> descriptor = xfeatures2d::SIFT::create();
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    detector->detect(img_1,keypoint1);
    detector->detect(img_2,keypoint2);
    // selectMax(NMS_Threshold, keypoint1);
    // selectMax(NMS_Threshold, keypoint2);
    cout << "keypoints_1 size:" << keypoint1.size() << endl;
    cout << "keypoints_2 size:" << keypoint2.size() << endl;
    
    descriptor->compute(img_1,keypoint1,desc1);
    descriptor->compute(img_2,keypoint2,desc2);
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "run_superpoint time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
    
    t1=clock();//程序段开始前取得系统运行时间(ms)
    vector<DMatch> matches;
    BFMatcher matcher(NORM_L2, true);
    matcher.match(desc1, desc2, matches);
    cout <<  "matches size:" << matches.size() << endl;
    
    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < matches.size(); i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );
    
    //cout << "RAN_KP" << endl;
    vector <Point2f> RAN_KP1, RAN_KP2;
    for(int i=0; i<matches.size(); i++) {
        if ( matches[i].distance> 200 )
            continue;
        if ( abs(keypoint1[matches[i].queryIdx].pt.x-keypoint2[matches[i].trainIdx].pt.x)> Width/10 )
            continue;
        if ( abs(keypoint1[matches[i].queryIdx].pt.y-keypoint2[matches[i].trainIdx].pt.y)> Height/10 )
            continue;
        RAN_KP1.push_back(keypoint1[matches[i].queryIdx].pt);
        RAN_KP2.push_back(keypoint2[matches[i].trainIdx].pt);
        //RAN_KP1是要存储img01中能与img02匹配的点
    }
    
    //cout <<  "RAN_KP1 size:" << RAN_KP1.size() << endl;
    vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(RAN_KP1, RAN_KP2, RansacStatus, FM_RANSAC);
	//重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
	vector <Point2f> RR_KP1, RR_KP2;
	for (size_t i = 0; i < RansacStatus.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			RR_KP1.push_back(RAN_KP1[i]);
			RR_KP2.push_back(RAN_KP2[i]);
		}
	}
    
    cout << "RR_KP1 size:" << RR_KP1.size() << endl;
    for ( int i=0; i<RR_KP1.size(); i++ )
    {
        ushort d = depth_1.ptr<unsigned short> (int ( RR_KP1[i].y )) [ int ( RR_KP1[i].x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/5000.0;
        Point2d p1 = pixel2cam ( RR_KP1[i], K );
        points1.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
        points2.push_back ( RR_KP2[i] );
        
        line(img_2, RR_KP1[i], RR_KP2[i], Scalar(0, 0, 255));
        circle(img_2, RR_KP2[i], 4, cv::Scalar(0, 0, 255));
    }

    cout<<"3d-2d pairs: "<<points1.size() <<endl;
    imshow( "Road facing camera", img_2 );
    
    t2=clock();//程序段结束后取得系统运行时间(ms)
    cout << "match time:" << float(t2-t1)/CLOCKS_PER_SEC << "s" << endl;
}