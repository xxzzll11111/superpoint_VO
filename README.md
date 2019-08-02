# superpoint_VO

## 准备工作

### 依赖库

* opencv（包含contrib)
* Eigen3
* g2o
* DNNDK

### 数据集

数据集的文件结构应与TUM的存储结构相同：

```
$DATA_DIR
├─ rgb
│  ├─ *.png
│  └─ …
├─ depth
│  ├─ *.png
│  └─ …
├─ rgb.txt
├─ depth.txt
└─ match.txt
```

其中`match.txt`由TUM官网上的[associate.py](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/associate.py)生成：

```
python associate.py ./rgb.txt ./depth.txt
```

## 简单运行

修改`src/visodo.cpp`文件中的`get_Name_and_Scale()`函数的代码：

```C++
char listFileDir[200] = "/home/linaro/dataset/tum/rgbd_dataset_freiburg2_desk/";
```

替换为你的数据集的地址。修改相机的编号为你的数据集对应的相机：

```C++
#define K_FREIBURG 2
```

运行指令：

```
mkdir build
cd build
cmake .. && make
./main
```

## 修改参数

### `src/visodo.cpp`文件中可修改的参数

* 最大运行帧数(运行到最大帧数或数据集结束都会停止)
```c++
#define MAX_FRAME 5000
```

* 特征点检测方法：orb/sift/superpoint
```c++
#define DETECTOR "superpoint"
```

### `src/vo_features.cpp`文件中可修改的参数

* NMS的阈值
```c++
#define NMS_Threshold 4
```

* 特征点数量
```c++
#define KEEP_K_POINTS 200
```

* descriptor匹配间距的阈值
```c++
#define NN_thresh 0.7
```

* descriptor匹配方法
    * BF：暴力匹配
    * FlannBased：Fast Library forApproximate Nearest Neighbors，一种近似方法
```c++
#define MATCHER "BF"
```