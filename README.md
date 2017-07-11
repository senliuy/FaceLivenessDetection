
## 人脸活体检测

活体检测代码是基于面部光流特征和面部68个关键点的数学特征，使用了四层验证关卡来保证登录用户的活体性。最后，用户的三张正脸的活体照片将被保存下来上传到人脸验证服务器用于验证人脸的合法性。

活体检测是基于下面两个开源工具。其中，OpenFace的CLNF算法用于人脸关键点匹配，SeetaFace进行更为实时的人脸检测和跟踪
* [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)
* [SeetaFace](https://github.com/seetaface/SeetaFaceEngine)

活体验证关卡包括：
* 基于面部材质的光流特征
* 基于面部旋转时关键点的欧氏距离特征
* 基于活体的嘴部可动性特征
* 基于活体的眼部可动性特征

## 安装

### MAC

#### 准备

* 首先安装[Homebrew](https://brew.sh)，homebrew是类似于Linux系统上apt-get的软件。
* 安装TBB，OpenCV3（必须3及以上版本）和 boost

    brew install boost tbb opencv3

#### Build

在命令行执行下列指令：

    mkdir build
    cd build 
    cmake -D CMAKE_BUILD_TYPE=RELEASE ..
    make

#### 测试

进入到<code>build/bin</code>目录下。执行

    ./FaceLivenessDetection

### Ubuntu 14.0

Get newest GCC, done using:

    sudo apt-get update

    sudo apt-get install build-essential

Cmake:

    sudo apt-get install cmake

Get BLAS (for dlib)

    sudo apt-get install libopenblas-dev liblapack-dev

OpenCV 3.1.0

4.1 Install OpenCV dependencies:

    sudo apt-get install git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

    sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev checkinstall
4.2 Download OpenCV 3.1.0 from https://github.com/Itseez/opencv/archive/3.1.0.zip

    wget https://github.com/Itseez/opencv/archive/3.1.0.zip
4.3 Unzip it and create a build folder:

    sudo unzip 3.1.0.zip
    cd opencv-3.1.0
    mkdir build
    cd build
4.4 Build it using:

    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_SHARED_LIBS=OFF ..
    make -j2

    sudo make install

Get Boost:

    sudo apt-get install libboost1.55-all-dev

alternatively: 

    sudo apt-get install libboost-all-dev
## 功能
