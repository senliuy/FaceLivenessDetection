
## 人脸活体检测

活体检测代码是基于面部光流特征和面部68个关键点的数学特征，使用了四层验证关卡来保证登录用户的活体性。最后，用户的三张正脸的活体照片将被保存下来上传到人脸验证服务器用于验证人脸的合法性。

活体检测是基于下面两个开源工具。其中，OpenFace的CLNF算法用于人脸关键点匹配，SeetaFace进行更为实时的人脸检测和跟踪
* OpenFace: <https://github.com/TadasBaltrusaitis/OpenFace>
* SeetaFace: <https://github.com/seetaface/SeetaFaceEngine>

活体验证关卡包括：
* 基于面部材质的光流特征
* 基于面部旋转时关键点的欧氏距离特征
* 基于活体的嘴部可动性特征
* 基于活体的眼部可动性特征

## 安装

安装仅在Mac上进行测试，其它系统安装方法类似。

### 准备

* 首先安装Homebrew:<https://brew.sh/>，homebrew是类似于Linux系统上apt-get的软件。
* 安装TBB，OpenCV3（必须3及以上版本）和 boost

<code> brew install boost tbb opencv3 </code>

### Build

在命令行执行下列指令：
'''bash
  mkdir build
  cd build 
  cmake -D CMAKE_BUILD_TYPE=RELEASE ..
  make'''

### 测试

进入到<code>build/bin</code>目录下。执行

<code> ./FaceLivenessDetection </code>

## 功能

