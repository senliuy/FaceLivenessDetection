#TBB library
include_directories(${TBB_ROOT_DIR}/include)

# Local libraries
#include_directories(${LandmarkDetector_SOURCE_DIR}/include)
#include_directories(${seetaFace_SOURCE_DIR}/include)
	
include_directories(../../lib/local/LandmarkDetector/include)
#include_directories(../../lib/local/seetaFace/include)
include_directories(LivenessDetection)
			
add_executable(FaceLivenessDetection FaceLivenessDetection)
target_link_libraries(FaceLivenessDetection LandmarkDetector)
#target_link_libraries(FaceLivenessDetection seetaFace)
target_link_libraries(FaceLivenessDetection dlib)

target_link_libraries(FaceLivenessDetection ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})

install (TARGETS FaceLivenessDetection DESTINATION bin)
