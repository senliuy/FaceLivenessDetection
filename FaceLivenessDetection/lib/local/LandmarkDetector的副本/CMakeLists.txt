#TBB library
include_directories(${TBB_ROOT_DIR}/include)

include_directories(${BOOST_INCLUDE_DIR})

SET(SOURCE
    src/CCNF_patch_expert.cpp
	src/LandmarkDetectionValidator.cpp
    src/LandmarkDetectorFunc.cpp
	src/LandmarkDetectorModel.cpp
    src/LandmarkDetectorUtils.cpp
	src/LandmarkDetectorParameters.cpp
	src/Patch_experts.cpp
	src/PAW.cpp
    src/PDM.cpp
	src/SVR_patch_expert.cpp
	src/stdafx.cpp
    src/classifier/lab_boosted_classifier.cpp
src/classifier/mlp.cpp
src/classifier/surf_mlp.cpp
src/face_detection.cpp
src/feat/lab_feature_map.cpp
src/feat/surf_feature_map.cpp
src/fust.cpp
src/io/lab_boost_model_reader.cpp
src/io/surf_mlp_model_reader.cpp
src/util/image_pyramid.cpp
src/util/nms.cpp
)

SET(HEADERS
    include/CCNF_patch_expert.h	
    include/LandmarkCoreIncludes.h
	include/LandmarkDetectionValidator.h
    include/LandmarkDetectorFunc.h
	include/LandmarkDetectorModel.h
	include/LandmarkDetectorParameters.h
	include/LandmarkDetectorUtils.h
	include/Patch_experts.h	
    include/PAW.h
	include/PDM.h
	include/SVR_patch_expert.h		
	include/stdafx.h
include/classifier/lab_boosted_classifier.h
include/classifier/mlp.h
include/classifier/surf_mlp.h
include/classifier.h
include/common.h
include/detector.h
include/face_detection.h
include/feat/lab_feature_map.h
include/feat/surf_feature_map.h
include/feature_map.h
include/fust.h
include/io/lab_boost_model_reader.h
include/io/surf_mlp_model_reader.h
include/model_reader.h
include/util/image_pyramid.h
include/util/math_func.h
include/util/nms.h
)

include_directories(./include)
include_directories(../seetaFace/include)
include_directories(${LandmarkDetector_SOURCE_DIR}/include)


add_library( LandmarkDetector ${SOURCE} ${HEADERS} )

install (TARGETS LandmarkDetector DESTINATION lib)
install (FILES ${HEADERS} DESTINATION include/FaceLivenessDetection)
