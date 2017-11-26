
if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267") # int64 -> int
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244") # float -> int
endif()

add_library(ao-kernel STATIC
    eval/base.cpp
    ../include/ao/eval/base.hpp
    eval/eval_interval.cpp
    ../include/ao/eval/eval_interval.hpp
    ../include/ao/eval/interval.hpp
    ../include/ao/eval/clause.hpp
    eval/eval_array.cpp
    ../include/ao/eval/eval_array.hpp
    eval/eval_deriv_array.cpp
    ../include/ao/eval/eval_deriv_array.hpp
    eval/eval_deriv.cpp
    ../include/ao/eval/eval_deriv.hpp
    eval/eval_feature.cpp
    ../include/ao/eval/eval_feature.hpp
    eval/eval_point.cpp
    ../include/ao/eval/eval_point.hpp
    eval/tape.cpp
    ../include/ao/eval/tape.hpp
    eval/feature.cpp
    ../include/ao/eval/feature.hpp
    render/discrete/heightmap.cpp
    ../include/ao/render/discrete/heightmap.hpp
    ../include/ao/render/discrete/eval_height.hpp
    render/discrete/voxels.cpp
    ../include/ao/render/discrete/voxels.hpp
    ../include/ao/render/axes.hpp
    render/brep/xtree.cpp
    ../include/ao/render/brep/xtree.hpp
    ../include/ao/render/brep/eval_xtree.hpp
    ../include/ao/render/brep/brep.hpp
    ../include/ao/render/brep/dual.hpp
    ../include/ao/render/brep/ao_MeshProximityTable.h
    ../include/ao/eval/primitive.h
    render/brep/contours.cpp
    ../include/ao/render/brep/contours.hpp
    render/brep/mesh.cpp
    ../include/ao/render/brep/mesh.hpp
    render/brep/marching.cpp
    ../include/ao/render/brep/marching.hpp
    ../include/ao/render/brep/region.hpp
    tree/cache.cpp
    ../include/ao/tree/cache.hpp
    tree/opcode.cpp
    ../include/ao/tree/opcode.hpp
    tree/template.cpp
    ../include/ao/tree/template.hpp
    tree/tree.cpp
    ../include/ao/tree/tree.hpp
    ao.cpp
    ../include/ao/ao.h
)

target_link_libraries(ao-kernel PUBLIC boost eigen3 ntglm)

if(WIN32)
  target_link_libraries(ao-kernel PUBLIC pngstatic)
endif()

target_include_directories(ao-kernel PUBLIC
    ../include
)

target_compile_definitions(ao-kernel PUBLIC
    _USE_MATH_DEFINES
)

target_compile_features(ao-kernel PUBLIC cxx_std_17)
target_compile_definitions(ao-kernel PUBLIC "_HAS_AUTO_PTR_ETC")
