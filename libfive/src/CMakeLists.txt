
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267") # int64 -> int
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244") # float -> int

add_library(ao-kernel STATIC
   ../include/libfive/eval/base.hpp
   ../include/libfive/eval/clause.hpp
   ../include/libfive/eval/eval_array.hpp
   ../include/libfive/eval/eval_deriv.hpp
   ../include/libfive/eval/eval_deriv_array.hpp
   ../include/libfive/eval/eval_feature.hpp
   ../include/libfive/eval/eval_interval.hpp
   ../include/libfive/eval/eval_point.hpp
   ../include/libfive/eval/feature.hpp
   ../include/libfive/eval/interval.hpp
   ../include/libfive/eval/primitive.h
   ../include/libfive/eval/tape.hpp
   ../include/libfive/eval/transformed_primitive.h
   ../include/libfive/render/brep/brep.hpp
   ../include/libfive/render/brep/contours.hpp
   ../include/libfive/render/brep/dual.hpp
   ../include/libfive/render/brep/eval_xtree.hpp
   ../include/libfive/render/brep/marching.hpp
   ../include/libfive/render/brep/mesh.hpp
   ../include/libfive/render/brep/partial_octree.h
   ../include/libfive/render/brep/region.hpp
   ../include/libfive/render/brep/xtree.hpp
   ../include/libfive/render/discrete/eval_height.hpp
   ../include/libfive/render/discrete/heightmap.hpp
   ../include/libfive/render/discrete/voxels.hpp
   ../include/libfive/solve/bounds.hpp
   ../include/libfive/solve/solver.hpp
   ../include/libfive/tree/cache.hpp
   ../include/libfive/tree/opcode.hpp
   ../include/libfive/tree/template.hpp
   ../include/libfive/tree/tree.hpp
   ../include/libfive.h
   

    eval/base.cpp
    eval/eval_array.cpp
    eval/eval_deriv.cpp
    eval/eval_deriv_array.cpp
    eval/eval_feature.cpp
    eval/eval_interval.cpp
    eval/eval_point.cpp
    eval/feature.cpp
    eval/tape.cpp
    eval/transformed_primitive.cpp
    render/brep/contours.cpp
    render/brep/marching.cpp
    render/brep/mesh.cpp
    render/brep/xtree.cpp
    render/discrete/heightmap.cpp
    render/discrete/voxels.cpp
    solve/bounds.cpp
    solve/solver.cpp
    tree/cache.cpp
    tree/opcode.cpp
    tree/template.cpp
    tree/tree.cpp
    libfive.cpp
)

target_link_libraries(ao-kernel PUBLIC boost eigen3 pngstatic)

target_include_directories(ao-kernel PUBLIC
    ../include
)

target_compile_definitions(ao-kernel PUBLIC
    _USE_MATH_DEFINES
)

target_compile_features(ao-kernel PUBLIC cxx_std_17)
target_compile_definitions(ao-kernel PUBLIC "_HAS_AUTO_PTR_ETC")
