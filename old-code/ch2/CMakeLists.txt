cmake_minimum_required( VERSION 2.8 )
project( HelloSLAM )

add_library( hello libHelloSLAM.cc)
add_library( hello_shared SHARED libHelloSLAM.cc)
add_executable( helloSLAM helloSLAM.cc)
target_link_libraries( helloSLAM hello_shared )