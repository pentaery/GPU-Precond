include_directories("/home/ET/yjzhou/HPCSoft/amgx/include")


ADD_EXECUTABLE(test1 test.cc)

target_link_libraries(test1 "/home/ET/yjzhou/HPCSoft/amgx/build/libamgxsh.so")