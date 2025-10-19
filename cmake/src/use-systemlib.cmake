find_package(PkgConfig REQUIRED)

pkg_check_modules(MINIZIP REQUIRED minizip)
include_directories(${MINIZIP_INCLUDE_DIRS})

pkg_check_modules(CJSON REQUIRED libcjson)
include_directories(${CJSON_INCLUDE_DIRS})

add_library(securec INTERFACE)
target_link_libraries(securec INTERFACE -lboundscheck)
