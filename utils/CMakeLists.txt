file(GLOB_RECURSE CPP_SOURCES "*.cpp")

add_library(utils SHARED ${CPP_SOURCES})