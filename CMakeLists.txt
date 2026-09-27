cmake_minimum_required(VERSION 3.16)
project(agnitio CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- System dependencies ---------------------------------------------------
# OpenCV must be built with the dnn module (default in most package
# distributions and the official builds).
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn objdetect)

find_package(PostgreSQL REQUIRED)

find_package(libpqxx QUIET CONFIG)
if(NOT libpqxx_FOUND)
    # Fall back to plain library discovery for distros that don't ship a
    # CMake config for pqxx (Ubuntu's libpqxx-dev, notably).
    find_library(PQXX_LIBRARY NAMES pqxx REQUIRED)
    find_path(PQXX_INCLUDE_DIR NAMES pqxx/pqxx REQUIRED)
endif()

# --- Header-only dependencies, fetched at configure time -------------------
include(FetchContent)

FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.15.3
)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(httplib nlohmann_json)

# --- Target ------------------------------------------------------------
add_executable(agnitio
    src/main.cpp
    src/face_engine.cpp
    src/db.cpp
)

target_include_directories(agnitio PRIVATE
    include
    ${PostgreSQL_INCLUDE_DIRS}
)

target_link_libraries(agnitio PRIVATE
    ${OpenCV_LIBS}
    ${PostgreSQL_LIBRARIES}
    httplib::httplib
    nlohmann_json::nlohmann_json
)

if(libpqxx_FOUND)
    target_link_libraries(agnitio PRIVATE libpqxx::pqxx)
else()
    target_include_directories(agnitio PRIVATE ${PQXX_INCLUDE_DIR})
    target_link_libraries(agnitio PRIVATE ${PQXX_LIBRARY})
endif()
