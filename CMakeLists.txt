cmake_minimum_required(VERSION 3.16)

project(ImageProcessTool VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)


# OpenCV-4.6.0特有配置 发布用 80MB
# set(OpenCV_DIR "E:/OpenCV/opencv460/cmake-mingw-build1/install/x64/mingw/lib")  # OpenCV4.6.0
# find_package(OpenCV 4.6.0 REQUIRED)  # 查找OpenCV包 OpenCVConfig.cmake
# include_directories(ImageProcess "E:/OpenCV/opencv460/cmake-mingw-build1/install/include")  # OpenCV包含目录(头文件) 指定目录
# OpenCV包含目录(头文件) 由于编译后太占存储空间只保留了E:\OpenCV\opencv460\cmake-mingw-build1\install下的include目录和x64目录，下面写法不行了
# include_directories(ImageProcess ${OpenCV_INCLUDE_DIRS})
# 还要将E:\OpenCV\opencv460\cmake-mingw-build1\install\x64\mingw\bin下libopencv_world460.dll动态链接库文件复制到E:\Qt\6.5.3\mingw_64\bin下
# 发布操作步骤 1: 切换至Release模式，构建项目; 2: 新建文件夹把构建生成的ImageProcessTool.exe移动到新建的文件夹，比如C:\Users\ouc\Downloads\ImageProcessTool
# 发布操作步骤 3: 再新建的文件夹目录下打开cmd，执行以下命令C:\Qt\6.5.2\mingw_64\bin\windeployqt.exe . (点代表当前目录，不要漏掉)
# 发布操作步骤 4: 最后把libopencv_world460.dll文件复制到新建的文件夹下，即可运行程序 一共106MB

# OpenCV-4.10.0特有配置 开发用 152MB 含opencv_contrib太大了
set(OpenCV_DIR "E:/OpenCV/opencv4100/cmake-mingw-build1-qt-contrib/install/x64/mingw/lib")  # OpenCV4.10.0
find_package(OpenCV 4.10.0 REQUIRED)  # 查找OpenCV包 OpenCVConfig.cmake
include_directories(ImageProcess "E:/OpenCV/opencv4100/cmake-mingw-build1-qt-contrib/install/include")  # OpenCV包含目录(头文件) 指定目录
# OpenCV包含目录(头文件) 由于编译后太占存储空间只保留了E:\OpenCV\opencv4100\cmake-mingw-build1-qt-contrib\install下的include目录和x64目录，下面写法不行了
# include_directories(ImageProcess ${OpenCV_INCLUDE_DIRS})
# 还要将E:\OpenCV\opencv4100\cmake-mingw-build1-qt-contrib\install\x64\mingw\bin下所有.dll动态链接库文件复制到E:\Qt\6.5.3\mingw_64\bin下

# 无论什么版本均要配置
# target_link_libraries(ImageProcessTool ${OpenCV_LIBS}) #  OpenCV静态链接库 已在下面合并配置，如下所示
# target_link_libraries(ImageProcessTool PRIVATE Qt${QT_VERSION_MAJOR}::Widgets ${OpenCV_LIBS})  # OpenCV静态链接库


set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(ImageProcessTool
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
        res.qrc
        ImageProcessAPI.h ImageProcessAPI.cpp
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET ImageProcessTool APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(ImageProcessTool SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(ImageProcessTool
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(ImageProcessTool PRIVATE Qt${QT_VERSION_MAJOR}::Widgets ${OpenCV_LIBS})  # OpenCV静态链接库

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.ImageProcessTool)
endif()
set_target_properties(ImageProcessTool PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS ImageProcessTool
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(ImageProcessTool)
endif()
