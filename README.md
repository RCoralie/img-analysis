# imageAnalysis

## Initial setup

Use the installer provided to install all prerequisites dependencies except opencv :

```
./installPrerequisites
```

OpenCV is not installed directly via the packaging tool (apt), it must be built and installed from source with Qt :

[OPTIONAL] If opencv is already installed you must uninstall it:
```
  sudo apt-get purge '*opencv*'
```

Using the following command to get the OpenCV source code and prepare the build:
```
git clone https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build
```

We don't include the examples in the build, but feel free to include them. Also feel free to set other flags and customise your build as you see fit.
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON  \
      -D WITH_GTK=ON  \
      -D WITH_OPENGL=ON ..
```

If CMake didn't report any errors or missing libraries, continue with the build.
```
make -j$(nproc)
```

If no errors were produced, we can carry on with installing OpenCV to the system:
```
sudo make install
```

Now OpenCV should be available to your system. You can use the following lines to know where OpenCV was installed and which libraries were installed:
```
pkg-config --cflags opencv  # get the include path (-I)
pkg-config --libs opencv    # get the libraries path (-L) and the libraries (-l)
```

If no errors were produced, we can build the project :
```
cd ../.. && mkdir build && cd build
conan profile update settings.compiler.libcxx=libstdc++11 default
conan install .. -s build_type=Release
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
