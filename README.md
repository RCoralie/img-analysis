# imageAnalysis

## Initial setup

```
./installPrerequisites
conan profile update settings.compiler.libcxx=libstdc++11 default
mkdir build
cd build
conan install .. -s build_type=Release
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```
