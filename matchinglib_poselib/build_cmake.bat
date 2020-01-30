pushd %~dp0
rmdir /Q /S build
mkdir build
cd build
cmake .. -G"Visual Studio 10 2010 Win64"
popd