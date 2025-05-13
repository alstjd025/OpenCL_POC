2025 Minsung Kim <alstjd025@gmail.com> 

# How to compile for Android 64 (e.g., Google Pixel 6)
First, prepare a docker container contains android-ndk build toolchains.
It is easy to use a prebuilt image from TensorFlow Lite Android. (https://www.tensorflow.org/lite/guide/build_android)
For instance, one can use dockerfile in android_build folder (exactley same as tflite given).

# Build example
If you wanna build hello.cpp file for 64-bit LSB arm64, the compile commands are just like below.
<pre>
<code>
/android/ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++   --target=aarch64-linux-android30   -static-libstdc++   -fPIE -pie -std=c++17   hello.cpp -o hello
# i recommend to statically link libstdc++ or libc, because sometimes the android LD cannot find them.
</code>
</pre>

If you wanna build some binary with external libraries..
<pre>
<code>
/android/ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++   --target=aarch64-linux-android30   -static-libstdc++   -fPIE -pie -std=c++17 -I./  -L./libs -lOpenCL unified.cpp -o unified
</code>
</pre>



