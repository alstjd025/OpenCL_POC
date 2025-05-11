#define CL_TARGET_OPENCL_VERSION 120
#include "CL/cl.h"
#include <iostream>

const char* kernelSource = R"CLC(
    __kernel void add_one(__global int* data) {
        int id = get_global_id(0);
        data[id] += 1;
    }
)CLC";

int main() {
    std::cout << "hello" << "\n";
    cl_platform_id platform;
    cl_device_id device;
    cl_uint numPlatforms;
    clGetPlatformIDs(1, &platform, &numPlatforms);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    std::cout << "init" << "\n";
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    // 3. CL_MEM_ALLOC_HOST_PTR을 사용하여 공유 메모리 버퍼 생성
    int dataSize = 10;
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, dataSize * sizeof(int), nullptr, nullptr);
    std::cout << "Created buffer on GPU with host ptr" << "\n";
    // 4. CPU에서 메모리 매핑 후 초기화
    int* hostData = (int*)clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_WRITE, 0, dataSize * sizeof(int), 0, nullptr, nullptr, nullptr);
    std::cout << "Get ptr from clEnqueueMapBuffer" << "\n";
    for (int i = 0; i < dataSize; i++) hostData[i] = i;
    clEnqueueUnmapMemObject(queue, buffer, hostData, 0, nullptr, nullptr);
    std::cout << "Changed values" << "\n";
    clFinish(queue);

    // 5. 프로그램 및 커널 생성
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "add_one", nullptr);

    // 6. 커널에 공유 버퍼 전달
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);

    // 7. GPU에서 커널 실행
    size_t globalSize = dataSize;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    // 8. CPU에서 다시 메모리 매핑 후 결과 확인
    hostData = (int*)clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ, 0, dataSize * sizeof(int), 0, nullptr, nullptr, nullptr);
    for (int i = 0; i < dataSize; i++) {
        std::cout << "data[" << i << "] = " << hostData[i] << std::endl;
    }
    clEnqueueUnmapMemObject(queue, buffer, hostData, 0, nullptr, nullptr);
    clFinish(queue);

    // 9. 자원 해제
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
