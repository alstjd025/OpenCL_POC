#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdio>
#include <ctime>  // clock_gettime
#include <chrono> // optional
#include <fstream>
#include <unistd.h>
#include <memory.h>
#include <sstream>  

#define CHECK(status, msg) if ((status) != CL_SUCCESS) { std::cerr << "Error: " \
                         << msg << " (" << status << ")" << std::endl; exit(1); }

#define RESET       "\033[0m"
#define BLACK       "\033[30m"
#define RED         "\033[31m"
#define GREEN       "\033[32m"
#define YELLOW      "\033[33m"
#define BLUE        "\033[34m"
#define MAGENTA     "\033[35m"
#define CYAN        "\033[36m"
#define WHITE       "\033[37m"

const char* kernelSource = R"CLC(
__kernel void add10(__global int* data) {
    int gid = get_global_id(0);
    data[gid] += 10;
}
)CLC";


size_t get_resident_set_size_kb() {
    std::ifstream status_file("/proc/self/status");
    std::string line;

    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            size_t res_kb;
            sscanf(line.c_str(), "VmRSS: %lu kB", &res_kb);
            return res_kb;
        }
    }
    return 0;
}

size_t get_current_process_pss_kb() {
  const char* smaps_path = "/proc/self/smaps";
  std::ifstream smaps_file(smaps_path);

  if (!smaps_file.is_open()) {
    std::cerr << "Failed to open " << smaps_path << std::endl;
    return 0;
  }
  std::string line;
  size_t total_pss_kb = 0;
  while (std::getline(smaps_file, line)) {
    if (line.find("Pss:") == 0) {
      std::istringstream iss(line);
      std::string key;
      size_t pss_kb;

      iss >> key >> pss_kb;
      total_pss_kb += pss_kb;
    }
  }
  smaps_file.close();
  return total_pss_kb;
}

void PrintRSSandPSS(std::string log){
  size_t res_kb;
  size_t pss_kb;
  std::cout << "Memory " << log << "\n";
  res_kb = get_resident_set_size_kb();
  pss_kb = get_current_process_pss_kb();
  std::cout << "  RSS (RES): " << GREEN << res_kb << " kB" << RESET 
        << " (" << GREEN << (res_kb / 1024.0) << " MB" << RESET << ")" << "\n";
  std::cout << "  PSS      : " << GREEN << pss_kb << " kB" << RESET 
            << " (" << GREEN << (pss_kb / 1024.0) << " MB" << RESET << ")" << "\n";
}

int main() {
    const size_t element_count = 25'000'000;
    const size_t buffer_size = element_count * sizeof(int);

    // 1. CPU 메모리 할당 및 초기화
    std::vector<int> host_input(element_count);
    for (size_t i = 0; i < element_count; ++i) {
        host_input[i] = static_cast<int>(i + 1);
    }
    PrintRSSandPSS("host buffer init");
    // 2. OpenCL 플랫폼 & 디바이스 설정
    cl_int status;
    cl_uint num_platforms;
    CHECK(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs");
    std::vector<cl_platform_id> platforms(num_platforms);
    CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr), "clGetPlatformIDs");

    cl_platform_id platform = platforms[0];

    cl_uint num_devices;
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices), "clGetDeviceIDs");
    std::vector<cl_device_id> devices(num_devices);
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr), "clGetDeviceIDs");

    cl_device_id device = devices[0];
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    CHECK(status, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &status);
    CHECK(status, "clCreateCommandQueueWithProperties");

    // 3. GPU 버퍼 생성 (HOST_PTR 사용하지 않음)
    cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, nullptr, &status);
    CHECK(status, "clCreateBuffer");
    PrintRSSandPSS("create GPU buffer");
    // 4. 호스트 데이터를 GPU로 복사
    CHECK(clEnqueueWriteBuffer(queue, device_buffer, CL_TRUE, 0, buffer_size, host_input.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer");
    PrintRSSandPSS("copy host buffer to GPU buffer");
    // 5. 커널 컴파일
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &status);
    CHECK(status, "clCreateProgramWithSource");
    CHECK(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr), "clBuildProgram");

    cl_kernel kernel = clCreateKernel(program, "add10", &status);
    CHECK(status, "clCreateKernel");

    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_buffer), "clSetKernelArg");

    // 6. 커널 실행
    size_t global_work_size = element_count;
    CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    PrintRSSandPSS("execute kernel");
    // 7. 결과 GPU→CPU 복사
    std::vector<int> host_output(element_count);
    CHECK(clEnqueueReadBuffer(queue, device_buffer, CL_TRUE, 0, buffer_size, host_output.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer");
    PrintRSSandPSS("copy GPU buffer to host");
    // 8. 결과 확인 (앞 10개 출력)
    for (int i = 0; i < 10; ++i) {
        std::cout << "output[" << i << "] = " << host_output[i] << std::endl;
    }
    
    /*
    Expectation
    100
    200
    300
    
    */

    // 9. 자원 정리
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(device_buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
