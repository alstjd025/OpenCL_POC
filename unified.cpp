/* 
  Copyright Minsung Kim, NXC, SNU
  alstjd025@gmail.com
*/

#include <cstdio>
#include <iostream>
#include <ctime>  // clock_gettime
#include <chrono> // optional
#include <fstream>
#include <unistd.h>
#include <sstream>  


#define RESET       "\033[0m"
#define BLACK       "\033[30m"
#define RED         "\033[31m"
#define GREEN       "\033[32m"
#define YELLOW      "\033[33m"
#define BLUE        "\033[34m"
#define MAGENTA     "\033[35m"
#define CYAN        "\033[36m"
#define WHITE       "\033[37m"

#define CL_TARGET_OPENCL_VERSION 120
#include "CL/cl.h"

const char* kernelSource = R"CLC(
    __kernel void add_one(__global int* data) {
        int id = get_global_id(0);
        data[id] += 1;
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
  std::cout << "OpenCL test program, PID " << RED << getpid() << RESET << "\n";
  cl_platform_id platform;
  cl_device_id device;
  cl_uint numPlatforms;
  PrintRSSandPSS("before OpenCL initialization");
  sleep(20); // 이 시점에서 smaps/lsof 확인 가능
  clGetPlatformIDs(1, &platform, &numPlatforms);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
  // cl_context context_two = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
  PrintRSSandPSS("after clCreateContext");
  std::cout << "sleep for 20sec." << std::endl;
  sleep(20); // 이 시점에서 smaps/lsof 확인 가능
  
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);
  // cl_command_queue queue_two = clCreateCommandQueue(context_two, device, 0, nullptr);
  PrintRSSandPSS("after clCreateCommandQueue");
  
  int dataSize = 10;
  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, dataSize * sizeof(int), nullptr, nullptr);
  // cl_mem buffer_two = clCreateBuffer(context_two, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, dataSize * sizeof(int), nullptr, nullptr);
  std::cout << "Created buffer on GPU with host ptr" << "\n";
  PrintRSSandPSS("after clCreateBuffer");

  // 4. CPU에서 메모리 매핑 후 초기화
  //  clEnqueueMapBuffer returns device buffer ptr to host accesible ptr.
  int* hostData = (int*)clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_WRITE, 0, dataSize * sizeof(int), 0, nullptr, nullptr, nullptr);
  // int* hostData_two = (int*)clEnqueueMapBuffer(queue_two, buffer_two, CL_TRUE, CL_MAP_WRITE, 0, dataSize * sizeof(int), 0, nullptr, nullptr, nullptr);
  std::cout << "Get ptr from clEnqueueMapBuffer" << "\n";
  for (int i = 0; i < dataSize; i++) hostData[i] = i;
  // for (int i = 0; i < dataSize; i++) hostData_two[i] = i+10;
  PrintRSSandPSS("after clEnqueueMapBuffer");

  clEnqueueUnmapMemObject(queue, buffer, hostData, 0, nullptr, nullptr);
  // clEnqueueUnmapMemObject(queue_two, buffer_two, hostData_two, 0, nullptr, nullptr);
  PrintRSSandPSS("after clEnqueueUnmapMemObject");
  clFinish(queue);
  // clFinish(queue_two);
  
  // 5. 프로그램 및 커널 생성
  cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, nullptr);
  PrintRSSandPSS("after clCreateProgramWithSource");
  // cl_program program_two = clCreateProgramWithSource(context_two, 1, &kernelSource, nullptr, nullptr);
  clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  PrintRSSandPSS("after clBuildProgram");
  std::cout << "sleep for 20sec." << std::endl;
  sleep(20); // 이 시점에서 smaps/lsof 확인 가능


  // clBuildProgram(program_two, 1, &device, nullptr, nullptr, nullptr);
  cl_kernel kernel = clCreateKernel(program, "add_one", nullptr);
  // cl_kernel kernel_two = clCreateKernel(program_two, "add_one", nullptr);
  PrintRSSandPSS("after clCreateKernel");

  // 6. 커널에 공유 버퍼 전달
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
  // clSetKernelArg(kernel_two, 0, sizeof(cl_mem), &buffer_two);

  // 7. GPU에서 커널 실행
  size_t globalSize = dataSize;
  clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
  // clEnqueueNDRangeKernel(queue_two, kernel_two, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
  clFinish(queue);
  // clFinish(queue_two);
  PrintRSSandPSS("after clEnqueueNDRangeKernel");

  // 8. CPU에서 다시 메모리 매핑 후 결과 확인
  hostData = (int*)clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ, 0, dataSize * sizeof(int), 0, nullptr, nullptr, nullptr);
  // hostData_two = (int*)clEnqueueMapBuffer(queue_two, buffer_two, CL_TRUE, CL_MAP_READ, 0, dataSize * sizeof(int), 0, nullptr, nullptr, nullptr);
  for (int i = 0; i < dataSize; i++) {
      std::cout << "data[" << i << "] = " << hostData[i] << std::endl;
  }
  // for (int i = 0; i < dataSize; i++) {
  //     std::cout << "data[" << i << "] = " << hostData_two[i] << std::endl;
  // }
  // see RES and RSS here.
  clEnqueueUnmapMemObject(queue, buffer, hostData, 0, nullptr, nullptr);
  // clEnqueueUnmapMemObject(queue_two, buffer_two, hostData_two, 0, nullptr, nullptr);
  clFinish(queue);
  // clFinish(queue_two);

  PrintRSSandPSS("after CPU data read");

  // 9. 자원 해제
  // [TODO] what happens to memory if release the OpenCL resources?
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(buffer);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}
