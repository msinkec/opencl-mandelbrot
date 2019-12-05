#include <stdio.h>
#include <stdlib.h>
#include <FreeImage.h>
#include <math.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE 10000

cl_int ret;

char* readKernel(const char* file) {
    FILE *fp;
    size_t source_size;
    char* source_str;

    fp = fopen(file, "r");
    if (!fp) {
        fprintf(stderr, ":-(#\n");
        exit(1);
    }

    source_str = (char*) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);

    return source_str;
}

void printKernelBuildLog(cl_program program, cl_device_id device_id)  {
       size_t build_log_len;
    char *build_log;
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                    0, NULL, &build_log_len);
    build_log = (char *) malloc(sizeof(char) * (build_log_len + 1));
    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                    build_log_len, build_log, NULL);
    printf("%s\n", build_log);
    free(build_log);
}


int main(void)
{
    int width = 3840;
    int height = 2160;
    int img_size = height * width * 4;
    int max_iteration = 800;
    int pitch = ((32 * width + 31) / 32) * 4;

    // Allocate memory for image (RGBA)
    unsigned char *image = (unsigned char *) malloc(img_size);


    ////////////////// CL ///////////////////
    
    // Define dimensions:
    // break problem down into local WGs, which must evenly fit into the global 2D space
    // that means: height must be divisible by WGx && width must be divisible with WGy
    size_t * global_work_size = (size_t*) malloc(sizeof(size_t)*2);
    size_t * local_work_size = (size_t*) malloc(sizeof(size_t)*2);
    local_work_size[0] = 16;
    local_work_size[1] = 16;

    global_work_size[0] = width;
    while (global_work_size[0] % 16 != 0) {
        global_work_size[0]++;
    }
    
    global_work_size[1] = height;
    while (global_work_size[1] % 16 != 0) {
        global_work_size[1]++;
    }
    
    printf("Global worksize: %ld x %ld\n", (long) global_work_size[0], (long) global_work_size[1]);
    printf("Local worksize: %ld x %ld\n", (long) local_work_size[0], (long) local_work_size[1]);

    // Read kernel
    char* source_str = readKernel("mandelbrot-kernel.cl");
    
    // Get platform info for OpenCL
    cl_platform_id    platform_id[10];
    cl_uint            n_platforms;
    ret = clGetPlatformIDs(10, platform_id, &n_platforms);


    // Get GPU device info of first platform
    cl_device_id    device_ids[10];
    cl_uint            n_devices;
    ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10,
                            device_ids, &n_devices);

    // Create context, we'll use only the first GPU
    cl_int ret;
       cl_context context = clCreateContext(NULL, 1, device_ids, NULL, NULL, &ret);

    // Create OpenCL command queue for context
       cl_command_queue command_queue = clCreateCommandQueue(context, device_ids[0], 0, &ret);

    // Memory allocation on the GPU
    size_t atom_buffer_size = height * width * sizeof(unsigned char) * 4;
    cl_mem img_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        atom_buffer_size, NULL, &ret);

    // Prepare and build kernel program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**) &source_str,
                                                    NULL, &ret);
    ret = clBuildProgram(program, 1, device_ids, NULL, NULL, NULL);

    // Print build log of kernel
    //printKernelBuildLog(program, device_ids[0]);

    // Prepare kernel object
    cl_kernel kernel = clCreateKernel(program, "mandelbrot", &ret);
    
    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &img_cl);
    ret |= clSetKernelArg(kernel, 1, sizeof(int), (void*) &max_iteration);

    // Run kernel
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                                global_work_size, local_work_size, 0, NULL, NULL);

    // Wait for the kernel to finish 
    clFinish(command_queue);

    // Retrieve image data from device
    ret = clEnqueueReadBuffer(command_queue, img_cl, CL_TRUE, 0,
                                atom_buffer_size, image, 0, NULL, NULL);
    
    // Save image
    FIBITMAP *dst = FreeImage_ConvertFromRawBits(image, width, height, pitch,
        32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
    FreeImage_Save(FIF_PNG, dst, "mandelbrot.png", 0);
   

    //////////////// CLEANUP /////////////////
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(img_cl);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(image);
    free(source_str);

    return 0;
}
