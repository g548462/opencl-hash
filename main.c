#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>

#define DEBUG 0
#define TUPLES 10000
#define ARRAY_SIZE 100000
#define TIMES 1000
#define MAX_SOURCE_SIZE (0x100000)

#define ZERO_KEY 0xFFFFFFE
#define EMPTY_KEY 0
#define hash_constant 0x348c3def

inline unsigned int hash1(unsigned int val) {
  val = (val+0x7ed55d16) + (val<<12);
  val = (val^0xc761c23c) ^ (val>>19);
  val = (val+0x165667b1) + (val<<5);
  val = (val+0xd3a2646c) ^ (val<<9);
  val = (val+0xfd7046c5) + (val<<3);
  val = (val^0xb55a4f09) ^ (val>>16);
  return val;
}


void loadKernel(char *filename, char **programSource, size_t *source_size)
{
    FILE *fp;
    fp = fopen(filename, "r");
    if(!fp)
    {
        fprintf(stderr, "**[Error]**\tFail to load kernel\n");
        exit(1);
    }
    *programSource = (char*)malloc(MAX_SOURCE_SIZE);
    *source_size = fread(*programSource, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
}

int main()
{
    srand(time(0));
    int i, tmp1, tmp2, tmp3;
    int *array;
    size_t datasize = sizeof(int) * TUPLES;


    array = (int*)malloc(sizeof(int) * ARRAY_SIZE);
    for(i=0; i<ARRAY_SIZE; i++) {
        array[i] = i;
    }
    clock_t begin, end;
    
    int *hash_table;
    int hash_table_size = 2 * sizeof(int) * TUPLES;
    int *data;
    data = (int*)malloc(sizeof(int) * TUPLES);
    hash_table = (int*)malloc(sizeof(int) * hash_table_size);

    cl_int status;
    cl_int numPlatforms = 0;

    //  Create unique random number
    for(i=0; i<ARRAY_SIZE; i++) {
        int tmp1 = rand()%ARRAY_SIZE;
        int tmp2 = rand()%ARRAY_SIZE;
        tmp3 = array[tmp1];
        array[tmp1] = array[tmp2];
        array[tmp2] = tmp3;
    }

    //  Init random data for Hash
    for(i=0; i<TUPLES; i++)
    {
        data[i] = array[i];
        //debug code
        if(DEBUG){
            printf("%d hash: %u\n", data[i], hash1(data[i]) % hash_table_size);
        }
    }

    printf("\n");
    //Get Platform
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    printf("\t[info] Total %d platforms... \n", numPlatforms);
    cl_platform_id *platforms = NULL;
    platforms  = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
    status |= clGetPlatformIDs(numPlatforms, platforms, NULL);

    //Get Devices
    cl_int numDevices = 0;
    status = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    printf("\t[info] Total %d Devices... \n", numDevices);
    cl_device_id *devices;
    devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
    status |= clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);

    size_t cb;
    char *devicename;
    devicename = (char*)malloc(sizeof(char)*100);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, devicename, 0);
    printf("DeviceName = %s\n", devicename);

    if(status == CL_SUCCESS) printf("\t[Info] Get DeviceIDs Success!\n");  
    else fprintf(stderr, "\t[Error] Get DeviceIDs Fail!\n");

    cl_context context;
    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status); 
    if(status == CL_SUCCESS) printf("\t[Info] Create Context Success!\n");  
    else fprintf(stderr, "\t[Error] Create Context Fail!\n");

    cl_command_queue cmdQueue;  
    cmdQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);  
    if(status == CL_SUCCESS) printf("\t[Info] Create CommandQueue Success!\n");  
    else fprintf(stderr, "\t[Error] Create CommandQueue Fail!\n");

    cl_mem bufA, buf_hash_table;
    buf_hash_table = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * hash_table_size, NULL, &status);
    status = clEnqueueWriteBuffer(cmdQueue, buf_hash_table, CL_FALSE, 0, sizeof(int) * hash_table_size, hash_table, 0, NULL, NULL);

    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * TUPLES, NULL, &status);
    status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_FALSE, 0, sizeof(int) * TUPLES, data, 0, NULL, NULL);

    if(status == CL_SUCCESS) printf("\t[Info] CreateBuffer and enQueue Success!\n");  
    else fprintf(stderr, "\t[Error] CreateBuffer and enQueue Fail!\n");

    char *filename = "m_hash.cl";
    char *source_str;
    size_t source_size;
    loadKernel(filename, &source_str, &source_size);

    cl_program program = clCreateProgramWithSource(context,   
        1,   
        (const char**)&source_str,   
        NULL,   
        &status);  

    if(status == CL_SUCCESS) printf("\t[Info] Create Program Success!\n");  
    else fprintf(stderr, "\t[Error] Create Program Fail!\n"); 

    status = clBuildProgram(program,        // The program object.  
        numDevices,                         // The number of devices listed in device_list.  
        devices,                            // A pointer to a list of devices associated with program.  
        NULL,  
        NULL,  
        NULL  
        );  

    if (status != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Error Build Prog.\n");
        printf("Error code = %d\n", status);
        exit(status);
    }

    if(status == CL_SUCCESS) printf("\t[Info] Build Program Success!\n");  
    else fprintf(stderr, "\t[Error] Build Program Fail! Msg:%d \n", status);  
    
    cl_kernel kernel;  
    kernel = clCreateKernel(program, "buildHashTableOptimistic", &status); 
    cl_kernel validKernel;
    validKernel = clCreateKernel(program, "validateHashTable", &status);
    cl_kernel kernelPessimistic;
    kernelPessimistic = clCreateKernel(program, "buildHashTablePessimistic", &status);

    if(status == CL_SUCCESS) printf("\t[Info] Create kernel Success!\n");  
    else fprintf(stderr, "\t[Error] Create kernel Fail! Msg:%d \n", status);

    size_t globalWorkSize[1];
    globalWorkSize[0] = TUPLES;
    unsigned char err = 0;
    cl_mem error = NULL;
    error = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(char), &err, &status);

    status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA); 
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_hash_table);  
    status |= clSetKernelArg(kernel, 2, sizeof(int), &hash_table_size); 

    status |= clSetKernelArg(validKernel, 0, sizeof(cl_mem), &bufA);
    status |= clSetKernelArg(validKernel, 1, sizeof(cl_mem), &buf_hash_table);
    status |= clSetKernelArg(validKernel, 2, sizeof(int), &hash_table_size);
    status |= clSetKernelArg(validKernel, 3, sizeof(cl_mem), &error);
    
    status |= clSetKernelArg(kernelPessimistic, 0, sizeof(cl_mem), &bufA);
    status |= clSetKernelArg(kernelPessimistic, 1, sizeof(cl_mem), &buf_hash_table);
    status |= clSetKernelArg(kernelPessimistic, 2, sizeof(int), &hash_table_size);
    status |= clSetKernelArg(kernelPessimistic, 3, sizeof(cl_mem), &error);
    if(status == CL_SUCCESS) printf("\t[Info] Set kernel Argu Success!\n");  
    else fprintf(stderr, "\t[Error] Set kernel Argu Fail! Msg:%d \n", status);

    cl_event k_event[2];
    cl_uint num_events_in_wait_list;
    cl_command_queue profile_queue;
    profile_queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);

    status = clEnqueueNDRangeKernel(cmdQueue,       // A valid command-queue  
        kernel,         // A valid kernel object.  
        1,              // work_dim  
        NULL,           // *global_work_offset  
        globalWorkSize, // *global_work_size  
        NULL,           // local_work_size  
        0,              // num_events_in_wait_list  
        NULL,           // *event_wait_list  
        &k_event[0]            // *event  
        );  
    
    clWaitForEvents(1, &k_event[0]);
    cl_ulong start_time, end_time;
    size_t return_bytes;
    err = clGetEventProfilingInfo(k_event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, &return_bytes);
    err = clGetEventProfilingInfo(k_event[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);
    double runtime = (double)(end_time - start_time);
    printf("\t[Info] Hash1 Run time = %fms\n", runtime/1000000);

    if(status == CL_SUCCESS) printf("\t[Info] enQueue kernel Success!\n");  
    else fprintf(stderr, "\t[Error] enQueue kernel Fail! Msg:%d \n", status);


    //  Read build HT result
    status = clEnqueueReadBuffer(cmdQueue,     // Refers to the command-queue in which the read command will be queued  
        buf_hash_table,         // Refers to a valid buffer object.  
        CL_TRUE,      // Indicates if the read operations are blocking or non-blocking.  
        0,            // The offset in bytes in the buffer object to read from.  
        sizeof(int) * hash_table_size,     // The size in bytes of data being read.  
        hash_table,            // The pointer to buffer in host memory where data is to be read into.  
        0,            // num_events_in_wait_list  
        NULL,         // *event_wait_list  
        NULL          // *event  
        ); 

    //  Enqueue validate kernel
    status = clEnqueueNDRangeKernel(cmdQueue,       // A valid command-queue  
        validKernel,         // A valid kernel object.  
        1,              // work_dim  
        NULL,           // *global_work_offset  
        globalWorkSize, // *global_work_size  
        NULL,           // local_work_size  
        0,              // num_events_in_wait_list  
        NULL,           // *event_wait_list  
        NULL            // *event  
        );  
    
    //  Return validate value
    status = clEnqueueReadBuffer(cmdQueue,     // Refers to the command-queue in which the read command will be queued  
        error,         // Refers to a valid buffer object.  
        CL_TRUE,      // Indicates if the read operations are blocking or non-blocking.  
        0,            // The offset in bytes in the buffer object to read from.  
        sizeof(char),     // The size in bytes of data being read.  
        &err,            // The pointer to buffer in host memory where data is to be read into.  
        0,            // num_events_in_wait_list  
        NULL,         // *event_wait_list  
        NULL          // *event  
        ); 

    // if collision occur, Do buildHashTablePessimistic
    if(err) {
        printf("\n\t-----Collision-----\n");

        //  Enqueue Pessimistic kernel
        status = clEnqueueNDRangeKernel(cmdQueue,       // A valid command-queue  
            kernelPessimistic,         // A valid kernel object.  
            1,              // work_dim  
            NULL,           // *global_work_offset  
            globalWorkSize, // *global_work_size  
            NULL,           // local_work_size  
            0,              // num_events_in_wait_list  
            NULL,           // *event_wait_list  
            &k_event[1]            // *event  
            );  
        if(status == CL_SUCCESS) printf("\t[Info] enQueue Pessimistic Success!\n");  
        else fprintf(stderr, "\t[Error] enQueue Pessimistic Fail! Msg:%d \n", status);

        //  Get runtime info
        clWaitForEvents(1, &k_event[1]);
        err = clGetEventProfilingInfo(k_event[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, &return_bytes);
        err = clGetEventProfilingInfo(k_event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);
        runtime = (double)(end_time - start_time);
        printf("\t[Info] Persimisstic Run time = %fms\n", runtime/1000000);




        status = clEnqueueReadBuffer(cmdQueue,     // Refers to the command-queue in which the read command will be queued  
            buf_hash_table,         // Refers to a valid buffer object.  
            CL_TRUE,      // Indicates if the read operations are blocking or non-blocking.  
            0,            // The offset in bytes in the buffer object to read from.  
            sizeof(int) * hash_table_size,     // The size in bytes of data being read.  
            hash_table,            // The pointer to buffer in host memory where data is to be read into.  
            0,            // num_events_in_wait_list  
            NULL,         // *event_wait_list  
            NULL          // *event  
            ); 
    } 

    //debug code
    if(DEBUG){ 
        if(status != CL_SUCCESS) {
            printf("status ERROR\n");
        }
    }
    return 0;

}
