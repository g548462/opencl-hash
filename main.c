#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>

#define DEBUG 0
#define MSG 0
#define TUPLES 10000000
#define ARRAY_SIZE 10000000
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
    
    int *hash_table, *rehash_table;
    int hash_table_size = 2 * sizeof(int) * TUPLES;
    int *data;
    data = (int*)malloc(sizeof(int) * TUPLES);
    hash_table = (int*)malloc(sizeof(int) * hash_table_size);
    rehash_table = (int*)malloc(sizeof(int) * hash_table_size);
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
    if(MSG) { printf("\t[info] Total %d platforms... \n", numPlatforms); }
    cl_platform_id *platforms = NULL;
    platforms  = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
    status |= clGetPlatformIDs(numPlatforms, platforms, NULL);

    //Get Devices
    //platform[0] is CPU, platform[1] is GPU
    cl_int CPU_numDevices = 0;
    cl_int GPU_numDevices = 0;
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 0, NULL, &CPU_numDevices);
    status = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, 0, NULL, &GPU_numDevices);
    if(MSG) {
        printf("\t[info] Total CPU %d Devices... \n", CPU_numDevices);
        printf("\t[info] Total GPU %d Devices... \n", GPU_numDevices);
    }

    cl_device_id *CPU_devices;
    cl_device_id *GPU_devices;
    CPU_devices = (cl_device_id*)malloc(sizeof(cl_device_id) * CPU_numDevices);
    GPU_devices = (cl_device_id*)malloc(sizeof(cl_device_id) * GPU_numDevices);
    status |= clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, CPU_numDevices, CPU_devices, NULL);
    status |= clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, GPU_numDevices, GPU_devices, NULL);

    size_t cb;
    char *devicename;
    devicename = (char*)malloc(sizeof(char)*100);
    clGetDeviceInfo(CPU_devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
    clGetDeviceInfo(CPU_devices[0], CL_DEVICE_NAME, cb, devicename, 0);
    if(MSG) { printf("DeviceName = %s\n", devicename); }
    clGetDeviceInfo(GPU_devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
    clGetDeviceInfo(GPU_devices[0], CL_DEVICE_NAME, cb, devicename, 0);
    if(MSG) { printf("DeviceName = %s\n", devicename); }

    if(MSG) {
        if(status == CL_SUCCESS) printf("\t[Info] Get DeviceIDs Success!\n");  
        else fprintf(stderr, "\t[Error] Get DeviceIDs Fail!\n");
    }

    cl_context CPU_context;
    cl_context GPU_context;
    CPU_context = clCreateContext(NULL, CPU_numDevices, CPU_devices, NULL, NULL, &status);
    GPU_context = clCreateContext(NULL, GPU_numDevices, GPU_devices, NULL, NULL, &status); 
    if(MSG) {
        if(status == CL_SUCCESS) printf("\t[Info] Create Context Success!\n");  
        else fprintf(stderr, "\t[Error] Create Context Fail!\n");
    }
    cl_command_queue CPU_cmdQueue;  
    cl_command_queue GPU_cmdQueue;
    CPU_cmdQueue = clCreateCommandQueue(CPU_context, CPU_devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
    GPU_cmdQueue = clCreateCommandQueue(GPU_context, GPU_devices[0], CL_QUEUE_PROFILING_ENABLE, &status);  
    if(MSG){
        if(status == CL_SUCCESS) printf("\t[Info] Create CommandQueue Success!\n");  
        else fprintf(stderr, "\t[Error] Create CommandQueue Fail!\n");
    }


    cl_mem CPU_data_buf, CPU_hashtable_buf;
    CPU_hashtable_buf = clCreateBuffer(CPU_context, CL_MEM_READ_WRITE, sizeof(int) * hash_table_size, NULL, &status);
    status = clEnqueueWriteBuffer(CPU_cmdQueue, CPU_hashtable_buf, CL_FALSE, 0, sizeof(int) * hash_table_size, hash_table, 0, NULL, NULL);
    CPU_data_buf = clCreateBuffer(CPU_context, CL_MEM_READ_ONLY, sizeof(int) * TUPLES, NULL, &status);
    status = clEnqueueWriteBuffer(CPU_cmdQueue, CPU_data_buf, CL_FALSE, 0, sizeof(int) * TUPLES, data, 0, NULL, NULL);



    cl_mem GPU_data_buf, GPU_hashtable_buf, GPU_rehashtable_buf;
    GPU_hashtable_buf = clCreateBuffer(GPU_context, CL_MEM_READ_WRITE, sizeof(int) * hash_table_size, NULL, &status);
    status = clEnqueueWriteBuffer(GPU_cmdQueue, GPU_hashtable_buf, CL_FALSE, 0, sizeof(int) * hash_table_size, hash_table, 0, NULL, NULL);
    GPU_data_buf = clCreateBuffer(GPU_context, CL_MEM_READ_ONLY, sizeof(int) * TUPLES, NULL, &status);
    status = clEnqueueWriteBuffer(GPU_cmdQueue, GPU_data_buf, CL_FALSE, 0, sizeof(int) * TUPLES, data, 0, NULL, NULL);
    GPU_rehashtable_buf = clCreateBuffer(GPU_context, CL_MEM_READ_WRITE, sizeof(int) * hash_table_size, NULL, &status);
    status = clEnqueueWriteBuffer(GPU_cmdQueue, GPU_rehashtable_buf, CL_FALSE, 0, sizeof(int) * hash_table_size, rehash_table, 0, NULL, NULL);

    if(MSG) {
        if(status == CL_SUCCESS) printf("\t[Info] CreateBuffer and enQueue Success!\n");  
        else fprintf(stderr, "\t[Error] CreateBuffer and enQueue Fail!\n");
    }

    char *filename = "m_hash.cl";
    char *source_str;
    size_t source_size;
    loadKernel(filename, &source_str, &source_size);

    cl_program CPU_program = clCreateProgramWithSource(CPU_context,   
        1,   
        (const char**)&source_str,   
        NULL,   
        &status);  
    cl_program GPU_program = clCreateProgramWithSource(GPU_context,   
        1,   
        (const char**)&source_str,   
        NULL,   
        &status); 
    if(MSG) {
        if(status == CL_SUCCESS) printf("\t[Info] Create Program Success!\n");  
        else fprintf(stderr, "\t[Error] Create Program Fail!\n"); 
    }
    status = clBuildProgram(CPU_program,        // The program object.  
        CPU_numDevices,                         // The number of devices listed in device_list.  
        CPU_devices,                            // A pointer to a list of devices associated with program.  
        NULL,  
        NULL,  
        NULL  
        );  

    status = clBuildProgram(GPU_program,        // The program object.  
        GPU_numDevices,                         // The number of devices listed in device_list.  
        GPU_devices,                            // A pointer to a list of devices associated with program.  
        NULL,  
        NULL,  
        NULL  
        );  

    if (status != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(CPU_program, CPU_devices[0], CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        printf("Error Build Prog.\n");
        printf("Error code = %d\n", status);
        exit(status);
    }

    if(MSG) {
        if(status == CL_SUCCESS) printf("\t[Info] Build Program Success!\n");  
        else fprintf(stderr, "\t[Error] Build Program Fail! Msg:%d \n", status);  
    }  

    cl_kernel CPU_optKernel;  
    CPU_optKernel = clCreateKernel(CPU_program, "buildHashTableOptimistic", &status); 
    cl_kernel CPU_validKernell;
    CPU_validKernell = clCreateKernel(CPU_program, "validateHashTable", &status);
    cl_kernel CPU_pessimisticKernel;
    CPU_pessimisticKernel = clCreateKernel(CPU_program, "buildHashTablePessimistic", &status);

    cl_kernel GPU_optKernel;  
    GPU_optKernel = clCreateKernel(GPU_program, "buildHashTableOptimistic", &status); 
    cl_kernel GPU_validKernell;
    GPU_validKernell = clCreateKernel(GPU_program, "validateHashTable", &status);
    cl_kernel GPU_pessimisticKernel;
    GPU_pessimisticKernel = clCreateKernel(GPU_program, "buildHashTablePessimistic", &status);

    if(MSG) {
        if(status == CL_SUCCESS) printf("\t[Info] Create kernel Success!\n");  
        else fprintf(stderr, "\t[Error] Create kernel Fail! Msg:%d \n", status);
    }

    size_t globalWorkSize[1];
    globalWorkSize[0] = TUPLES;
    unsigned char err = 0;
    cl_mem CPU_error = NULL;
    cl_mem GPU_error = NULL;
    CPU_error = clCreateBuffer(CPU_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(char), &err, &status);
    GPU_error = clCreateBuffer(GPU_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(char), &err, &status);

    //CPU
    status |= clSetKernelArg(CPU_optKernel, 0, sizeof(cl_mem), &CPU_data_buf); 
    status |= clSetKernelArg(CPU_optKernel, 1, sizeof(cl_mem), &CPU_hashtable_buf);  
    status |= clSetKernelArg(CPU_optKernel, 2, sizeof(int), &hash_table_size); 
    status |= clSetKernelArg(CPU_validKernell, 0, sizeof(cl_mem), &CPU_data_buf);
    status |= clSetKernelArg(CPU_validKernell, 1, sizeof(cl_mem), &CPU_hashtable_buf);
    status |= clSetKernelArg(CPU_validKernell, 2, sizeof(int), &hash_table_size);
    status |= clSetKernelArg(CPU_validKernell, 3, sizeof(cl_mem), &CPU_error);   
    status |= clSetKernelArg(CPU_pessimisticKernel, 0, sizeof(cl_mem), &CPU_data_buf);
    status |= clSetKernelArg(CPU_pessimisticKernel, 1, sizeof(cl_mem), &CPU_hashtable_buf);
    status |= clSetKernelArg(CPU_pessimisticKernel, 2, sizeof(int), &hash_table_size);
    status |= clSetKernelArg(CPU_pessimisticKernel, 3, sizeof(cl_mem), &CPU_error);
    if(MSG) {
        if(status == CL_SUCCESS) printf("\t[Info] CPU Set kernel Argu Success!\n");  
        else fprintf(stderr, "\t[Error] Set kernel Argu Fail! Msg:%d \n", status);
    }

    //GPU
    status |= clSetKernelArg(GPU_optKernel, 0, sizeof(cl_mem), &GPU_data_buf); 
    status |= clSetKernelArg(GPU_optKernel, 1, sizeof(cl_mem), &GPU_hashtable_buf);  
    status |= clSetKernelArg(GPU_optKernel, 2, sizeof(int), &hash_table_size); 
    status |= clSetKernelArg(GPU_validKernell, 0, sizeof(cl_mem), &GPU_data_buf);
    status |= clSetKernelArg(GPU_validKernell, 1, sizeof(cl_mem), &GPU_hashtable_buf);
    status |= clSetKernelArg(GPU_validKernell, 2, sizeof(int), &hash_table_size);
    status |= clSetKernelArg(GPU_validKernell, 3, sizeof(cl_mem), &GPU_error);   
    status |= clSetKernelArg(GPU_pessimisticKernel, 0, sizeof(cl_mem), &GPU_data_buf);
    status |= clSetKernelArg(GPU_pessimisticKernel, 1, sizeof(cl_mem), &GPU_hashtable_buf);
    status |= clSetKernelArg(GPU_pessimisticKernel, 2, sizeof(int), &hash_table_size);
    status |= clSetKernelArg(GPU_pessimisticKernel, 3, sizeof(cl_mem), &GPU_error);

    if(MSG) {
        if(status == CL_SUCCESS) printf("\t[Info] Set kernel Argu Success!\n");  
        else fprintf(stderr, "\t[Error] Set kernel Argu Fail! Msg:%d \n", status);
    }
    cl_event CPU_event[2];
    cl_event GPU_event[2];
    cl_uint num_events_in_wait_list;


    cl_ulong start_time, end_time;
    size_t return_bytes;
    double runtime;

    // Enqueue CPU Start
    status = clEnqueueNDRangeKernel(CPU_cmdQueue,       // A valid command-queue  
        CPU_optKernel,         // A valid kernel object.  
        1,              // work_dim  
        NULL,           // *global_work_offset  
        globalWorkSize, // *global_work_size  
        NULL,           // local_work_size  
        0,              // num_events_in_wait_list  
        NULL,           // *event_wait_list  
        &CPU_event[0]            // *event  
        );  
    
    clWaitForEvents(1, &CPU_event[0]);
    err = clGetEventProfilingInfo(CPU_event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, &return_bytes);
    err = clGetEventProfilingInfo(CPU_event[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);
    runtime = (double)(end_time - start_time);
    printf("\t[Info] CPU Hash1 Run time = %fms\n", runtime/1000000);

    if(MSG) {
        if(status == CL_SUCCESS) printf("\t[Info] enQueue kernel Success!\n");  
        else fprintf(stderr, "\t[Error] enQueue kernel Fail! Msg:%d \n", status);
    }
    // Enqueue CPU End

    //Enqueue GPU Start
    status = clEnqueueNDRangeKernel(GPU_cmdQueue,       // A valid command-queue  
        GPU_optKernel,         // A valid kernel object.  
        1,              // work_dim  
        NULL,           // *global_work_offset  
        globalWorkSize, // *global_work_size  
        NULL,           // local_work_size  
        0,              // num_events_in_wait_list  
        NULL,           // *event_wait_list  
        &GPU_event[0]            // *event  
        );  
    
    clWaitForEvents(1, &GPU_event[0]);
    err = clGetEventProfilingInfo(GPU_event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, &return_bytes);
    err = clGetEventProfilingInfo(GPU_event[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);
    runtime = (double)(end_time - start_time);
    printf("\t[Info] GPU Hash1 Run time = %fms\n", runtime/1000000);
    //ENQUEUE GPU End




    //  Read build HT result
    status = clEnqueueReadBuffer(CPU_cmdQueue,     // Refers to the command-queue in which the read command will be queued  
        CPU_hashtable_buf,         // Refers to a valid buffer object.  
        CL_TRUE,      // Indicates if the read operations are blocking or non-blocking.  
        0,            // The offset in bytes in the buffer object to read from.  
        sizeof(int) * hash_table_size,     // The size in bytes of data being read.  
        hash_table,            // The pointer to buffer in host memory where data is to be read into.  
        0,            // num_events_in_wait_list  
        NULL,         // *event_wait_list  
        NULL          // *event  
        ); 

    //  CPU validate kernel
    status = clEnqueueNDRangeKernel(CPU_cmdQueue,       // A valid command-queue  
        CPU_validKernell,         // A valid kernel object.  
        1,              // work_dim  
        NULL,           // *global_work_offset  
        globalWorkSize, // *global_work_size  
        NULL,           // local_work_size  
        0,              // num_events_in_wait_list  
        NULL,           // *event_wait_list  
        &CPU_event[1]            // *event  
        );  
    clWaitForEvents(1, &CPU_event[1]);
    err = clGetEventProfilingInfo(CPU_event[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, &return_bytes);
    err = clGetEventProfilingInfo(CPU_event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);
    runtime = (double)(end_time - start_time);
    printf("\t[Info] CPU Validate Run time = %fms\n", runtime/1000000);

    //  GPU validate kernel
    status = clEnqueueNDRangeKernel(GPU_cmdQueue,       // A valid command-queue  
        GPU_validKernell,         // A valid kernel object.  
        1,              // work_dim  
        NULL,           // *global_work_offset  
        globalWorkSize, // *global_work_size  
        NULL,           // local_work_size  
        0,              // num_events_in_wait_list  
        NULL,           // *event_wait_list  
        &GPU_event[1]            // *event  
        );  
    clWaitForEvents(1, &GPU_event[1]);
    err = clGetEventProfilingInfo(GPU_event[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, &return_bytes);
    err = clGetEventProfilingInfo(GPU_event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);
    runtime = (double)(end_time - start_time);
    printf("\t[Info] GPU Validate Run time = %fms\n", runtime/1000000);
    //  GPU validate kernel End


    //  Return validate value
    status = clEnqueueReadBuffer(CPU_cmdQueue,     // Refers to the command-queue in which the read command will be queued  
        CPU_error,         // Refers to a valid buffer object.  
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
        status = clEnqueueNDRangeKernel(CPU_cmdQueue,       // A valid command-queue  
            CPU_pessimisticKernel,         // A valid kernel object.  
            1,              // work_dim  
            NULL,           // *global_work_offset  
            globalWorkSize, // *global_work_size  
            NULL,           // local_work_size  
            0,              // num_events_in_wait_list  
            NULL,           // *event_wait_list  
            &CPU_event[1]            // *event  
            );  
        if(MSG) {
            if(status == CL_SUCCESS) printf("\t[Info] enQueue Pessimistic Success!\n");  
            else fprintf(stderr, "\t[Error] enQueue Pessimistic Fail! Msg:%d \n", status);
        }

        //  Get runtime info
        clWaitForEvents(1, &CPU_event[1]);
        err = clGetEventProfilingInfo(CPU_event[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, &return_bytes);
        err = clGetEventProfilingInfo(CPU_event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);
        runtime = (double)(end_time - start_time);
        printf("\t[Info] CPU Persimisstic Run time = %fms\n", runtime/1000000);

        status = clEnqueueNDRangeKernel(GPU_cmdQueue,       // A valid command-queue  
            GPU_pessimisticKernel,         // A valid kernel object.  
            1,              // work_dim  
            NULL,           // *global_work_offset  
            globalWorkSize, // *global_work_size  
            NULL,           // local_work_size  
            0,              // num_events_in_wait_list  
            NULL,           // *event_wait_list  
            &GPU_event[1]            // *event  
            );  
        clWaitForEvents(1, &GPU_event[1]);
        err = clGetEventProfilingInfo(GPU_event[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, &return_bytes);
        err = clGetEventProfilingInfo(GPU_event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);
        runtime = (double)(end_time - start_time);
        printf("\t[Info] GPU Persimisstic Run time = %fms\n", runtime/1000000);



        status = clEnqueueReadBuffer(CPU_cmdQueue,     // Refers to the command-queue in which the read command will be queued  
            CPU_hashtable_buf,         // Refers to a valid buffer object.  
            CL_TRUE,      // Indicates if the read operations are blocking or non-blocking.  
            0,            // The offset in bytes in the buffer object to read from.  
            sizeof(int) * hash_table_size,     // The size in bytes of data being read.  
            hash_table,            // The pointer to buffer in host memory where data is to be read into.  
            0,            // num_events_in_wait_list  
            NULL,         // *event_wait_list  
            &CPU_event[1]          // *event  
            ); 
        clWaitForEvents(1, &CPU_event[1]);
        int *hashT;
        hashT = (int*)malloc(sizeof(int) * hash_table_size);

        status = clEnqueueReadBuffer(GPU_cmdQueue,     // Refers to the command-queue in which the read command will be queued  
            GPU_hashtable_buf,         // Refers to a valid buffer object.  
            CL_TRUE,      // Indicates if the read operations are blocking or non-blocking.  
            0,            // The offset in bytes in the buffer object to read from.  
            sizeof(int) * hash_table_size,     // The size in bytes of data being read.  
            hashT,            // The pointer to buffer in host memory where data is to be read into.  
            0,            // num_events_in_wait_list  
            NULL,         // *event_wait_list  
            &GPU_event[1]          // *event  
            );
        clWaitForEvents(1, &GPU_event[1]);

        if(DEBUG) {
            for(i=0; i<hash_table_size; i++) {
                printf("GPU [%d]\t%d\n",i,hashT[i]);
            }

            // Validate CPU and GPU hash result
            for(i=0; i<hash_table_size; i++)
            {
                if(hash_table[i] != hashT[i]) {
                    printf("DONT equal! CPU: %d\tGPU: %d\n",hash_table[i], hashT[i]);
                }

            }
        }



    } 

    //debug code
    if(DEBUG){ 
        for(i=0; i<hash_table_size; i++) {
            printf("[%d]\t%d\n",i,hash_table[i]);
        }
        if(status != CL_SUCCESS) {
            printf("status ERROR\n");
        }
    }
    return 0;

}
