/**
 *  This is a tool to profile performance counters on GPU 
 * 
 *  It uses the CUDA Profiling Tools Interface (CUPTI) to
 *  periodically fetch GPU performance counters and store them
 *  in a file.
 * 
 *  The goal is to place this tool as a spy application that 
 *  montiors GPU usage of target application (eg - web browser)
 * 
*/

#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <time.h>

using namespace std;

////////////////////////////////////
//				CONFIG			  //
////////////////////////////////////

// Timer //
const int NUM_SECONDS = 1;
const int TIMES = 30;

// List of performance counters to fetch //
char *PC[] = {"sm_efficiency","achieved_occupancy","ipc","issued_ipc","issue_slot_utilization","gld_requested_throughput","gst_requested_throughput","gld_throughput","gst_throughput","tex_cache_throughput","tex_fu_utilization","single_precision_fu_utilization","stall_inst_fetch","stall_exec_dependency","stall_memory_dependency","stall_other","stall_constant_memory_dependency","stall_pipe_busy","stall_memory_throttle","stall_not_selected","l2_read_transactions","l2_tex_read_throughput","l2_tex_write_throughput","l2_read_throughput","l2_write_throughput","dram_utilization","eligible_warps_per_cycle"};
const char PC_COUNT = 27;

// Kernel launch //
const int N = 1024;
const int THREADS = 1024;
const int BLOCKS = 1;

////////////////////////////////////
//			GLOBAL VARS			  //
////////////////////////////////////
size_t f, t;
FILE *ptr;
char data[5000];
static uint64_t kernelDuration;
CUpti_SubscriberHandle subscriber[PC_COUNT];
CUcontext context = 0;
CUdevice device = 0;
int deviceNum = 0;
int deviceCount;
char deviceName[32];
CUpti_MetricID metricId[PC_COUNT];
CUpti_EventGroupSets *passData[PC_COUNT];
CUpti_MetricValue metricValue[PC_COUNT];

////////////////////////////////////
//				HELPERS			  //
////////////////////////////////////

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
	CUresult _status = apiFuncCall;                                            \
	if (_status != CUDA_SUCCESS) {                                             \
		fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
				__FILE__, __LINE__, #apiFuncCall, _status);                    \
		exit(-1);                                                              \
	}                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
	cudaError_t _status = apiFuncCall;                                         \
	if (_status != cudaSuccess) {                                              \
		fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
				__FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
		exit(-1);                                                              \
	}                                                                          \
} while (0)

#define CUPTI_CALL(call)                                                   \
  do {                                                                     \
	CUptiResult _status = call;                                            \
	if (_status != CUPTI_SUCCESS) {                                        \
	  const char *errstr;                                                  \
	  cuptiGetResultString(_status, &errstr);                              \
	  fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
			  __FILE__, __LINE__, #call, errstr);                          \
	  if(_status == CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED)             \
		  exit(0);                                                         \
	  else                                                                 \
		  exit(-1);                                                        \
	}                                                                      \
  } while (0)

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
	(((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

// User data for event collection callback
typedef struct MetricData_st {
	// the device where metric is being collected
	CUdevice device;
	// the set of event groups to collect for a pass
	CUpti_EventGroupSet *eventGroups;
	// the current number of events collected in eventIdArray and
	// eventValueArray
	uint32_t eventIdx;
	// the number of entries in eventIdArray and eventValueArray
	uint32_t numEvents;
	// array of event ids
	CUpti_EventID *eventIdArray;
	// array of event values
	uint64_t *eventValueArray;
} MetricData_t;
MetricData_t metricData[PC_COUNT];

// Device code
__global__ void VecAdd(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

static void
initVec(int *vec, int n)
{
  for (int i=0; i< n; i++)
    vec[i] = i;
}

void CUPTIAPI
getMetricValueCallback(void *userdata, CUpti_CallbackDomain domain,
					   CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
  MetricData_t *metricData = (MetricData_t*)userdata;
  if(metricData->eventIdx >= metricData->numEvents){
	  metricData->eventIdx = 0;
  }

  unsigned int i, j, k;

  // This callback is enabled only for launch so we shouldn't see
  // anything else.
  if ((cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
      (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
  {
    printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
    exit(-1);
  }

  // on entry, enable all the event groups being collected this pass,
  // for metrics we collect for all instances of the event
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
	cudaDeviceSynchronize();

	CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context,
										   CUPTI_EVENT_COLLECTION_MODE_KERNEL));

	for (i = 0; i < metricData->eventGroups->numEventGroups; i++) {
	  uint32_t all = 1;
	  CUPTI_CALL(cuptiEventGroupSetAttribute(metricData->eventGroups->eventGroups[i],
											 CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
											 sizeof(all), &all));
	  CUPTI_CALL(cuptiEventGroupEnable(metricData->eventGroups->eventGroups[i]));
	}
  }

  // on exit, read and record event values
  if (cbInfo->callbackSite == CUPTI_API_EXIT) {
	cudaDeviceSynchronize();

	// for each group, read the event values from the group and record
	// in metricData
	for (i = 0; i < metricData->eventGroups->numEventGroups; i++) {
	  CUpti_EventGroup group = metricData->eventGroups->eventGroups[i];
	  CUpti_EventDomainID groupDomain;
	  uint32_t numEvents, numInstances, numTotalInstances;
	  CUpti_EventID *eventIds;
	  size_t groupDomainSize = sizeof(groupDomain);
	  size_t numEventsSize = sizeof(numEvents);
	  size_t numInstancesSize = sizeof(numInstances);
	  size_t numTotalInstancesSize = sizeof(numTotalInstances);
	  uint64_t *values, normalized, *sum;
	  size_t valuesSize, eventIdsSize;
	  size_t numCountersRead = 0;

	  CUPTI_CALL(cuptiEventGroupGetAttribute(group,
											 CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
											 &groupDomainSize, &groupDomain));
	  CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(metricData->device, groupDomain,
													CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
													&numTotalInstancesSize, &numTotalInstances));
	  CUPTI_CALL(cuptiEventGroupGetAttribute(group,
											 CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
											 &numInstancesSize, &numInstances));
	  CUPTI_CALL(cuptiEventGroupGetAttribute(group,
											 CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
											 &numEventsSize, &numEvents));
	  eventIdsSize = numEvents * sizeof(CUpti_EventID);
	  eventIds = (CUpti_EventID *)malloc(eventIdsSize);
	  CUPTI_CALL(cuptiEventGroupGetAttribute(group,
											 CUPTI_EVENT_GROUP_ATTR_EVENTS,
											 &eventIdsSize, eventIds));

	  valuesSize = sizeof(uint64_t) * numInstances * numEvents;
	  values = (uint64_t *)malloc(valuesSize);

	  CUPTI_CALL(cuptiEventGroupReadAllEvents(group,
										  CUPTI_EVENT_READ_FLAG_NONE,
										  &valuesSize,
										  values,
										  &eventIdsSize,
										  eventIds,
										  &numCountersRead));

	  if (metricData->eventIdx >= metricData->numEvents) {
	  	printf("%d\n", metricData->eventIdx);
		fprintf(stderr, "error: too many events collected, metric expects only %d\n", (int)metricData->numEvents);
		exit(-1);
	  }

	  sum = (uint64_t *)calloc(sizeof(uint64_t), numEvents);
	  // sum collect event values from all instances
	  for (k = 0; k < numInstances; k++) {
		for (j = 0; j < numEvents; j++) {
			sum[j] += values[(k * numEvents) + j];
		}
	  }

	  for (j = 0; j < numEvents; j++) {
		// normalize the event value to represent the total number of
		// domain instances on the device
		normalized = (sum[j] * numTotalInstances) / numInstances;

		metricData->eventIdArray[metricData->eventIdx] = eventIds[j];
		metricData->eventValueArray[metricData->eventIdx] = normalized;
		metricData->eventIdx++;

		// print collected value
		{
		  char eventName[128];
		  size_t eventNameSize = sizeof(eventName) - 1;
		  CUPTI_CALL(cuptiEventGetAttribute(eventIds[j], CUPTI_EVENT_ATTR_NAME,
											&eventNameSize, eventName));
		  eventName[127] = '\0';
		 //  printf("\t%s = %llu (", eventName, (unsigned long long)sum[j]);
		 //  if (numInstances > 1) {
			// for (k = 0; k < numInstances; k++) {
			//   if (k != 0)
			// 	printf(", ");
			//   printf("%llu", (unsigned long long)values[(k * numEvents) + j]);
			// }
		 //  }

		  // printf(")\n");
		  // printf("\t%s (normalized) (%llu * %u) / %u = %llu\n",
				//  eventName, (unsigned long long)sum[j],
				//  numTotalInstances, numInstances,
				//  (unsigned long long)normalized);
		}
	  }

	  free(values);
	  free(sum);
	}

	for (i = 0; i < metricData->eventGroups->numEventGroups; i++)
	  CUPTI_CALL(cuptiEventGroupDisable(metricData->eventGroups->eventGroups[i]));
  }
}

static void
cleanUp(int *h_A, int *h_B, int *h_C, int *d_A, int *d_B, int *d_C)
{
  if (d_A)
    cudaFree(d_A);
  if (d_B)
    cudaFree(d_B);
  if (d_C)
    cudaFree(d_C);

  // Free host memory
  if (h_A)
    free(h_A);
  if (h_B)
    free(h_B);
  if (h_C)
    free(h_C);
}

static void
runPass()
{
  size_t size = N * sizeof(int);
  int threadsPerBlock = 0;
  int blocksPerGrid = 0;
  int *h_A, *h_B, *h_C;
  int *d_A, *d_B, *d_C;
  int i, sum;

  // Allocate input vectors h_A and h_B in host memory
  h_A = (int*)malloc(size);
  h_B = (int*)malloc(size);
  h_C = (int*)malloc(size);

  // Initialize input vectors
  initVec(h_A, N);
  initVec(h_B, N);
  memset(h_C, 0, size);

  // Allocate vectors in device memory
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  // Copy vectors from host memory to device memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Invoke kernel, let them occupy the least amount of space possible,
  // if the profiler has least memory footprint, it is easier to
  // detect performance counters of victim application
  threadsPerBlock = THREADS;
  blocksPerGrid = BLOCKS;
  // printf("Launching kernel: blocks %d, thread/block %d\n", blocksPerGrid, threadsPerBlock);

  VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Verify result
  for (i = 0; i < N; ++i) {
    sum = h_A[i] + h_B[i];
    if (h_C[i] != sum) {
      fprintf(stderr, "error: result verification failed\n");
      exit(-1);
    }
  }

  cleanUp(h_A, h_B, h_C, d_A, d_B, d_C);
}

static void CUPTIAPI
bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *rawBuffer;

  *size = 16 * 1024;
  rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

  *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
  *maxNumRecords = 0;

  if (*buffer == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }
}

static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUpti_Activity *record = NULL;
  CUpti_ActivityKernel4 *kernel;

  //since we launched only 1 kernel, we should have only 1 kernel record
  CUPTI_CALL(cuptiActivityGetNextRecord(buffer, validSize, &record));

  kernel = (CUpti_ActivityKernel4 *)record;
  if (kernel->kind != CUPTI_ACTIVITY_KIND_KERNEL) {
    fprintf(stderr, "Error: expected kernel activity record, got %d\n", (int)kernel->kind);
    exit(-1);
  }

  kernelDuration = kernel->end - kernel->start;
  free(buffer);
}

////////////////////////////////////
//			MAIN FUNCTIONS		  //
////////////////////////////////////

void getMetric(char *metricName, int i, char *metric_data){

	// Subscribe //
	// setup launch callback for event collection //
	CUPTI_CALL(cuptiSubscribe(&subscriber[i], (CUpti_CallbackFunc)getMetricValueCallback, &metricData[i]));
	CUPTI_CALL(cuptiEnableCallback(1, subscriber[i], CUPTI_CB_DOMAIN_RUNTIME_API,
									CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
	CUPTI_CALL(cuptiEnableCallback(1, subscriber[i], CUPTI_CB_DOMAIN_RUNTIME_API,
									CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

	// allocate space to hold all the events needed for the metric
	CUPTI_CALL(cuptiMetricGetIdFromName(device, PC[i], &metricId[i]));
	CUPTI_CALL(cuptiMetricGetNumEvents(metricId[i], &metricData[i].numEvents));
	metricData[i].device = device;
	metricData[i].eventIdArray = (CUpti_EventID *)malloc(metricData[i].numEvents * sizeof(CUpti_EventID));
	metricData[i].eventValueArray = (uint64_t *)malloc(metricData[i].numEvents * sizeof(uint64_t));
	metricData[i].eventIdx = 0;

	// get the number of passes required to collect all the events
	// needed for the metric and the event groups for each pass
	CUPTI_CALL(cuptiMetricCreateEventGroupSets(context, sizeof(metricId[i]), &metricId[i], &passData[i]));
	for (unsigned int pass = 0; pass < passData[i]->numSets; pass++) {
		// printf("Pass %u\n", pass);
		metricData[i].eventGroups = passData[i]->sets + pass;
		runPass();
	}

	if (metricData[i].eventIdx != metricData[i].numEvents) {
		fprintf(stderr, "error: expected %u metric events, got %u\n",
		metricData[i].numEvents, metricData[i].eventIdx);
		exit(-1);
	}

	// use all the collected events to calculate the metric value
	CUPTI_CALL(cuptiMetricGetValue(device, metricId[i],
									metricData[i].numEvents * sizeof(CUpti_EventID),
									metricData[i].eventIdArray,
									metricData[i].numEvents * sizeof(uint64_t),
									metricData[i].eventValueArray,
									kernelDuration, &metricValue[i]));

	CUpti_MetricValueKind valueKind;
	size_t valueKindSize = sizeof(valueKind);
	CUPTI_CALL(cuptiMetricGetAttribute(metricId[i], CUPTI_METRIC_ATTR_VALUE_KIND, &valueKindSize, &valueKind));
	switch (valueKind) {
		case CUPTI_METRIC_VALUE_KIND_DOUBLE:
			sprintf(metric_data, "%f", metricValue[i].metricValueDouble);
			break;
		case CUPTI_METRIC_VALUE_KIND_UINT64:
			sprintf(metric_data, "%llu", (unsigned long long)metricValue[i].metricValueUint64);
			break;
		case CUPTI_METRIC_VALUE_KIND_INT64:
			sprintf(metric_data, "%lld", (long long)metricValue[i].metricValueInt64);
			break;
		case CUPTI_METRIC_VALUE_KIND_PERCENT:
			sprintf(metric_data, "%f", metricValue[i].metricValuePercent);
			break;
		case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
			sprintf(metric_data, "%llu", (unsigned long long)metricValue[i].metricValueThroughput);
			break;
		case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
			sprintf(metric_data, "%u", (unsigned int)metricValue[i].metricValueUtilizationLevel);
			break;
		default:
			fprintf(stderr, "error: unknown value kind\n");
		exit(-1);
	}
	// Unsubscribe //
	CUPTI_CALL(cuptiUnsubscribe(subscriber[i]));
}

int collect_metrics(int count){

	// Open file to record metrics //
	ptr = fopen("result.csv","a");
	if(!ptr)
	{
		printf("file could not be opened\n");
		getchar();
		return -1;
	}

	// Memory Usage //
	cudaMemGetInfo(&f, &t);
	sprintf(data, "%d,%lu,%lu,%lu", count, f, t, (t-f));

	// TODO: Collect other metrics //
	for(int i=0; i<PC_COUNT; i++){
		char *metric_data = (char *) malloc(100 * sizeof(char));
		getMetric(PC[i], i, metric_data);
		sprintf(data, "%s,%s", data, metric_data);
	}
	sprintf(data, "%s\n", data);

	printf("%s", data);
	fputs(data, ptr);
	fclose(ptr);
	return 0;
}

int timer()
{
	int count = 1;

	double time_counter = 0;

	clock_t this_time = clock();
	clock_t last_time = this_time;

	while(count < TIMES)
	{
		this_time = clock();

		time_counter += (double)(this_time - last_time);

		last_time = this_time;

		if(time_counter > (double)(NUM_SECONDS * CLOCKS_PER_SEC))
		{
			time_counter -= (double)(NUM_SECONDS * CLOCKS_PER_SEC);
			collect_metrics(count);
			count++;
		}
	}
	return 0;
}

int main(int argc, char *argv[]){

	// make sure activity is enabled before any CUDA API
	CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

	DRIVER_API_CALL(cuInit(0));
	DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0) {
		printf("There is no device supporting CUDA.\n");
		return -2;
	}
	printf("CUDA Device Number: %d\n", deviceNum);

	DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
	DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, device));
	printf("CUDA Device Name: %s\n", deviceName);

	DRIVER_API_CALL(cuCtxCreate(&context, 0, device));

	// need to collect duration of kernel execution without any event
	// collection enabled (some metrics need kernel duration as part of
	// calculation). The only accurate way to do this is by using the
	// activity API.
	{
		CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
		runPass();
		cudaDeviceSynchronize();
		CUPTI_CALL(cuptiActivityFlushAll(0));
	}

	// Set up output file //
	char header[5000];
	sprintf(header, "count,free,total,used");
	for(int i=0; i<PC_COUNT; i++){
		sprintf(header, "%s,%s", header, PC[i]);
	}
	sprintf(header, "%s\n", header);
	printf("%s", header);

	ptr = fopen("result.csv","w+");
	if(!ptr)
	{
		printf("file could not be opened\n");
		getchar();
		return -1;
	}
	fputs(header, ptr);
	fclose(ptr);

	// Start profiling //
	timer();
	
	return 0;
}
