#include <windows.h>
#include <assert.h>
#include "resize.h"

#define MAX_MODALITIES				5				// traj, hog, hof, mbhx, mbhy
#define MAX_DESCRIPTOR_TYPES		MAX_MODALITIES	// for backwards compatibility
#define SPATIO_TEMPORAL_INFO_LEN	10
#define FULL_DESCRIPTOR_LEN			426				// len of all modalities concatenated
#define PCA_SCALE					2
#define TOTAL_GMM_CLUSTERS			256
#define MAX_DESCRIPTOR_LEN			108				// length of largest feature in g_rgFeatureDims

#define MAX_TRACK_LENGTH			(15+1)
#define NBINS_8						8
#define NBINS_9						9
#define MAX_BINS					9 // max(NBINS_8, NBINS_9)
#define MAX_FRAME_WIDTH				512
#define MAX_FRAME_HEIGHT			256
#define MAX_SCALES					6
#define TRACK_STRIDE				5	// pixels between tracks

#define HOG_DESC					0
#define HOF_DESC					1
#define MBHy_DESC					2
#define MBHx_DESC					3


#define DESCRIPTOR_STRIDE			128	 // next multiple of 128 to largest descriptor len (for memory cache alignment)
#define VOCABULARY_STRIDE			4096 // next multiple of 128 to vocaulary len (for memory cache alignment)
#define MAX_DESCRIPTORS_PER_FRAME	500  // maximum number of valid tracks expected to 'mature' per frame (per scale) - usually in the teens but large scene changes can kick this up to ~400 in tests
#define MAX_ACTIVITY_FRAME_COUNT	256	 // maximum number of frames an activity is expected to span (~8 sec @ 30fps)
#define MAX_TEMPORAL_WINDOWS		5	 // maximum different activity frame counts
#define MAX_RECOGNIZED_ACTIVITIES	500	 // maximum different activites supported
#define MAX_TRAINING_SET_ROWS		10000

#define MAX_KSIZE					13	// so far 13 is largest seen
#define MAGIC_FIVE					5


typedef struct TAG_OPTICAL_FLOW_SCRATCH
{
	float rgPrevFlow[MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * 2];
	float rgTemp[MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT];
	float rgTempx[MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT];
	float rgTemp0[MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * MAGIC_FIVE];
	float rgM[MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * MAGIC_FIVE];
	float rgFlowScratch[MAX_SCALES][MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * 2];

	float rgRowScratch[MAX_FRAME_HEIGHT * MAX_FRAME_WIDTH * 8]; // *4 because PolyExp requires 3x width and there is an iPoly_n * 3 augmentation
	float rgVSumScratch[MAX_FRAME_HEIGHT * MAX_FRAME_WIDTH * 6]; // *5 because UpdateFlow requires 5x width and there is an (m+1)*5 augmentation
	float rgHSumScratch[MAX_FRAME_HEIGHT * MAX_FRAME_WIDTH * 5]; // *5 because UpdateFlow requires 5x width
	float* rgSRowScratch[MAX_FRAME_HEIGHT * 5];
}OPTICAL_FLOW_SCRATCH;

typedef struct TAG_GEN_DENSETRAJ_BUFFERS
{
	float rgFlow_y[MAX_SCALES][MAX_FRAME_HEIGHT * MAX_FRAME_WIDTH];
	float rgFlow_x[MAX_SCALES][MAX_FRAME_HEIGHT * MAX_FRAME_WIDTH];
	OPTICAL_FLOW_SCRATCH ofs;
}GEN_DENSETRAJ_BUFFERS;





bool g_fTexInitialized = false;
int g_iWidthStride;


float Round(float f)
{
	return floor(f + 0.5f); // OPTOPT
}


__global__ void InitializeScratchMemory(OPTICAL_FLOW_SCRATCH* pOFScratch, int iHeight, int iWidth, int iWidthStride, int iScale);
__global__ void PolyExp_a(float* pfSrc, float* pfScratch, int iHeight, int iWidth, int iWidthStride, int iPoly_n, float fPoly_sigma);
__global__ void PolyExp_ax(float* pfSrc, float* pfScratch, int iHeight, int iWidth, int iWidthStride, int iPoly_n, float fPoly_sigma);
__global__ void PolyExp_b(float* pfSrc, float* pfDst, float* pfScratch, int iHeight, int iWidth, int iWidthStride, int iPoly_n, float fPoly_sigma);


__global__ void UpdateMatrices(float* pfScratch1, float* pfScratch2, float* pfFlow, float* pfM, int iHeight, int iWidth, int iWidthStride, int iWindowHeight, int iOffset_y);
__global__ void UpdateMatrices(cudaTextureObject_t texObjTemp1, float* pfScratch1, float* pfScratch2, float* pfFlow, float* pfM, int iHeight, int iWidth, int iWidthStride, int iWindowHeight, int iOffset_y);

__global__ void UpdateFlow_a(OPTICAL_FLOW_SCRATCH* pOFScratch, int iHeight, int iWidth, int iWidthStride);
__global__ void UpdateFlow_b(float* pfVSumScratch, float** pfSRowScratch, int iHeight, int iWidth, int iWidthStride, int iWindowHeight, int iOffset_y);
__global__ void UpdateFlow_c(float* pfVSumScratch, int iHeight, int iWidth, int iWidthStride, int iWindowHeight, int iOffset_y);
__global__ void UpdateFlow_d(float* pfVSumScratch, float* pfHSumScratch, int iHeight, int iWidth, int iWidthStride, int iWindowHeight, int iOffset_y);
__global__ void UpdateFlow_e(float* pfFlow, float* pfHSumScratch, int iHeight, int iWidth, int iWidthStride, int iWindowHeight, int iOffset_y);


__global__ void MoveFlow(float* pfPrevFlow, float* pfFlowScratch, int iHeight, int iWidth, int iWidthStride);
__global__ void MoveFlow(GEN_DENSETRAJ_BUFFERS* pDenseTrajBuffers, OPTICAL_FLOW_SCRATCH* pOFScratch, int iHeight, int iWidth, int iWidthStride, int iScale); // OPTOPT this splits flowscratch to flow_x and flow_y, need to get rid of this an generate flow into flow_x and flow_y
__global__ void ScaleFlow(OPTICAL_FLOW_SCRATCH* pOFScratch, int iHeight, int iWidth, int iWidthStride, int iScale);

__global__ void ScaleFlow(OPTICAL_FLOW_SCRATCH* pOFScratch, int iHeight, int iWidth, int iWidthStride, int iScale);

__global__ void ComputeGaussianBlur(unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iSrcWidthStep, int iDstWidthStep, float fSigma, int ksize);

void UpdateFlowAndMatrices(OPTICAL_FLOW_SCRATCH* pOFScratch, float* pfTemp0, float* pfTemp1, int iHeight, int iWidth, int iWidthStride, int iScale);
void UpdateFlowAndMatrices(cudaTextureObject_t texObjTemp1, OPTICAL_FLOW_SCRATCH* pOFScratch, float* pfTemp0, float* pfTemp1, int iHeight, int iWidth, int iWidthStride, int iScale);


void PseudoCuResize2(void* pvFrame, int iHeight, int iWidth, void* pvScaledFrame, int iScaledHeight, int iScaledWidth, int iWidthStride);
void PseudoCuResize(void* pvFrame, int iHeight, int iWidth, void* pvScaledFrame, int iScaledHeight, int iScaledWidth, int iWidthStride);


texture <float, 2, cudaReadModeElementType> texTemp1;

void* pvTemp1;

void InitOpticalFlowTextures(int iHeight, int iWidth, int* piWidthStride)
{
	size_t pitch;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaMallocPitch(&pvTemp1, &pitch, sizeof(float) * MAX_FRAME_WIDTH, MAX_FRAME_HEIGHT * MAGIC_FIVE);

	texTemp1.filterMode = cudaFilterModePoint;

	cudaBindTexture2D(0, texTemp1, pvTemp1, channelDesc, iWidth, iHeight * MAGIC_FIVE, pitch);

	*piWidthStride = pitch / sizeof(float);
}

void FreeOpticalFlowTextures()
{
	cudaUnbindTexture(texTemp1);
	cudaFree(pvTemp1);
}

float rgScale[6];
float rgSigma[6];
int rgSmoothSz[6];

void CuOpticalFlowFarneback(void* pvPrev, void* pvCurr, void* pvDenseTrajBuffers, void* pvOFScratch, int iHeight, int iWidth, int iWidthStep, int iScale, float* pfFlow_y, float* pfFlow_x)
{
	int i;
	int k;
	int iScaledHeight;
	int iScaledWidth;
	int iTotalLevels;
	int iPrevScaledHeight;
	int iPrevScaledWidth;
	float fPyrScale;
	GEN_DENSETRAJ_BUFFERS* pDenseTrajBuffers;
	OPTICAL_FLOW_SCRATCH* pOFScratch;
	const int min_size = 32;
	
	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = 6;
	dimBlock.x = 32;

	fPyrScale = sqrt(2.0f) / 2.0f;


	if (!g_fTexInitialized)
	{
		InitOpticalFlowTextures(MAX_FRAME_HEIGHT, MAX_FRAME_WIDTH, &g_iWidthStride);
		g_fTexInitialized = true;
	}

	//cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );


	dimGrid.x = (int)ceil((float)iWidth / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)iHeight / (float)dimBlock.y);

	InitializeScratchMemory << <dimGrid, dimBlock >> >((OPTICAL_FLOW_SCRATCH*)pvOFScratch, iHeight, iWidth, g_iWidthStride, iScale);


	iTotalLevels = 1;
	float fScale = 1.0f;
	for (k = 0; k < iTotalLevels; k++)
	{
		fScale *= fPyrScale;
		if (iWidth * fScale < min_size || iHeight * fScale < min_size)
		{
			break;
		}
	}

	iTotalLevels = k;

	for (k = iTotalLevels; k >= 0; k--)
	{
		fScale = 1;
		for (i = 0; i < k; i++)
		{
			fScale *= fPyrScale;
		}

		rgSigma[k] = (1.0f / fScale - 1.0f) * 0.5f;
		rgSmoothSz[k] = (int)Round(rgSigma[k] * 5) | 1;
		rgSmoothSz[k] = max(rgSmoothSz[k], 3);
		rgScale[k] = fScale;
	}


	pOFScratch = (OPTICAL_FLOW_SCRATCH*)pvOFScratch;

	for (i = iTotalLevels; i >= 0; i--)
	{
		iScaledHeight = (int)Round(iHeight * rgScale[i]);
		iScaledWidth = (int)Round(iWidth * rgScale[i]);

		if (i != iTotalLevels)
		{
			//PseudoCuResize2( pOFScratch->rgPrevFlow, iPrevScaledHeight, iPrevScaledWidth, pOFScratch->rgFlowScratch, iScaledHeight, iScaledWidth, g_iWidthStride );
			ResizeFermiPF(pOFScratch->rgPrevFlow, pOFScratch->rgFlowScratch, iPrevScaledHeight, iPrevScaledWidth, g_iWidthStride * 4, iScaledHeight, iScaledWidth, g_iWidthStride * 4, 2);


			dimGrid.x = (int)ceil((float)iScaledWidth / (float)dimBlock.x);
			dimGrid.y = (int)ceil((float)(iScaledHeight * 2) / (float)dimBlock.y);

			ScaleFlow << <dimGrid, dimBlock >> >(pOFScratch, iScaledHeight, iScaledWidth, g_iWidthStride, iScale);
		}


		dimGrid.x = (int)ceil((float)iWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iHeight / (float)dimBlock.y);

		assert((rgSmoothSz[i] * rgSmoothSz[i]) <= (dimBlock.x * dimBlock.y)); // need this for "if( iBlockThreadId < ksize * ksize )" statement in ComputeGaussianBlur
		ComputeGaussianBlur << <dimGrid, dimBlock >> >((unsigned char*)pvPrev, pOFScratch->rgTempx, iHeight, iWidth, iWidthStep, g_iWidthStride, rgSigma[i], rgSmoothSz[i]);


		//PseudoCuResize( pOFScratch->rgTempx, iHeight, iWidth, pOFScratch->rgTemp, iScaledHeight, iScaledWidth, g_iWidthStride );
		ResizeFermiPF(pOFScratch->rgTempx, pOFScratch->rgTemp, iHeight, iWidth, g_iWidthStride * 4, iScaledHeight, iScaledWidth, g_iWidthStride * 4, 1);


		dimGrid.x = (int)ceil((float)iScaledWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iScaledHeight / (float)dimBlock.y);

		PolyExp_a << <dimGrid, dimBlock >> >(pOFScratch->rgTemp, pOFScratch->rgRowScratch, iScaledHeight, iScaledWidth, g_iWidthStride, 7, 1.5f);


		dimGrid.x = 1;  // NOTE: ok as long as poly_n < dimBlock.x
		dimGrid.y = (int)ceil((float)iScaledHeight / (float)dimBlock.y);
		PolyExp_ax << <dimGrid, dimBlock >> >(pOFScratch->rgTemp, pOFScratch->rgRowScratch, iScaledHeight, iScaledWidth, g_iWidthStride, 7, 1.5f);



		dimGrid.x = (int)ceil((float)iScaledWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iScaledHeight / (float)dimBlock.y);
		PolyExp_b << <dimGrid, dimBlock >> >(pOFScratch->rgTemp, pOFScratch->rgTemp0, pOFScratch->rgRowScratch, iScaledHeight, iScaledWidth, g_iWidthStride, 7, 1.5f);


		dimGrid.x = (int)ceil((float)iWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iHeight / (float)dimBlock.y);

		assert((rgSmoothSz[i] * rgSmoothSz[i]) <= (dimBlock.x * dimBlock.y)); // need this for "if( iBlockThreadId < ksize * ksize )" statement in ComputeGaussianBlur
		ComputeGaussianBlur << <dimGrid, dimBlock >> >((unsigned char*)pvCurr, pOFScratch->rgTempx, iHeight, iWidth, iWidthStep, g_iWidthStride, rgSigma[i], rgSmoothSz[i]);

		//PseudoCuResize( ((OPTICAL_FLOW_SCRATCH*)pvOFScratch)->rgTempx, iHeight, iWidth, pOFScratch->rgTemp, iScaledHeight, iScaledWidth, g_iWidthStride );
		ResizeFermiPF(pOFScratch->rgTempx, pOFScratch->rgTemp, iHeight, iWidth, g_iWidthStride * 4, iScaledHeight, iScaledWidth, g_iWidthStride * 4, 1);

		dimGrid.x = (int)ceil((float)iScaledWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iScaledHeight / (float)dimBlock.y);

		PolyExp_a << <dimGrid, dimBlock >> >(pOFScratch->rgTemp, pOFScratch->rgRowScratch, iScaledHeight, iScaledWidth, g_iWidthStride, 7, 1.5f); // for now these are the only params for iPoly_n and fPoly_sigma supported

		dimGrid.x = 1; // NOTE: ok as long as poly_n < dimBlock.x
		dimGrid.y = (int)ceil((float)iScaledHeight / (float)dimBlock.y);
		PolyExp_ax << <dimGrid, dimBlock >> >(pOFScratch->rgTemp, pOFScratch->rgRowScratch, iScaledHeight, iScaledWidth, g_iWidthStride, 7, 1.5f);


		dimGrid.x = (int)ceil((float)iScaledWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iScaledHeight / (float)dimBlock.y);
		PolyExp_b << <dimGrid, dimBlock >> >(pOFScratch->rgTemp, (float*)pvTemp1, pOFScratch->rgRowScratch, iScaledHeight, iScaledWidth, g_iWidthStride, 7, 1.5f);


		UpdateMatrices << <dimGrid, dimBlock >> >(pOFScratch->rgTemp0, (float*)pvTemp1, pOFScratch->rgFlowScratch[iScale], pOFScratch->rgM, iScaledHeight, iScaledWidth, g_iWidthStride, iScaledHeight, 0);


		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		dimGrid.x = min(65535, (int)ceil((float)iScaledHeight / (float)(dimBlock.x * dimBlock.y)));
		dimGrid.y = (int)ceil((float)iScaledHeight / (float)(dimBlock.x * dimBlock.y * dimGrid.x));

		UpdateFlow_a << <dimGrid, dimBlock >> >((OPTICAL_FLOW_SCRATCH*)pvOFScratch, iScaledHeight, iScaledWidth, g_iWidthStride);

		//-------------------------------------------------------------------------------------------------------------
		// FarnebackUpdateFlow_GaussianBlur( ..., true )
		//-------------------------------------------------------------------------------------------------------------
		UpdateFlowAndMatrices(pOFScratch, pOFScratch->rgTemp0, (float*)pvTemp1, iScaledHeight, iScaledWidth, g_iWidthStride, iScale);


		//-------------------------------------------------------------------------------------------------------------
		// FarnebackUpdateFlow_GaussianBlur( ..., false )
		//-------------------------------------------------------------------------------------------------------------

		dimGrid.x = (int)ceil((float)iScaledWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iScaledHeight / (float)dimBlock.y);

		UpdateFlow_b << <dimGrid, dimBlock >> >(pOFScratch->rgVSumScratch, pOFScratch->rgSRowScratch, iScaledHeight, iScaledWidth, g_iWidthStride, iScaledHeight, 0);


		dimGrid.x = 1; // NOTE: ok as long as MAGIC_FIVE < dimBlock.x
		dimGrid.y = (int)ceil((float)iScaledHeight / (float)dimBlock.y);

		UpdateFlow_c << <dimGrid, dimBlock >> >(pOFScratch->rgVSumScratch, iScaledHeight, iScaledWidth, g_iWidthStride, iScaledHeight, 0);


		dimGrid.x = (int)ceil((float)iScaledWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iScaledHeight / (float)dimBlock.y);

		UpdateFlow_d << <dimGrid, dimBlock >> >(pOFScratch->rgVSumScratch, pOFScratch->rgHSumScratch, iScaledHeight, iScaledWidth, g_iWidthStride, iScaledHeight, 0);

		UpdateFlow_e << <dimGrid, dimBlock >> >(pOFScratch->rgFlowScratch[iScale], pOFScratch->rgHSumScratch, iScaledHeight, iScaledWidth, g_iWidthStride, iScaledHeight, 0);


		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		if (i > 0)
		{
			dimGrid.x = (int)ceil((float)iScaledWidth / (float)dimBlock.x);
			dimGrid.y = (int)ceil((float)(iScaledHeight * 2) / (float)dimBlock.y);


			MoveFlow << <dimGrid, dimBlock >> >(pOFScratch->rgPrevFlow, pOFScratch->rgFlowScratch[iScale], iScaledHeight, iScaledWidth, g_iWidthStride);

			iPrevScaledHeight = iScaledHeight;
			iPrevScaledWidth = iScaledWidth;
		}
		else
		{
			dimGrid.x = (int)ceil((float)iWidth / (float)dimBlock.x);
			dimGrid.y = (int)ceil((float)(iHeight) / (float)dimBlock.y);

			MoveFlow << <dimGrid, dimBlock >> >((GEN_DENSETRAJ_BUFFERS*)pvDenseTrajBuffers, pOFScratch, iHeight, iWidth, g_iWidthStride, iScale);
			break;
		}
	}

	pDenseTrajBuffers = (GEN_DENSETRAJ_BUFFERS*)pvDenseTrajBuffers;

	cudaMemcpy2D(pfFlow_x, sizeof(float) * iWidth, pDenseTrajBuffers->rgFlow_x[iScale], sizeof(float) * g_iWidthStride, sizeof(float) * iWidth, iHeight, cudaMemcpyDeviceToDevice);
	cudaMemcpy2D(pfFlow_y, sizeof(float) * iWidth, pDenseTrajBuffers->rgFlow_y[iScale], sizeof(float) * g_iWidthStride, sizeof(float) * iWidth, iHeight, cudaMemcpyDeviceToDevice);

}



__constant__  __device__ float ig11 = 0.44444890219447092f;
__constant__  __device__ float ig03 = -0.22225068649192234f;
__constant__  __device__ float ig33 = 0.098779072163238141f;
__constant__  __device__ float ig55 = 0.19753482374208653f;

__constant__  __device__ float kbuf[] =
{
	4.9640303e-006, 8.9220186e-005, 0.0010281866f, 0.0075973268f, 0.035993990f,
	0.10934009f, 0.21296541f, 0.26596162f, 0.21296541f, 0.10934009f,
	0.035993990f, 0.0075973268f, 0.0010281866f, 8.9220186e-005, 4.9640303e-006,
	-3.4748213e-005, -0.00053532113f, -0.0051409332f, -0.030389307f, -0.10798196f,
	-0.21868017f, -0.21296541f, 0.00000000f, 0.21296541f, 0.21868017f,
	0.10798196f, 0.030389307f, 0.0051409332f, 0.00053532113f, 3.4748213e-005,
	0.00024323749f, 0.0032119267f, 0.025704667f, 0.12155723f, 0.32394591f,
	0.43736035f, 0.21296541f, 0.00000000f, 0.21296541f, 0.43736035f,
	0.32394591f, 0.12155723f, 0.025704667f, 0.0032119267f, 0.00024323749f
};

__global__ void InitializeScratchMemory(OPTICAL_FLOW_SCRATCH* pOFScratch, int iHeight, int iWidth, int iWidthStride, int iScale)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = y * iWidthStride + x;


	if (x < iWidth && y < iHeight)
	{
		pOFScratch->rgFlowScratch[iScale][i] = 0;
		pOFScratch->rgFlowScratch[iScale][i + iHeight * iWidthStride] = 0;
	}
}


__global__ void ScaleFlow(OPTICAL_FLOW_SCRATCH* pOFScratch, int iHeight, int iWidth, int iWidthStride, int iScale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y >= (iHeight * 2) || x >= iWidth)
	{
		return;
	}

	pOFScratch->rgFlowScratch[iScale][y * iWidthStride + x] *= 1.4142135623730949f; // 1.0/(sqrt(2.0)/2.0)
}


__device__ float rgFilter3[] = { 0.25f, 0.50f, 0.25f };
__device__ float rgFilter5[] = { 0.06250f, 0.25000f, 0.37500f, 0.25000f, 0.06250f };


__global__ void ComputeGaussianBlur(unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iSrcWidthStep, int iDstWidthStep, float fSigma, int ksize)
{
	int j;
	int filter_x;
	int filter_y;
	float fScale;
	float fVal;

	int y_valid;
	int x_valid;
	int iOffset;

	int iBlockThreadId;

	__shared__ float rgFilter[MAX_KSIZE];
	__shared__ float rgFilter2[MAX_KSIZE*MAX_KSIZE];
	__shared__ float fSum;

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = y * iDstWidthStep + x;

	iBlockThreadId = threadIdx.y * blockDim.x + threadIdx.x; // thread id within block

	if (iBlockThreadId < ksize)
	{
		if (fSigma)
		{
			fScale = -0.5f / (fSigma * fSigma);
			fVal = iBlockThreadId - (ksize - 1) * 0.5f;

			rgFilter[iBlockThreadId] = expf(fScale * fVal * fVal);  // OPTOPT can this be pre computed?
		}
		else
		{
			if (ksize == 3)
			{
				rgFilter[iBlockThreadId] = rgFilter3[iBlockThreadId];
			}
			else
			{
				if (ksize == 5)
				{
					rgFilter[iBlockThreadId] = rgFilter5[iBlockThreadId];
				}
			}

		}
	}
	__syncthreads();

	if (iBlockThreadId == 0)
	{
		fVal = rgFilter[iBlockThreadId];
		for (j = 1; j < ksize; j++)
		{
			fVal += rgFilter[j];
		}

		fSum = fVal;
	}
	__syncthreads();

	if (iBlockThreadId < ksize)
	{
		rgFilter[iBlockThreadId] = rgFilter[iBlockThreadId] / fSum;
	}
	__syncthreads();

	if (iBlockThreadId < ksize * ksize)
	{
		filter_y = iBlockThreadId / ksize;
		filter_x = iBlockThreadId%ksize;

		rgFilter2[filter_y * ksize + filter_x] = rgFilter[filter_y] * rgFilter[filter_x];
	}

	__syncthreads();

	if (y >= iHeight || x >= iWidth)
	{
		return;
	}

	iOffset = ksize >> 1;

	fVal = 0;

	for (filter_y = 0; filter_y < ksize; filter_y++)
	{
		for (filter_x = 0; filter_x < ksize; filter_x++)
		{
			x_valid = (filter_x + (x - iOffset));
			y_valid = (filter_y + (y - iOffset));

			if (x_valid < 0)  //BORDER_REFLECT_101
			{
				x_valid = -x_valid;
			}

			if (y_valid < 0)
			{
				y_valid = -y_valid;
			}


			if (x_valid >= iWidth)
			{
				x_valid = (iWidth - 1) - (x_valid - (iWidth - 1));
			}

			if (y_valid >= iHeight)
			{
				y_valid = (iHeight - 1) - (y_valid - (iHeight - 1));
			}

			fVal += pchSrc[y_valid * iSrcWidthStep + x_valid] * rgFilter2[filter_y * ksize + filter_x];

		}
	}

	pfDst[i] = fVal;
}



__global__ void PolyExp_a(float* pfSrc, float* pfScratch, int iHeight, int iWidth, int iWidthStride, int iPoly_n, float fPoly_sigma)
{
	float g0;
	float g1;
	float g2;
	float *srow0;
	float *srow1;
	float* g;
	float* xg;
	float* xxg;
	float p;
	float t0;
	float t1;
	float t2;

	float* row0;
	float* row1;
	float* row2;

	int k;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= iWidth || y >= iHeight)
	{
		return;
	}


	g = kbuf + iPoly_n;
	xg = g + iPoly_n * 2 + 1;
	xxg = xg + iPoly_n * 2 + 1;


	g0 = g[0];
	srow0 = pfSrc + iWidthStride * y;
	//row = pfScratch + y * iWidth * 4 + iPoly_n * 3; // OPTOPT not sure if using too much memory but (*4) used in order to clear (iPoly_n * 3) and prevent overwrites. Anything less yields overwrites
	row0 = pfScratch + y * (iWidth + iPoly_n * 2) + iPoly_n;
	row1 = row0 + iHeight * (iWidth + iPoly_n * 2);
	row2 = row0 + iHeight * (iWidth + iPoly_n * 2) * 2;


	//row[x * 3] = srow0[x] * g0;
	//row[x * 3 + 1] = 0.0f;
	//row[x * 3 + 2] = 0.0f;
	row0[x] = srow0[x] * g0;
	row1[x] = 0.0f;
	row2[x] = 0.0f;

	for (k = 1; k <= iPoly_n; k++)
	{
		g0 = g[k];
		g1 = xg[k];
		g2 = xxg[k];

		srow0 = (float*)(pfSrc + iWidthStride * max(y - k, 0));
		srow1 = (float*)(pfSrc + iWidthStride * min(y + k, iHeight - 1));

		p = srow0[x] + srow1[x];
		//t0 = row[x*3] + g0*p;
		//t1 = row[x*3+1] + g1*(srow1[x] - srow0[x]);
		//t2 = row[x*3+2] + g2*p;
		t0 = row0[x] + g0*p;
		t1 = row1[x] + g1*(srow1[x] - srow0[x]);
		t2 = row2[x] + g2*p;

		//row[x*3] = t0;
		//row[x*3+1] = t1;
		//row[x*3+2] = t2;
		row0[x] = t0;
		row1[x] = t1;
		row2[x] = t2;
	}

}

__global__ void PolyExp_ax(float* pfSrc, float* pfScratch, int iHeight, int iWidth, int iWidthStride, int iPoly_n, float fPoly_sigma)
{
	float* row0;
	float* row1;
	float* row2;


	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y >= iHeight || x >= iPoly_n)
	{
		return;
	}

	row0 = pfScratch + y * (iWidth + iPoly_n * 2) + iPoly_n;
	row1 = row0 + iHeight * (iWidth + iPoly_n * 2);
	row2 = row0 + iHeight * (iWidth + iPoly_n * 2) * 2;


	row0[-1 - x] = row0[0];
	row1[-1 - x] = row1[0];
	row2[-1 - x] = row2[0];

	row0[iWidth + x] = row0[iWidth - 1];
	row1[iWidth + x] = row1[iWidth - 1];
	row2[iWidth + x] = row2[iWidth - 1];
}




__global__ void PolyExp_b(float* pfSrc, float* pfDst, float* pfScratch, int iHeight, int iWidth, int iWidthStride, int iPoly_n, float fPoly_sigma)
{
	float g0;
	//float *drow;
	float *drow0;
	float *drow1;
	float *drow2;
	float *drow3;
	float *drow4;
	float* g;
	float* xg;
	float* xxg;

	//float* row;
	float* row0;
	float* row1;
	float* row2;

	int k;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= iWidth || y >= iHeight)
	{
		return;
	}


	g = kbuf + iPoly_n;
	xg = g + iPoly_n * 2 + 1;
	xxg = xg + iPoly_n * 2 + 1;


	g0 = g[0];
	drow0 = pfDst + y * iWidthStride;
	drow1 = drow0 + iHeight * iWidthStride;
	drow2 = drow1 + iHeight * iWidthStride;
	drow3 = drow2 + iHeight * iWidthStride;
	drow4 = drow3 + iHeight * iWidthStride;

	//row = pfScratch + y * iWidth * 4 + iPoly_n * 3; // OPTOPT not sure if using too much memory but (*4) used in order to clear (iPoly_n * 3) and prevent overwrites. Anything less yields overwrites
	row0 = pfScratch + y * (iWidth + iPoly_n * 2) + iPoly_n;
	row1 = row0 + iHeight * (iWidth + iPoly_n * 2);
	row2 = row0 + iHeight * (iWidth + iPoly_n * 2) * 2;



	//float b1 = row[x*3] * g0;
	float b1 = row0[x] * g0;
	float b2 = 0;
	//float b3 = row[x*3+1] * g0;
	float b3 = row1[x] * g0;
	float b4 = 0;
	//float b5 = row[x*3+2] * g0;
	float b5 = row2[x] * g0;
	float b6 = 0;
	float tg;

	for (k = 1; k <= iPoly_n; k++)
	{
		//tg = row[(x+k) * 3] + row[(x-k) * 3];
		tg = row0[(x + k)] + row0[(x - k)];

		g0 = g[k];
		b1 += tg * g0;
		b4 += tg * xxg[k];
		//b2 += (row[(x+k)*3] - row[(x-k)*3])*xg[k];
		//b3 += (row[(x+k)*3+1] + row[(x-k)*3+1])*g0;
		//b6 += (row[(x+k)*3+1] - row[(x-k)*3+1])*xg[k];
		//b5 += (row[(x+k)*3+2] + row[(x-k)*3+2])*g0;

		b2 += (row0[(x + k)] - row0[(x - k)])*xg[k];
		b3 += (row1[(x + k)] + row1[(x - k)])*g0;
		b6 += (row1[(x + k)] - row1[(x - k)])*xg[k];
		b5 += (row2[(x + k)] + row2[(x - k)])*g0;

	}

	// do not store r1
	/*
	drow[x*5+1] = (float)(b2*ig11);
	drow[x*5] = (float)(b3*ig11);
	drow[x*5+3] = (float)(b1*ig03 + b4*ig33);
	drow[x*5+2] = (float)(b1*ig03 + b5*ig33);
	drow[x*5+4] = (float)(b6*ig55);
	*/
	drow1[x] = (float)(b2*ig11);
	drow0[x] = (float)(b3*ig11);
	drow3[x] = (float)(b1*ig03 + b4*ig33);
	drow2[x] = (float)(b1*ig03 + b5*ig33);
	drow4[x] = (float)(b6*ig55);
}


#define BORDER	5
__constant__  __device__ float border[BORDER] = { 0.14f, 0.14f, 0.4472f, 0.4472f, 0.4472f };

__global__ void UpdateMatrices(float* pfScratch1, float* pfScratch2, float* pfFlow, float* pfM, int iHeight, int iWidth, int iWidthStride, int iWindowHeight, int iOffset_y)
{
	const float* flow;
	const float* R0;
	const float* R1;
	const float* R2;
	const float* R3;
	const float* R4;

	float* M0;
	float* M1;
	float* M2;
	float* M3;
	float* M4;

	int x1;
	int y1;
	float r2;
	float r3;
	float r4;
	float r5;
	float r6;
	float dx;
	float dy;
	float fx;
	float fy;



	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y + iOffset_y;

	if (x >= iWidth || y >= iWindowHeight)
	{
		return;
	}

	R0 = pfScratch1 + y * iWidthStride;
	R1 = R0 + iHeight * iWidthStride;
	R2 = R1 + iHeight * iWidthStride;
	R3 = R2 + iHeight * iWidthStride;
	R4 = R3 + iHeight * iWidthStride;

	//float* M;
	//M = (pfM + y * iWidth * MAGIC_FIVE);


	M0 = pfM + y * iWidthStride;
	M1 = M0 + iHeight * iWidthStride;
	M2 = M1 + iHeight * iWidthStride;
	M3 = M2 + iHeight * iWidthStride;
	M4 = M3 + iHeight * iWidthStride;

	//flow = (pfFlow + y * iWidth * 2);
	//dx = flow[x*2];
	//dy = flow[x*2+1];
	flow = pfFlow + y * iWidthStride;
	dx = flow[x];
	dy = flow[x + iHeight * iWidthStride];


	fx = x + dx, fy = y + dy;

	x1 = (int)floor(fx);
	y1 = (int)floor(fy);




	fx -= x1; fy -= y1;

	if ((unsigned)x1 < (unsigned)(iWidth - 1) && (unsigned)y1 < (unsigned)(iHeight - 1))
	{
		float a00 = (1.f - fx)*(1.f - fy), a01 = fx*(1.f - fy), a10 = (1.f - fx)*fy, a11 = fx*fy;

		r2 = a00 * tex2D(texTemp1, x1, y1) + a01 * tex2D(texTemp1, x1 + 1, y1) + a10 * tex2D(texTemp1, x1, y1 + 1) + a11 * tex2D(texTemp1, x1 + 1, y1 + 1);
		y1 += iHeight;
		r3 = a00 * tex2D(texTemp1, x1, y1) + a01 * tex2D(texTemp1, x1 + 1, y1) + a10 * tex2D(texTemp1, x1, y1 + 1) + a11 * tex2D(texTemp1, x1 + 1, y1 + 1);
		y1 += iHeight;
		r4 = a00 * tex2D(texTemp1, x1, y1) + a01 * tex2D(texTemp1, x1 + 1, y1) + a10 * tex2D(texTemp1, x1, y1 + 1) + a11 * tex2D(texTemp1, x1 + 1, y1 + 1);
		y1 += iHeight;
		r5 = a00 * tex2D(texTemp1, x1, y1) + a01 * tex2D(texTemp1, x1 + 1, y1) + a10 * tex2D(texTemp1, x1, y1 + 1) + a11 * tex2D(texTemp1, x1 + 1, y1 + 1);
		y1 += iHeight;
		r6 = a00 * tex2D(texTemp1, x1, y1) + a01 * tex2D(texTemp1, x1 + 1, y1) + a10 * tex2D(texTemp1, x1, y1 + 1) + a11 * tex2D(texTemp1, x1 + 1, y1 + 1);


		r4 = (R2[x] + r4)*0.5f;
		r5 = (R3[x] + r5)*0.5f;
		r6 = (R4[x] + r6)*0.25f;
	}
	else
	{
		r2 = r3 = 0.f;
		r4 = R2[x];
		r5 = R3[x];
		r6 = R4[x] * 0.5f;
	}

	r2 = (R0[x] - r2)*0.5f;
	r3 = (R1[x] - r3)*0.5f;

	r2 += r4*dy + r6*dx;
	r3 += r6*dy + r5*dx;

	if ((unsigned)(x - BORDER) >= (unsigned)(iWidth - BORDER * 2) || (unsigned)(y - BORDER) >= (unsigned)(iHeight - BORDER * 2))
	{
		float scale = (x < BORDER ? border[x] : 1.f)*
			(x >= iWidth - BORDER ? border[iWidth - x - 1] : 1.f)*
			(y < BORDER ? border[y] : 1.f)*
			(y >= iHeight - BORDER ? border[iHeight - y - 1] : 1.f);

		r2 *= scale; r3 *= scale; r4 *= scale;
		r5 *= scale; r6 *= scale;
	}


	/*
	M[x*5]   = r4*r4 + r6*r6; // G(1,1)
	M[x*5+1] = (r4 + r5)*r6;  // G(1,2)=G(2,1)
	M[x*5+2] = r5*r5 + r6*r6; // G(2,2)
	M[x*5+3] = r4*r2 + r6*r3; // h(1)
	M[x*5+4] = r6*r2 + r5*r3; // h(2)
	*/

	M0[x] = r4*r4 + r6*r6; // G(1,1)
	M1[x] = (r4 + r5)*r6;  // G(1,2)=G(2,1)
	M2[x] = r5*r5 + r6*r6; // G(2,2)
	M3[x] = r4*r2 + r6*r3; // h(1)
	M4[x] = r6*r2 + r5*r3; // h(2)


}


__global__ void UpdateMatrices(cudaTextureObject_t texObjTemp1, float* pfScratch1, float* pfScratch2, float* pfFlow, float* pfM, int iHeight, int iWidth, int iWidthStride, int iWindowHeight, int iOffset_y)
{
	const float* flow;
	const float* R0;
	const float* R1;
	const float* R2;
	const float* R3;
	const float* R4;

	float* M0;
	float* M1;
	float* M2;
	float* M3;
	float* M4;

	int x1;
	int y1;
	float r2;
	float r3;
	float r4;
	float r5;
	float r6;
	float dx;
	float dy;
	float fx;
	float fy;



	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y + iOffset_y;

	if (x >= iWidth || y >= iWindowHeight)
	{
		return;
	}

	R0 = pfScratch1 + y * iWidthStride;
	R1 = R0 + iHeight * iWidthStride;
	R2 = R1 + iHeight * iWidthStride;
	R3 = R2 + iHeight * iWidthStride;
	R4 = R3 + iHeight * iWidthStride;

	//float* M;
	//M = (pfM + y * iWidth * MAGIC_FIVE);


	M0 = pfM + y * iWidthStride;
	M1 = M0 + iHeight * iWidthStride;
	M2 = M1 + iHeight * iWidthStride;
	M3 = M2 + iHeight * iWidthStride;
	M4 = M3 + iHeight * iWidthStride;

	//flow = (pfFlow + y * iWidth * 2);
	//dx = flow[x*2];
	//dy = flow[x*2+1];
	flow = pfFlow + y * iWidthStride;
	dx = flow[x];
	dy = flow[x + iHeight * iWidthStride];


	fx = x + dx, fy = y + dy;

	x1 = (int)floor(fx);
	y1 = (int)floor(fy);




	fx -= x1; fy -= y1;

	if ((unsigned)x1 < (unsigned)(iWidth - 1) && (unsigned)y1 < (unsigned)(iHeight - 1))
	{
		float a00 = (1.f - fx)*(1.f - fy), a01 = fx*(1.f - fy), a10 = (1.f - fx)*fy, a11 = fx*fy;

		r2 = a00 * tex2D<float>(texObjTemp1, x1, y1) + a01 * tex2D<float>(texObjTemp1, x1 + 1, y1) + a10 * tex2D<float>(texObjTemp1, x1, y1 + 1) + a11 * tex2D<float>(texObjTemp1, x1 + 1, y1 + 1);
		y1 += iHeight;
		r3 = a00 * tex2D<float>(texObjTemp1, x1, y1) + a01 * tex2D<float>(texObjTemp1, x1 + 1, y1) + a10 * tex2D<float>(texObjTemp1, x1, y1 + 1) + a11 * tex2D<float>(texObjTemp1, x1 + 1, y1 + 1);
		y1 += iHeight;
		r4 = a00 * tex2D<float>(texObjTemp1, x1, y1) + a01 * tex2D<float>(texObjTemp1, x1 + 1, y1) + a10 * tex2D<float>(texObjTemp1, x1, y1 + 1) + a11 * tex2D<float>(texObjTemp1, x1 + 1, y1 + 1);
		y1 += iHeight;
		r5 = a00 * tex2D<float>(texObjTemp1, x1, y1) + a01 * tex2D<float>(texObjTemp1, x1 + 1, y1) + a10 * tex2D<float>(texObjTemp1, x1, y1 + 1) + a11 * tex2D<float>(texObjTemp1, x1 + 1, y1 + 1);
		y1 += iHeight;
		r6 = a00 * tex2D<float>(texObjTemp1, x1, y1) + a01 * tex2D<float>(texObjTemp1, x1 + 1, y1) + a10 * tex2D<float>(texObjTemp1, x1, y1 + 1) + a11 * tex2D<float>(texObjTemp1, x1 + 1, y1 + 1);


		r4 = (R2[x] + r4)*0.5f;
		r5 = (R3[x] + r5)*0.5f;
		r6 = (R4[x] + r6)*0.25f;
	}
	else
	{
		r2 = r3 = 0.f;
		r4 = R2[x];
		r5 = R3[x];
		r6 = R4[x] * 0.5f;
	}

	r2 = (R0[x] - r2)*0.5f;
	r3 = (R1[x] - r3)*0.5f;

	r2 += r4*dy + r6*dx;
	r3 += r6*dy + r5*dx;

	if ((unsigned)(x - BORDER) >= (unsigned)(iWidth - BORDER * 2) || (unsigned)(y - BORDER) >= (unsigned)(iHeight - BORDER * 2))
	{
		float scale = (x < BORDER ? border[x] : 1.f)*
			(x >= iWidth - BORDER ? border[iWidth - x - 1] : 1.f)*
			(y < BORDER ? border[y] : 1.f)*
			(y >= iHeight - BORDER ? border[iHeight - y - 1] : 1.f);

		r2 *= scale; r3 *= scale; r4 *= scale;
		r5 *= scale; r6 *= scale;
	}

	/*
	M[x*5]   = r4*r4 + r6*r6; // G(1,1)
	M[x*5+1] = (r4 + r5)*r6;  // G(1,2)=G(2,1)
	M[x*5+2] = r5*r5 + r6*r6; // G(2,2)
	M[x*5+3] = r4*r2 + r6*r3; // h(1)
	M[x*5+4] = r6*r2 + r5*r3; // h(2)
	*/

	M0[x] = r4*r4 + r6*r6; // G(1,1)
	M1[x] = (r4 + r5)*r6;  // G(1,2)=G(2,1)
	M2[x] = r5*r5 + r6*r6; // G(2,2)
	M3[x] = r4*r2 + r6*r3; // h(1)
	M4[x] = r6*r2 + r5*r3; // h(2)


}



__constant__  __device__ float kernel[] = { 0.26601174f, 0.21300554f, 0.10936069f, 0.036000773f, 0.0075987582f, 0.0010283801 };

__global__ void UpdateFlow_a(OPTICAL_FLOW_SCRATCH* pOFScratch, int iHeight, int iWidth, int iWidthStride)
{
	int i;
	float** srow;
	int m;


	int iGlobalThreadid2D;

	iGlobalThreadid2D = blockIdx.y * gridDim.x * blockDim.y * blockDim.x; // total threads in all blocks of previous row in grid
	iGlobalThreadid2D += blockIdx.x * blockDim.y * blockDim.x; // total threads is all previous blocks in current row of grid
	iGlobalThreadid2D += threadIdx.y * blockDim.x + threadIdx.x; // thread id in current block

	if (iGlobalThreadid2D >= iHeight)
	{
		return;
	}

	int y = iGlobalThreadid2D;  // OPTOPT only using iHeight threads!!!

	m = 5;

	srow = &(pOFScratch->rgSRowScratch + y * 11)[0];

	for (i = 0; i <= m; i++)
	{
		//srow[m-i] = (pOFScratch->rgM + iWidth * MAGIC_FIVE * max(y-i,0));
		//srow[m+i] = (pOFScratch->rgM + iWidth * MAGIC_FIVE * min(y+i,iHeight-1));

		srow[m - i] = (pOFScratch->rgM + iWidthStride * max(y - i, 0));
		srow[m + i] = (pOFScratch->rgM + iWidthStride * min(y + i, iHeight - 1));
	}
}



__global__ void UpdateFlow_b(float* pfVSumScratch, float** pfSRowScratch, int iHeight, int iWidth, int iWidthStride, int iWindowHeight, int iOffset_y)
{
	int i;
	int m;
	float* vsum;
	float** srow;
	float s0;
	int iBigStride;
	int iBigStride2;
	int iAugmentedWidth;
	int x_5;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y + iOffset_y;


	m = 5; // (Winsize/2) MAGICMAGIC
	srow = &(pfSRowScratch + y * 11)[0];

	//vsum = pfVSumScratch + y * iWidth * 6 + (m+1) * 5;
	iAugmentedWidth = iWidth + 2 * (m + 1);
	vsum = pfVSumScratch + y * iAugmentedWidth + (m + 1);
	iBigStride2 = iHeight * iAugmentedWidth;


	if (x >= iWidth || y >= iWindowHeight)
	{
		return;
	}

	iBigStride = iHeight * iWidthStride;


	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	x_5 = x;
	s0 = srow[m][x] * kernel[0];

	for (i = 1; i <= m; i++) // OPTOPT will unrolling help here?
	{
		s0 += (srow[m + i][x] + srow[m - i][x])*kernel[i];
	}
	vsum[x_5] = s0;

	x_5 += iBigStride2;
	s0 = srow[m][x + iBigStride] * kernel[0];
	for (i = 1; i <= m; i++)
	{
		s0 += (srow[m + i][x + iBigStride] + srow[m - i][x + iBigStride])*kernel[i];
	}
	vsum[x_5] = s0;

	x_5 += iBigStride2;
	s0 = srow[m][x + 2 * iBigStride] * kernel[0];
	for (i = 1; i <= m; i++)
	{
		s0 += (srow[m + i][x + 2 * iBigStride] + srow[m - i][x + 2 * iBigStride])*kernel[i];
	}
	vsum[x_5] = s0;

	x_5 += iBigStride2;
	s0 = srow[m][x + 3 * iBigStride] * kernel[0];
	for (i = 1; i <= m; i++)
	{
		s0 += (srow[m + i][x + 3 * iBigStride] + srow[m - i][x + 3 * iBigStride])*kernel[i];
	}
	vsum[x_5] = s0;

	x_5 += iBigStride2;
	s0 = srow[m][x + 4 * iBigStride] * kernel[0];
	for (i = 1; i <= m; i++)
	{
		s0 += (srow[m + i][x + 4 * iBigStride] + srow[m - i][x + 4 * iBigStride])*kernel[i];
	}
	vsum[x_5] = s0;
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

}


__global__ void UpdateFlow_c(float* pfVSumScratch, int iHeight, int iWidth, int iWidthStride, int iWindowHeight, int iOffset_y)
{
	//float* vsum;
	float* vsum0;
	float* vsum1;
	float* vsum2;
	float* vsum3;
	float* vsum4;
	int m;
	int iAugmentedWidth;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y + iOffset_y;

	m = 5;

	if (y >= iWindowHeight || x >= m)
	{
		return;
	}



	//vsum = pfVSumScratch + y * iWidth * 6 + (m+1) * 5;
	iAugmentedWidth = iWidth + 2 * (m + 1);
	vsum0 = pfVSumScratch + y * iAugmentedWidth + (m + 1);
	vsum1 = vsum0 + iHeight * iAugmentedWidth;
	vsum2 = vsum1 + iHeight * iAugmentedWidth;
	vsum3 = vsum2 + iHeight * iAugmentedWidth;
	vsum4 = vsum3 + iHeight * iAugmentedWidth;

	/*
	for( x = 0; x < m * 5; x++ )
	{
	vsum[-1-x] = vsum[4-x];
	vsum[iWidth*5+x] = vsum[iWidth*5+x-5];
	}
	*/


	vsum0[-1 - x] = vsum0[0];
	vsum1[-1 - x] = vsum1[0];
	vsum2[-1 - x] = vsum2[0];
	vsum3[-1 - x] = vsum3[0];
	vsum4[-1 - x] = vsum4[0];

	vsum0[iWidth + x] = vsum0[iWidth - 1];
	vsum1[iWidth + x] = vsum1[iWidth - 1];
	vsum2[iWidth + x] = vsum2[iWidth - 1];
	vsum3[iWidth + x] = vsum3[iWidth - 1];
	vsum4[iWidth + x] = vsum4[iWidth - 1];
}


__global__ void UpdateFlow_d(float* pfVSumScratch, float* pfHSumScratch, int iHeight, int iWidth, int iWidthStride, int iWindowHeight, int iOffset_y)
{
	int i;
	int m;
	float* vsum;
	//float* hsum;
	float* hsum0;
	float* hsum1;
	float* hsum2;
	float* hsum3;
	float* hsum4;
	int x_5;
	float sum;
	int iAugmentedWidth;
	int iBigStride2;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y + iOffset_y;


	m = 5; // (Winsize/2) MAGICMAGIC
	//hsum = pfHSumScratch + y * iWidth * 5;
	hsum0 = pfHSumScratch + y * iWidth;
	hsum1 = hsum0 + iHeight * iWidth;
	hsum2 = hsum1 + iHeight * iWidth;
	hsum3 = hsum2 + iHeight * iWidth;
	hsum4 = hsum3 + iHeight * iWidth;



	//vsum = pfVSumScratch + y * iWidth * 6 + (m+1) * 5;
	iAugmentedWidth = iWidth + 2 * (m + 1);
	vsum = pfVSumScratch + y * iAugmentedWidth + (m + 1);

	iBigStride2 = iHeight * iAugmentedWidth;


	if (x >= iWidth || y >= iWindowHeight)
	{
		return;
	}

	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	int x_v;
	x_5 = x * 5;
	x_v = x;
	sum = vsum[x_v] * kernel[0];

	for (i = 1; i <= m; i++) // OPTOPT will unrolling help here?
	{
		sum += kernel[i] * (vsum[x_v - i] + vsum[x_v + i]);
	}
	//hsum[x_5] = sum;
	hsum0[x] = sum;


	x_5++;
	x_v += iBigStride2;
	sum = vsum[x_v] * kernel[0];
	for (i = 1; i <= m; i++)
	{
		sum += kernel[i] * (vsum[x_v - i] + vsum[x_v + i]);
	}
	//hsum[x_5] = sum;
	hsum1[x] = sum;

	x_5++;
	x_v += iBigStride2;
	sum = vsum[x_v] * kernel[0];
	for (i = 1; i <= m; i++)
	{
		sum += kernel[i] * (vsum[x_v - i] + vsum[x_v + i]);
	}
	//hsum[x_5] = sum;
	hsum2[x] = sum;

	x_5++;
	x_v += iBigStride2;
	sum = vsum[x_v] * kernel[0];
	for (i = 1; i <= m; i++)
	{
		sum += kernel[i] * (vsum[x_v - i] + vsum[x_v + i]);
	}
	//hsum[x_5] = sum;
	hsum3[x] = sum;

	x_5++;
	x_v += iBigStride2;
	sum = vsum[x_v] * kernel[0];
	for (i = 1; i <= m; i++)
	{
		sum += kernel[i] * (vsum[x_v - i] + vsum[x_v + i]);
	}
	//hsum[x_5] = sum;
	hsum4[x] = sum;
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



}


__global__ void UpdateFlow_e(float* pfFlow, float* pfHSumScratch, int iHeight, int iWidth, int iWidthStride, int iWindowHeight, int iOffset_y)
{
	//float* hsum;
	float* hsum0;
	float* hsum1;
	float* hsum2;
	float* hsum3;
	float* hsum4;
	float g11, g12, g22, h1, h2;
	float* flow;
	float fdet;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y + iOffset_y;


	//hsum = pfHSumScratch + y * iWidth * 5;
	hsum0 = pfHSumScratch + y * iWidth;
	hsum1 = hsum0 + iHeight * iWidth;
	hsum2 = hsum1 + iHeight * iWidth;
	hsum3 = hsum2 + iHeight * iWidth;
	hsum4 = hsum3 + iHeight * iWidth;


	//flow = pfFlow + iWidth * 2 * y;
	flow = pfFlow + y * iWidthStride;

	if (x >= iWidth || y >= iWindowHeight)
	{
		return;
	}

	/*
	g11 = hsum[x*5];
	g12 = hsum[x*5+1];
	g22 = hsum[x*5+2];
	h1 = hsum[x*5+3];
	h2 = hsum[x*5+4];
	*/

	g11 = hsum0[x];
	g12 = hsum1[x];
	g22 = hsum2[x];
	h1 = hsum3[x];
	h2 = hsum4[x];

	fdet = 1.0f / (g11*g22 - g12*g12 + 1e-3);

	//flow[x*2] = (g11*h2-g12*h1)*fdet;
	//flow[x*2+1] = (g22*h1-g12*h2)*fdet;
	flow[x] = (g11*h2 - g12*h1)*fdet;
	flow[x + iHeight * iWidthStride] = (g22*h1 - g12*h2)*fdet;

}


__global__ void MoveFlow(float* pfPrevFlow, float* pfFlowScratch, int iHeight, int iWidth, int iWidthStride)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = y * iWidthStride + x;

	if (y >= (iHeight * 2) || x >= iWidth)
	{
		return;
	}

	pfPrevFlow[i] = pfFlowScratch[i];
}

__global__ void MoveFlow(GEN_DENSETRAJ_BUFFERS* pDenseTrajBuffers, OPTICAL_FLOW_SCRATCH* pOFScratch, int iHeight, int iWidth, int iWidthStride, int iScale) // OPTOPT this splits flowscratch to flow_x and flow_y, need to get rid of this an generate flow into flow_x and flow_y
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = y * iWidthStride + x;

	if (y >= iHeight || x >= iWidth)
	{
		return;
	}

	pDenseTrajBuffers->rgFlow_x[iScale][i] = pOFScratch->rgFlowScratch[iScale][i];
	pDenseTrajBuffers->rgFlow_y[iScale][i] = pOFScratch->rgFlowScratch[iScale][iHeight * iWidthStride + i];

}

void UpdateFlowAndMatrices(OPTICAL_FLOW_SCRATCH* pOFScratch, float* pfTemp0, float* pfTemp1, int iHeight, int iWidth, int iWidthStride, int iScale)
{
	int i;
	int y;
	int iWinSize = 10;
	int min_update_stripe = max((1 << 10) / iWidth, iWinSize);
	int rgFlowUpdateRows[100];
	int rgMatrixUpdateRows[100];
	int iTotalUpdateRows;
	int y0;
	int y1;
	int iRows;
	int iOffset_y;

	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = 6;
	dimBlock.x = 32;

	y0 = 0;
	rgFlowUpdateRows[0] = 0;
	rgMatrixUpdateRows[0] = 0;
	iTotalUpdateRows = 1;
	for (y = 0; y < iHeight; y++)
	{
		y1 = y == iHeight - 1 ? iHeight : y - iWinSize;
		if (y1 == iHeight || y1 >= y0 + min_update_stripe)
		{
			rgFlowUpdateRows[iTotalUpdateRows] = y + 1;
			rgMatrixUpdateRows[iTotalUpdateRows] = y1;
			iTotalUpdateRows++;
			y0 = y1;
		}
	}
	rgFlowUpdateRows[iTotalUpdateRows] = iHeight;
	rgMatrixUpdateRows[iTotalUpdateRows] = iHeight;


	for (i = 0; i < iTotalUpdateRows; i++)
	{
		iRows = rgFlowUpdateRows[i + 1] - rgFlowUpdateRows[i];
		iOffset_y = rgFlowUpdateRows[i];

		if (!iRows)
		{
			break;
		}
		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		dimGrid.x = (int)ceil((float)iWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iRows / (float)dimBlock.y);

		UpdateFlow_b << <dimGrid, dimBlock >> >(pOFScratch->rgVSumScratch, pOFScratch->rgSRowScratch, iHeight, iWidth, iWidthStride, rgFlowUpdateRows[i + 1], iOffset_y);


		dimGrid.x = 1; // NOTE: ok as long as MAGIC_FIVE < dimBlock.x
		dimGrid.y = (int)ceil((float)iHeight / (float)dimBlock.y); // OPTOPT should be able to use less than iHeight here (rgUpdateRows[i+1] maybe?)

		UpdateFlow_c << <dimGrid, dimBlock >> >(pOFScratch->rgVSumScratch, iHeight, iWidth, iWidthStride, rgFlowUpdateRows[i + 1], iOffset_y);


		dimGrid.x = (int)ceil((float)iWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iRows / (float)dimBlock.y);

		UpdateFlow_d << <dimGrid, dimBlock >> >(pOFScratch->rgVSumScratch, pOFScratch->rgHSumScratch, iHeight, iWidth, iWidthStride, rgFlowUpdateRows[i + 1], iOffset_y);

		UpdateFlow_e << <dimGrid, dimBlock >> >(pOFScratch->rgFlowScratch[iScale], pOFScratch->rgHSumScratch, iHeight, iWidth, iWidthStride, rgFlowUpdateRows[i + 1], iOffset_y);
		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


		dimGrid.x = (int)ceil((float)iWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iHeight / (float)dimBlock.y); // OPTOPT should be able to use less than iHeight here (rgUpdateRows[i+1] maybe?)

		UpdateMatrices << <dimGrid, dimBlock >> >(pfTemp0, pfTemp1, pOFScratch->rgFlowScratch[iScale], pOFScratch->rgM, iHeight, iWidth, iWidthStride, rgMatrixUpdateRows[i + 1], rgMatrixUpdateRows[i]);
		//UpdateMatrices<<<dimGrid, dimBlock>>>( pfTemp0, pfTemp1, pOFScratch->rgFlowScratch, pOFScratch->rgM, iHeight, iWidth, iWidthStride, iHeight, 0 );

	}

	//float* pfFlow_y = new float[640 * 480];
	//CopyDataFromGPU( pfFlow_y, pOFScratch->rgFlowScratch, sizeof(float) * iHeight * iWidth );
	//delete pfFlow_y;

}

void UpdateFlowAndMatrices(cudaTextureObject_t texObjTemp1, OPTICAL_FLOW_SCRATCH* pOFScratch, float* pfTemp0, float* pfTemp1, int iHeight, int iWidth, int iWidthStride, int iScale)
{
	int i;
	int y;
	int iWinSize = 10;
	int min_update_stripe = max((1 << 10) / iWidth, iWinSize);
	int rgFlowUpdateRows[100];
	int rgMatrixUpdateRows[100];
	int iTotalUpdateRows;
	int y0;
	int y1;
	int iRows;
	int iOffset_y;

	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = 6;
	dimBlock.x = 32;

	y0 = 0;
	rgFlowUpdateRows[0] = 0;
	rgMatrixUpdateRows[0] = 0;
	iTotalUpdateRows = 1;
	for (y = 0; y < iHeight; y++)
	{
		y1 = y == iHeight - 1 ? iHeight : y - iWinSize;
		if (y1 == iHeight || y1 >= y0 + min_update_stripe)
		{
			rgFlowUpdateRows[iTotalUpdateRows] = y + 1;
			rgMatrixUpdateRows[iTotalUpdateRows] = y1;
			iTotalUpdateRows++;
			y0 = y1;
		}
	}
	rgFlowUpdateRows[iTotalUpdateRows] = iHeight;
	rgMatrixUpdateRows[iTotalUpdateRows] = iHeight;


	for (i = 0; i < iTotalUpdateRows; i++)
	{
		iRows = rgFlowUpdateRows[i + 1] - rgFlowUpdateRows[i];
		iOffset_y = rgFlowUpdateRows[i];

		if (!iRows)
		{
			break;
		}
		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		dimGrid.x = (int)ceil((float)iWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iRows / (float)dimBlock.y);

		UpdateFlow_b << <dimGrid, dimBlock >> >(pOFScratch->rgVSumScratch, pOFScratch->rgSRowScratch, iHeight, iWidth, g_iWidthStride, rgFlowUpdateRows[i + 1], iOffset_y);


		dimGrid.x = 1; // NOTE: ok as long as MAGIC_FIVE < dimBlock.x
		dimGrid.y = (int)ceil((float)iHeight / (float)dimBlock.y); // OPTOPT should be able to use less than iHeight here (rgUpdateRows[i+1] maybe?)

		UpdateFlow_c << <dimGrid, dimBlock >> >(pOFScratch->rgVSumScratch, iHeight, iWidth, iWidthStride, rgFlowUpdateRows[i + 1], iOffset_y);


		dimGrid.x = (int)ceil((float)iWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iRows / (float)dimBlock.y);

		UpdateFlow_d << <dimGrid, dimBlock >> >(pOFScratch->rgVSumScratch, pOFScratch->rgHSumScratch, iHeight, iWidth, iWidthStride, rgFlowUpdateRows[i + 1], iOffset_y);

		UpdateFlow_e << <dimGrid, dimBlock >> >(pOFScratch->rgFlowScratch[iScale], pOFScratch->rgHSumScratch, iHeight, iWidth, iWidthStride, rgFlowUpdateRows[i + 1], iOffset_y);
		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


		dimGrid.x = (int)ceil((float)iWidth / (float)dimBlock.x);
		dimGrid.y = (int)ceil((float)iHeight / (float)dimBlock.y); // OPTOPT should be able to use less than iHeight here (rgUpdateRows[i+1] maybe?)

		UpdateMatrices << <dimGrid, dimBlock >> >(texObjTemp1, pfTemp0, pfTemp1, pOFScratch->rgFlowScratch[iScale], pOFScratch->rgM, iHeight, iWidth, iWidthStride, rgMatrixUpdateRows[i + 1], rgMatrixUpdateRows[i]);
		//UpdateMatrices<<<dimGrid, dimBlock>>>( pfTemp0, pfTemp1, pOFScratch->rgFlowScratch, pOFScratch->rgM, iHeight, iWidth, iWidthStride, iHeight, 0 );

	}

	//float* pfFlow_y = new float[640 * 480];
	//CopyDataFromGPU( pfFlow_y, pOFScratch->rgFlowScratch, sizeof(float) * iHeight * iWidth );
	//delete pfFlow_y;

}


#include "resize_fermi.cu"

