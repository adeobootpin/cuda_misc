#include <windows.h>
#include <assert.h>
#include "cuboxfilter.h"

__global__ void CuBoxFilterMid( float* pfSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );
__global__ void CuBoxFilterBorders( float* pfSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );


void CuBoxFilter( void* pvSrc, void* pvDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = 6;
	dimBlock.x = 32;



	dimGrid.x = (int)ceil((float)iWidth/(float)dimBlock.x);
	dimGrid.y = (int)ceil((float)iHeight/(float)dimBlock.y);

	CuBoxFilterMid<<<dimGrid, dimBlock>>>( (float*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );


	dimGrid.x = min( 65535, (int)ceil((float)max(iHeight,iWidth)/(float)(dimBlock.x * dimBlock.y)) );
	dimGrid.y = (int)ceil((float)max(iHeight,iWidth)/(float)(dimBlock.x * dimBlock.y * dimGrid.x));

	CuBoxFilterBorders<<<dimGrid, dimBlock>>>( (float*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );

	//CopyDataFromGPU( pfDst, pvDst, sizeof(float) * iHeight * iWidth );

}

__global__ void CuBoxFilterMid( float* pfSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i =  y * iWidthStep + x;
	

	if( x <= 0 || x >= (iWidth-1) || y <= 0 || y >= (iHeight-1) )
	{
		return;
	}

	pfDst[i] = pfSrc[i-iWidthStep-1] + pfSrc[i-iWidthStep] + pfSrc[i-iWidthStep+1] + pfSrc[i-1] + pfSrc[i]	+ pfSrc[i+1] + pfSrc[i+iWidthStep-1] + pfSrc[i+iWidthStep] + pfSrc[i+iWidthStep+1];

}

__global__ void CuBoxFilterBorders( float* pfSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	int iGlobalThreadid2D;
	int x;
	int y;

	iGlobalThreadid2D = blockIdx.y * gridDim.x * blockDim.y * blockDim.x; // total threads in all blocks of previous row in grid
	iGlobalThreadid2D += blockIdx.x * blockDim.y * blockDim.x; // total threads is all previous blocks in current row of grid
	iGlobalThreadid2D += threadIdx.y * blockDim.x + threadIdx.x; // thread id in current block

	x = iGlobalThreadid2D;

	//
	// BORDER_REPLICATE
	//
	if( x >=1 && x < iWidth-1 ) // do the horizontal borders
	{
		pfDst[x] = 2 * (pfSrc[x-1] + pfSrc[x] + pfSrc[x+1]) + pfSrc[iWidthStep + x-1] + pfSrc[iWidthStep + x] + pfSrc[iWidthStep + x+1];
		pfDst[(iHeight-1) * iWidthStep + x] = 2 * (pfSrc[(iHeight-1) * iWidthStep + x - 1] + pfSrc[(iHeight-1) * iWidthStep + x] + pfSrc[(iHeight-1) * iWidthStep + x + 1]) + pfSrc[(iHeight-2) * iWidthStep + x - 1] + pfSrc[(iHeight-2) * iWidthStep + x] + pfSrc[(iHeight-2) * iWidthStep + x + 1];
	}

	y = iGlobalThreadid2D;

	if( y >=1 && y < iHeight-1 ) // do the vertical borders
	{
		pfDst[y * iWidthStep] = 2 * (pfSrc[(y - 1) * iWidthStep] + pfSrc[y * iWidthStep] + pfSrc[(y + 1) * iWidthStep]) + pfSrc[(y - 1) * iWidthStep + 1] + pfSrc[y * iWidthStep + 1] + pfSrc[(y + 1) * iWidthStep + 1];
		pfDst[y * iWidthStep + (iWidth-1)] = 2 * (pfSrc[(y - 1) * iWidthStep + (iWidth-1)] + pfSrc[y * iWidthStep + (iWidth-1)] + pfSrc[(y + 1) * iWidthStep + (iWidth-1)]) + pfSrc[(y - 1) * iWidthStep + (iWidth-2)] + pfSrc[y * iWidthStep + (iWidth-2)] + pfSrc[(y + 1) * iWidthStep + (iWidth-2)];
	}

	if( iGlobalThreadid2D == 0 ) // do the corners
	{
		pfDst[0] = 2 * (pfSrc[0] + pfSrc[1]) + (pfSrc[iWidthStep] + pfSrc[iWidthStep+1]) + (pfSrc[0] + pfSrc[iWidthStep]) + pfSrc[0]; // top left corner
		pfDst[iWidth-1] = 2 * (pfSrc[iWidth-1] + pfSrc[iWidth-2]) + (pfSrc[iWidthStep + iWidth-1] + pfSrc[iWidthStep+iWidth-2]) + (pfSrc[iWidth-1] + pfSrc[iWidthStep + iWidth-1]) + pfSrc[iWidth-1]; // top right corner
		pfDst[(iHeight-1) * iWidthStep] = 2 * (pfSrc[(iHeight-1) * iWidthStep] + pfSrc[(iHeight-1) * iWidthStep + 1]) + (pfSrc[(iHeight-2) * iWidthStep] + pfSrc[(iHeight-2) * iWidthStep + 1]) + (pfSrc[(iHeight-1) * iWidthStep] + pfSrc[(iHeight-2) * iWidthStep]) + pfSrc[(iHeight-1) * iWidthStep]; // bottom left corner
		pfDst[(iHeight-1) * iWidthStep + (iWidth-1)] = 2 * (pfSrc[(iHeight-1) * iWidthStep + (iWidth-1)] + pfSrc[(iHeight-1) * iWidthStep + (iWidth-2)]) + (pfSrc[(iHeight-2) * iWidthStep + (iWidth-1)] + pfSrc[(iHeight-2) * iWidthStep + (iWidth-2)]) + (pfSrc[(iHeight-1) * iWidthStep + (iWidth-1)] + pfSrc[(iHeight-2) * iWidthStep + (iWidth-1)]) + pfSrc[(iHeight-1) * iWidthStep + (iWidth-1)]; // bottom right corner
	}
}
