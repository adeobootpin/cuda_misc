#include <windows.h>
#include <assert.h>
#include "cusobel.h"



__global__ void CuSobel3DxMid( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );
__global__ void CuSobel3DxBorders( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );
__global__ void CuSobel3DyMid( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );
__global__ void CuSobel3DyBorders( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );


void CuSobel3DxUF( void* pvSrc, void* pvDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = 6;
	dimBlock.x = 32;


	dimGrid.x = (int)ceil((float)iWidth/(float)dimBlock.x);
	dimGrid.y = (int)ceil((float)iHeight/(float)dimBlock.y);

	CuSobel3DxMid<<<dimGrid, dimBlock>>>( (unsigned char*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );

	dimGrid.x = min( 65535, (int)ceil((float)max(iHeight,iWidth)/(float)(dimBlock.x * dimBlock.y)) );
	dimGrid.y = (int)ceil((float)max(iHeight,iWidth)/(float)(dimBlock.x * dimBlock.y * dimGrid.x));

	CuSobel3DxBorders<<<dimGrid, dimBlock>>>( (unsigned char*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );
}

void CuSobel3DyUF( void* pvSrc, void* pvDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	dim3 dimGrid;
	dim3 dimBlock;

	dimBlock.y = 6;
	dimBlock.x = 32;


	dimGrid.x = (int)ceil((float)iWidth/(float)dimBlock.x);
	dimGrid.y = (int)ceil((float)iHeight/(float)dimBlock.y);

	CuSobel3DyMid<<<dimGrid, dimBlock>>>( (unsigned char*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );


	dimGrid.x = min( 65535, (int)ceil((float)max(iHeight,iWidth)/(float)(dimBlock.x * dimBlock.y)) );
	dimGrid.y = (int)ceil((float)max(iHeight,iWidth)/(float)(dimBlock.x * dimBlock.y * dimGrid.x));

	CuSobel3DyBorders<<<dimGrid, dimBlock>>>( (unsigned char*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );


}





__global__ void CuSobel3DxMid( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	float fVal0;
	float fVal1;
	float fVal2;
	float fVal3;
	float fVal4;
	float fVal5;
	float fVal6;
	float fVal7;
	float fVal8;

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i =  y * iWidthStep + x;
	

	if( x <= 0 || x >= (iWidth-1) || y <= 0 || y >= (iHeight-1) )
	{
		return;
	}

	fVal0 = pchSrc[i-iWidthStep-1] * (-1);
	fVal1 = pchSrc[i-iWidthStep]   * (0);
	fVal2 = pchSrc[i-iWidthStep+1] * (1);
	fVal3 = pchSrc[i-1] * (-2);
	fVal4 = pchSrc[i] * (0);
	fVal5 = pchSrc[i+1] * (2);
	fVal6 = pchSrc[i+iWidthStep-1] * (-1);
	fVal7 = pchSrc[i+iWidthStep] * (0);
	fVal8 = pchSrc[i+iWidthStep+1] * (1);


	pfDst[i] = (fVal0 + fVal1 + fVal2 + fVal3 + fVal4 + fVal5 + fVal6 + fVal7 + fVal8) * fScale;

}


__global__ void CuSobel3DxBorders( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	int iGlobalThreadid2D;
	int x;
	int y;
	float fVal;

	iGlobalThreadid2D = blockIdx.y * gridDim.x * blockDim.y * blockDim.x; // total threads in all blocks of previous row in grid
	iGlobalThreadid2D += blockIdx.x * blockDim.y * blockDim.x; // total threads is all previous blocks in current row of grid
	iGlobalThreadid2D += threadIdx.y * blockDim.x + threadIdx.x; // thread id in current block

	x = iGlobalThreadid2D;

	//
	// BORDER_REPLICATE
	//
	if( x >=1 && x < iWidth-1 ) // do the horizontal borders
	{
		pfDst[x] = fScale * (pchSrc[x+1] + 2 * pchSrc[x+1] + pchSrc[iWidthStep + x+1] - pchSrc[x-1] - 2 * pchSrc[x-1] - pchSrc[iWidthStep + x-1]);

		fVal = pchSrc[(iHeight-2) * iWidthStep + x + 1] + 2 * pchSrc[(iHeight-1) * iWidthStep + x + 1] + pchSrc[(iHeight-1) * iWidthStep + x + 1];
		fVal += -pchSrc[(iHeight-2) * iWidthStep + x - 1] - 2 * pchSrc[(iHeight-1) * iWidthStep + x - 1] - pchSrc[(iHeight-1) * iWidthStep + x - 1];

		pfDst[(iHeight-1) * iWidthStep + x] = fScale * fVal;
	}

	y = iGlobalThreadid2D;

	if( y >=1 && y < iHeight-1 ) // do the vertical borders
	{
		pfDst[y * iWidthStep] = fScale * (pchSrc[(y - 1) * iWidthStep + 1] + 2 * pchSrc[y * iWidthStep + 1] + pchSrc[(y + 1) * iWidthStep + 1] - pchSrc[(y - 1) * iWidthStep] - 2 * pchSrc[y * iWidthStep] - pchSrc[(y + 1) * iWidthStep]);
		pfDst[y * iWidthStep + (iWidth-1)] =  fScale * (pchSrc[(y - 1) * iWidthStep + (iWidth-1)] + 2 * pchSrc[y * iWidthStep + (iWidth-1)] + pchSrc[(y + 1) * iWidthStep + (iWidth-1)] - pchSrc[(y - 1) * iWidthStep + (iWidth-2)] - 2 * pchSrc[y * iWidthStep + (iWidth-2)] - pchSrc[(y + 1) * iWidthStep + (iWidth-2)]);
	}

	if( iGlobalThreadid2D == 0 ) // do the corners
	{
		pfDst[0] = fScale * (pchSrc[1] + 2 * pchSrc[1] + pchSrc[iWidthStep+1] - pchSrc[0] - 2 * pchSrc[0] - pchSrc[iWidthStep]); // top left corner
		pfDst[iWidth-1] = fScale * (pchSrc[iWidth-1] + 2 * pchSrc[iWidth-1] + pchSrc[iWidthStep + iWidth-1] - pchSrc[iWidth-2] - 2 * pchSrc[iWidth-2] - pchSrc[iWidthStep + iWidth-2]); // top right corner
		pfDst[(iHeight-1) * iWidthStep] = fScale * (pchSrc[(iHeight-2) * iWidthStep + 1] + 2 * pchSrc[(iHeight-1) * iWidthStep + 1] + pchSrc[(iHeight-1) * iWidthStep + 1] - pchSrc[(iHeight-2) * iWidthStep] - 2 * pchSrc[(iHeight-1) * iWidthStep] - pchSrc[(iHeight-1) * iWidthStep]); // bottom left corner
		pfDst[(iHeight-1) * iWidthStep + (iWidth-1)] = fScale * (pchSrc[(iHeight-2) * iWidthStep + (iWidth-1)] + 2 * pchSrc[(iHeight-1) * iWidthStep + (iWidth-1)] + pchSrc[(iHeight-1) * iWidthStep + (iWidth-1)] - pchSrc[(iHeight-2) * iWidthStep + (iWidth-2)] - 2 * pchSrc[(iHeight-1) * iWidthStep + (iWidth-2)] - pchSrc[(iHeight-1) * iWidthStep + (iWidth-2)]); // bottom right corner
	}
}


__global__ void CuSobel3DyMid( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	float fVal0;
	float fVal1;
	float fVal2;
	float fVal3;
	float fVal4;
	float fVal5;
	float fVal6;
	float fVal7;
	float fVal8;

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i =  y * iWidthStep + x;
	

	if( x <= 0 || x >= (iWidth-1) || y <= 0 || y >= (iHeight-1) )
	{
		return;
	}

	fVal0 = pchSrc[i-iWidthStep-1] * (-1);
	fVal1 = pchSrc[i-iWidthStep]   * (-2);
	fVal2 = pchSrc[i-iWidthStep+1] * (-1);
	fVal3 = pchSrc[i-1] * (0);
	fVal4 = pchSrc[i] * (0);
	fVal5 = pchSrc[i+1] * (0);
	fVal6 = pchSrc[i+iWidthStep-1] * (1);
	fVal7 = pchSrc[i+iWidthStep] * (2);
	fVal8 = pchSrc[i+iWidthStep+1] * (1);


	pfDst[i] = (fVal0 + fVal1 + fVal2 + fVal3 + fVal4 + fVal5 + fVal6 + fVal7 + fVal8) * fScale;
}



__global__ void CuSobel3DyBorders( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	int iGlobalThreadid2D;
	int x;
	int y;
	float fVal;

	iGlobalThreadid2D = blockIdx.y * gridDim.x * blockDim.y * blockDim.x; // total threads in all blocks of previous row in grid
	iGlobalThreadid2D += blockIdx.x * blockDim.y * blockDim.x; // total threads is all previous blocks in current row of grid
	iGlobalThreadid2D += threadIdx.y * blockDim.x + threadIdx.x; // thread id in current block

	x = iGlobalThreadid2D;

	//
	// BORDER_REPLICATE
	//
	if( x >=1 && x < iWidth-1 ) // do the horizontal borders
	{
		pfDst[x] = fScale * (pchSrc[iWidthStep + x - 1] + 2 * pchSrc[iWidthStep + x] + pchSrc[iWidthStep + x + 1] - pchSrc[x-1] - 2 * pchSrc[x] - pchSrc[x+1]);

		fVal = pchSrc[(iHeight-1) * iWidthStep + x - 1] + 2 * pchSrc[(iHeight-1) * iWidthStep + x] + pchSrc[(iHeight-1) * iWidthStep + x + 1];
		fVal += -pchSrc[(iHeight-2) * iWidthStep + x - 1] - 2 * pchSrc[(iHeight-2) * iWidthStep + x] - pchSrc[(iHeight-2) * iWidthStep + x + 1];

		pfDst[(iHeight-1) * iWidthStep + x] = fVal * fScale;
	}

	y = iGlobalThreadid2D;

	if( y >=1 && y < iHeight-1 ) // do the vertical borders
	{
		pfDst[y * iWidthStep] = fScale * (pchSrc[(y + 1) * iWidthStep] + 2 * pchSrc[(y + 1) * iWidthStep] + pchSrc[(y + 1) * iWidthStep + 1] - pchSrc[(y - 1) * iWidthStep] - 2 * pchSrc[(y - 1) * iWidthStep] - pchSrc[(y - 1) * iWidthStep + 1]);
		pfDst[y * iWidthStep + (iWidth-1)] =  fScale * (pchSrc[(y + 1) * iWidthStep + (iWidth-2)] + 2 * pchSrc[(y + 1) * iWidthStep + (iWidth-1)] + pchSrc[(y + 1) * iWidthStep + (iWidth-1)] - pchSrc[(y - 1) * iWidthStep + (iWidth-2)] - 2 * pchSrc[(y - 1) * iWidthStep + (iWidth-1)] - pchSrc[(y - 1) * iWidthStep + (iWidth-1)]);
	}

	if( iGlobalThreadid2D == 0 ) // do the corners
	{
		pfDst[0] = fScale * (pchSrc[iWidthStep] + 2 * pchSrc[iWidthStep] + pchSrc[iWidthStep + 1] - pchSrc[0] - 2 * pchSrc[0] - pchSrc[1]); // top left corner
		pfDst[iWidth-1] = fScale * (pchSrc[iWidthStep + iWidth - 2] + 2 * pchSrc[iWidthStep + iWidth-1] + pchSrc[iWidthStep + iWidth-1] - pchSrc[iWidth-2] - 2 * pchSrc[iWidth-1] - pchSrc[iWidth-1]); // top right corner
		pfDst[(iHeight-1) * iWidthStep] = fScale * (pchSrc[(iHeight-1) * iWidthStep] + 2 * pchSrc[(iHeight-1) * iWidthStep] + pchSrc[(iHeight-1) * iWidthStep + 1] - pchSrc[(iHeight-2) * iWidthStep] - 2 * pchSrc[(iHeight-2) * iWidthStep] - pchSrc[(iHeight-2) * iWidthStep + 1]); // bottom left corner
		pfDst[(iHeight-1) * iWidthStep + (iWidth-1)] = fScale * (pchSrc[(iHeight-1) * iWidthStep + (iWidth-2)] + 2 * pchSrc[(iHeight-1) * iWidthStep + (iWidth-1)] + pchSrc[(iHeight-1) * iWidthStep + (iWidth - 1)] - pchSrc[(iHeight-2) * iWidthStep + (iWidth-2)] - 2 * pchSrc[(iHeight-2) * iWidthStep + (iWidth-1)] - pchSrc[(iHeight-2) * iWidthStep + (iWidth - 1)]); // bottom right corner
	}
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// 
//  1D Sobel below, 3D above
//
//
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__global__ void CuSobelDxMid( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );
__global__ void CuSobelDxBorders( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );
__global__ void CuSobelDyMid( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );
__global__ void CuSobelDyBorders( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );

__global__ void CuSobelDxMid( float* pfSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );
__global__ void CuSobelDxBorders( float* pfSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );
__global__ void CuSobelDyMid( float* pfSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );
__global__ void CuSobelDyBorders( float* pfSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale );

void CuSobelDxUF( void* pvSrc, void* pvDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	dim3 dimGrid;
	dim3 dimBlock;
	
	dimBlock.y = 6;
	dimBlock.x = 32;


	dimGrid.x = (int)ceil((float)iWidth/(float)dimBlock.x);
	dimGrid.y = (int)ceil((float)iHeight/(float)dimBlock.y);

	CuSobelDxMid<<<dimGrid, dimBlock>>>( (unsigned char*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );


	dimGrid.x = min( 65535, (int)ceil((float)iHeight/(float)(dimBlock.x * dimBlock.y)) );
	dimGrid.y = (int)ceil((float)iHeight/(float)(dimBlock.x * dimBlock.y * dimGrid.x));


	CuSobelDxBorders<<<dimGrid, dimBlock>>>( (unsigned char*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );
	
}


void CuSobelDyUF( void* pvSrc, void* pvDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	dim3 dimGrid;
	dim3 dimBlock;
	
	dimBlock.y = 6;
	dimBlock.x = 32;
	dimGrid.x = (int)ceil((float)iWidth/(float)dimBlock.x);
	dimGrid.y = (int)ceil((float)iHeight/(float)dimBlock.y);


	CuSobelDyMid<<<dimGrid, dimBlock>>>( (unsigned char*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );


	dimGrid.x = min( 65535, (int)ceil((float)iWidth/(float)(dimBlock.x * dimBlock.y)) );
	dimGrid.y = (int)ceil((float)iWidth/(float)(dimBlock.x * dimBlock.y * dimGrid.x));


	CuSobelDyBorders<<<dimGrid, dimBlock>>>( (unsigned char*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );
	
}

void CuSobelDxFF( void* pvSrc, void* pvDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	dim3 dimGrid;
	dim3 dimBlock;
	
	dimBlock.y = 6;
	dimBlock.x = 32;


	dimGrid.x = (int)ceil((float)iWidth/(float)dimBlock.x);
	dimGrid.y = (int)ceil((float)iHeight/(float)dimBlock.y);

	CuSobelDxMid<<<dimGrid, dimBlock>>>( (float*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );


	dimGrid.x = min( 65535, (int)ceil((float)iHeight/(float)(dimBlock.x * dimBlock.y)) );
	dimGrid.y = (int)ceil((float)iHeight/(float)(dimBlock.x * dimBlock.y * dimGrid.x));


	CuSobelDxBorders<<<dimGrid, dimBlock>>>( (float*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );
}

void CuSobelDyFF( void* pvSrc, void* pvDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	dim3 dimGrid;
	dim3 dimBlock;
	
	dimBlock.y = 6;
	dimBlock.x = 32;
	dimGrid.x = (int)ceil((float)iWidth/(float)dimBlock.x);
	dimGrid.y = (int)ceil((float)iHeight/(float)dimBlock.y);


	CuSobelDyMid<<<dimGrid, dimBlock>>>( (float*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );


	dimGrid.x = min( 65535, (int)ceil((float)iWidth/(float)(dimBlock.x * dimBlock.y)) );
	dimGrid.y = (int)ceil((float)iWidth/(float)(dimBlock.x * dimBlock.y * dimGrid.x));


	CuSobelDyBorders<<<dimGrid, dimBlock>>>( (float*)pvSrc, (float*)pvDst, iHeight, iWidth, iWidthStep, fScale );
}


__constant__  __device__ float rgDxDyFilter[3] = { -1, 0, 1 };

__global__ void CuSobelDxMid( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i =  y * iWidthStep + x;
	

	if( x <= 0 || x >= (iWidth-1) || y > (iHeight-1) )
	{
		return;
	}

	pfDst[i] = fScale * (pchSrc[i+1] - pchSrc[i-1]);

	/*
	// General version
	float fVal0;
	float fVal1;
	float fVal2;

	fVal0 = pfSrc[i-1] * rgDxDyFilter[0];
	fVal1 = pfSrc[i]   * rgDxDyFilter[1];
	fVal2 = pfSrc[i+1] * rgDxDyFilter[2];

	pfDst[i] = fVal0 + fVal1 + fVal2;
	*/

}

__global__ void CuSobelDxBorders( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	int iGlobalThreadid2D;
	int iIndex;

	iGlobalThreadid2D = blockIdx.y * gridDim.x * blockDim.y * blockDim.x; // total threads in all blocks of previous row in grid
	iGlobalThreadid2D += blockIdx.x * blockDim.y * blockDim.x; // total threads is all previous blocks in current row of grid
	iGlobalThreadid2D += threadIdx.y * blockDim.x + threadIdx.x; // thread id in current block

	if( iGlobalThreadid2D >= iHeight )
	{
		return;
	}

	iIndex = iGlobalThreadid2D * iWidthStep;
	//
	// BORDER_REPLICATE
	//
	pfDst[iIndex] = fScale * (pchSrc[iIndex+1] - pchSrc[iIndex]);
	pfDst[iIndex + (iWidth-1)] = fScale * (pchSrc[iIndex + (iWidth-1)] - pchSrc[iIndex + (iWidth-2)]);
}



__global__ void CuSobelDyMid( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i =  y * iWidthStep + x;
	

	if( x > (iWidth-1) || y <=0 || y >= (iHeight-1) )
	{
		return;
	}

	pfDst[i] = fScale * (pchSrc[i+iWidthStep] - pchSrc[i-iWidthStep]);

	/*
	// General version
	float fVal0;
	float fVal1;
	float fVal2;

	fVal0 = pfSrc[i-iWidth] * rgDxDyFilter[0];
	fVal1 = pfSrc[i]   * rgDxDyFilter[1];
	fVal2 = pfSrc[i+iWidth] * rgDxDyFilter[2];

	pfDst[i] = fVal0 + fVal1 + fVal2;
	*/
	

}

__global__ void CuSobelDyBorders( unsigned char* pchSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	int iGlobalThreadid2D;
	int iBottomIndex;

	
	iGlobalThreadid2D = blockIdx.y * gridDim.x * blockDim.y * blockDim.x; // total threads in all blocks of previous row in grid
	iGlobalThreadid2D += blockIdx.x * blockDim.y * blockDim.x; // total threads is all previous blocks in current row of grid
	iGlobalThreadid2D += threadIdx.y * blockDim.x + threadIdx.x; // thread id in current block

	iBottomIndex = (iHeight-1) * iWidthStep + iGlobalThreadid2D;

	if( iGlobalThreadid2D >= iWidth )
	{
		return;
	}

	//
	// BORDER_REPLICATE
	//
	pfDst[iGlobalThreadid2D] = fScale * (pchSrc[iGlobalThreadid2D+iWidthStep] - pchSrc[iGlobalThreadid2D]);
	pfDst[iBottomIndex] = fScale * (pchSrc[iBottomIndex] - pchSrc[iBottomIndex-iWidthStep]);

}


__global__ void CuSobelDxMid( float* pfSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i =  y * iWidthStep + x;
	

	if( x <= 0 || x >= (iWidth-1) || y > (iHeight-1) )
	{
		return;
	}

	pfDst[i] = fScale * (pfSrc[i+1] - pfSrc[i-1]);

}


__global__ void CuSobelDxBorders( float* pfSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	int iGlobalThreadid2D;
	int iIndex;

	iGlobalThreadid2D = blockIdx.y * gridDim.x * blockDim.y * blockDim.x; // total threads in all blocks of previous row in grid
	iGlobalThreadid2D += blockIdx.x * blockDim.y * blockDim.x; // total threads is all previous blocks in current row of grid
	iGlobalThreadid2D += threadIdx.y * blockDim.x + threadIdx.x; // thread id in current block

	if( iGlobalThreadid2D >= iHeight )
	{
		return;
	}

	iIndex = iGlobalThreadid2D * iWidthStep;
	//
	// BORDER_REPLICATE
	//
	pfDst[iIndex] = fScale * (pfSrc[iIndex+1] - pfSrc[iIndex]);
	pfDst[iIndex + (iWidth-1)] = fScale * (pfSrc[iIndex + (iWidth-1)] - pfSrc[iIndex + (iWidth-2)]);

}

__global__ void CuSobelDyMid( float* pfSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i =  y * iWidthStep + x;
	

	if( x > (iWidth-1) || y <=0 || y >= (iHeight-1) )
	{
		return;
	}

	pfDst[i] = fScale * (pfSrc[i+iWidthStep] - pfSrc[i-iWidthStep]);
	
}

__global__ void CuSobelDyBorders( float* pfSrc, float* pfDst, int iHeight, int iWidth, int iWidthStep, float fScale )
{
	int iGlobalThreadid2D;
	int iBottomIndex;

	
	iGlobalThreadid2D = blockIdx.y * gridDim.x * blockDim.y * blockDim.x; // total threads in all blocks of previous row in grid
	iGlobalThreadid2D += blockIdx.x * blockDim.y * blockDim.x; // total threads is all previous blocks in current row of grid
	iGlobalThreadid2D += threadIdx.y * blockDim.x + threadIdx.x; // thread id in current block

	iBottomIndex = (iHeight-1) * iWidthStep + iGlobalThreadid2D;

	if( iGlobalThreadid2D >= iWidth )
	{
		return;
	}

	//
	// BORDER_REPLICATE
	//
	pfDst[iGlobalThreadid2D] = fScale * (pfSrc[iGlobalThreadid2D+iWidthStep] - pfSrc[iGlobalThreadid2D]);
	pfDst[iBottomIndex] = fScale * (pfSrc[iBottomIndex] - pfSrc[iBottomIndex-iWidthStep]);

}
