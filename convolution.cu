#include <windows.h>
#include <assert.h>
#include <stdio.h>

int AllocateMemoryOnGPU(void** ppvMemory, int iSize);
void FreeMemoryOnGPU(void* pvMemory);
int CopyDataToGPU(void* pvGPU, void* pvHost, int iSize);
int CopyDataFromGPU(void* pvHost, void* pvGPU, int iSize);
int SetDevice(int iDevice);

__global__ void Convolve_15x5(float* pfImage, float* pfFilter, float* pfResponse, int iHeight, int iWidth, int iFilterHeight, int iFilterWidth);
__global__ void Convolve_15x5SM(float* pfImage, float* pfFilter, float* pfResponse, int iHeight, int iWidth, int iFilterHeight, int iFilterWidth);
__global__ void Convolve_15x5SMTexture(float* pfFilter, float* pfResponse, int iHeight, int iWidth, int iFilterHeight, int iFilterWidth);
__global__ void Convolve_15x5SMTexturePitched(float* pfFilter, float* pfResponse, int iHeight, int iWidth, int iFilterHeight, int iFilterWidth);
__global__ void Convolve_15x5SMW(float* pfImage, float* pfFilter, float* pfResponse, int iHeight, int iWidth, int iFilterHeight, int iFilterWidth); // one warp per pixel
float Convolve(float* pfImageFeatures, int iHeight, int iWidth, float* pfFilter, int iFilterHeight, int iFilterWidth, int iFilterDepth, int iLoc_y, int iLoc_x);

texture <float, cudaTextureType2D, cudaReadModeElementType> texImage;
cudaArray* d_imageArray;

texture <float, cudaTextureType2D, cudaReadModeElementType> texImage2;




__device__ float ConvRow(int x, int y, float* pfFilter)
{
	float a = tex2D(texImage, x, y) * pfFilter[0];
	float b = tex2D(texImage, x + 1, y) * pfFilter[1];
	float c = tex2D(texImage, x + 2, y) * pfFilter[2];
	float d = tex2D(texImage, x + 3, y) * pfFilter[3];
	float e = tex2D(texImage, x + 4, y) * pfFilter[4];

	return a + b + c + d + e;
}


__global__ void Convolve_15x5SMTexture(float* pfFilter, float* pfResponse, int iHeight, int iWidth, int iFilterHeight, int iFilterWidth)
{
	__shared__ float rgFilter[15 * 5];

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = y * iWidth + x;

	float fAccum;


	int iBlockThreadId = threadIdx.y * blockDim.x + threadIdx.x; // thread id within block


	if (iBlockThreadId < 75)
	{
		rgFilter[iBlockThreadId] = pfFilter[iBlockThreadId];
	}
	__syncthreads();


	pfFilter = rgFilter;

	if ((x  < iWidth - iFilterWidth) && (y < iHeight - iFilterHeight))
	{
		fAccum = 0;
		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;

		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		fAccum += ConvRow(x, y, pfFilter);
		y++;
		pfFilter += 5;


		pfResponse[i] = fAccum;

	}
}



__device__ float ConvRowPitched(int x, int y, float* pfFilter, int iThreadId)
{
	int iOffset0 = iThreadId % 5;
	int iOffset1 = (iThreadId + 1) % 5;
	int iOffset2 = (iThreadId + 2) % 5;
	int iOffset3 = (iThreadId + 3) % 5;
	int iOffset4 = (iThreadId + 4) % 5;


	float a = tex2D(texImage2, x + iOffset0, y) * pfFilter[iOffset0];
	float b = tex2D(texImage2, x + iOffset1, y) * pfFilter[iOffset1];
	float c = tex2D(texImage2, x + iOffset2, y) * pfFilter[iOffset2];
	float d = tex2D(texImage2, x + iOffset3, y) * pfFilter[iOffset3];
	float e = tex2D(texImage2, x + iOffset4, y) * pfFilter[iOffset4];


	/*
	float a = tex2D(texImage2, x, y ) * pfFilter[0];
	float b = tex2D(texImage2, x+1, y ) * pfFilter[1];
	float c = tex2D(texImage2, x+2, y ) * pfFilter[2];
	float d = tex2D(texImage2, x+3, y ) * pfFilter[3];
	float e = tex2D(texImage2, x+4, y ) * pfFilter[4];
	*/

	return a + b + c + d + e;

}




__global__ void Convolve_15x5SMTexturePitched(float* pfFilter, float* pfResponse, int iHeight, int iWidth, int iFilterHeight, int iFilterWidth)
{
	__shared__ float rgFilter[15 * 5];

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = y * iWidth + x;

	float fAccum;


	int iBlockThreadId = threadIdx.y * blockDim.x + threadIdx.x; // thread id within block


	if (iBlockThreadId < 75)
	{
		rgFilter[iBlockThreadId] = pfFilter[iBlockThreadId];
	}
	__syncthreads();



	pfFilter = rgFilter;

	if ((x  < iWidth - iFilterWidth) && (y < iHeight - iFilterHeight))
	{
		fAccum = 0;
		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;

		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;


		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;


		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;


		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;


		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;


		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;


		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;


		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;


		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;


		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;


		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;


		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;


		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;

		fAccum += ConvRowPitched(x, y, pfFilter, iBlockThreadId);
		y++;
		pfFilter += 5;

		pfResponse[i] = fAccum;

	}
}


__global__ void Convolve_15x5(float* pfImage, float* pfFilter, float* pfResponse, int iHeight, int iWidth, int iFilterHeight, int iFilterWidth)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = y * iWidth + x;

	float fVal0;
	float fVal1;
	float fVal2;
	float fVal3;
	float fVal4;
	float fAccum;

	int iRow_1;
	int iRow_2;
	int iRow_3;
	int iRow_4;


	if ((x  < iWidth - iFilterWidth) && (y < iHeight - iFilterHeight))
	{
		iRow_1 = i + iWidth;
		iRow_2 = i + iWidth * 2;
		iRow_3 = i + iWidth * 3;
		iRow_4 = i + iWidth * 4;

		fVal0 = pfImage[i] * pfFilter[0] + pfImage[i + 1] * pfFilter[1] + pfImage[i + 2] * pfFilter[2] + pfImage[i + 3] * pfFilter[3] + pfImage[i + 4] * pfFilter[4];
		fVal1 = pfImage[iRow_1] * pfFilter[5] + pfImage[iRow_1 + 1] * pfFilter[6] + pfImage[iRow_1 + 2] * pfFilter[7] + pfImage[iRow_1 + 3] * pfFilter[8] + pfImage[iRow_1 + 4] * pfFilter[9];
		fVal2 = pfImage[iRow_2] * pfFilter[10] + pfImage[iRow_2 + 1] * pfFilter[11] + pfImage[iRow_2 + 2] * pfFilter[12] + pfImage[iRow_2 + 3] * pfFilter[13] + pfImage[iRow_2 + 4] * pfFilter[14];
		fVal3 = pfImage[iRow_3] * pfFilter[15] + pfImage[iRow_3 + 1] * pfFilter[16] + pfImage[iRow_3 + 2] * pfFilter[17] + pfImage[iRow_3 + 3] * pfFilter[18] + pfImage[iRow_3 + 4] * pfFilter[19];
		fVal4 = pfImage[iRow_4] * pfFilter[20] + pfImage[iRow_4 + 1] * pfFilter[21] + pfImage[iRow_4 + 2] * pfFilter[22] + pfImage[iRow_4 + 3] * pfFilter[23] + pfImage[iRow_4 + 4] * pfFilter[24];

		i += iWidth * 5;
		pfFilter += (5 * 5);
		fAccum = fVal0 + fVal1 + fVal2 + fVal3 + fVal4;


		iRow_1 = i + iWidth;
		iRow_2 = i + iWidth * 2;
		iRow_3 = i + iWidth * 3;
		iRow_4 = i + iWidth * 4;

		fVal0 = pfImage[i] * pfFilter[0] + pfImage[i + 1] * pfFilter[1] + pfImage[i + 2] * pfFilter[2] + pfImage[i + 3] * pfFilter[3] + pfImage[i + 4] * pfFilter[4];
		fVal1 = pfImage[iRow_1] * pfFilter[5] + pfImage[iRow_1 + 1] * pfFilter[6] + pfImage[iRow_1 + 2] * pfFilter[7] + pfImage[iRow_1 + 3] * pfFilter[8] + pfImage[iRow_1 + 4] * pfFilter[9];
		fVal2 = pfImage[iRow_2] * pfFilter[10] + pfImage[iRow_2 + 1] * pfFilter[11] + pfImage[iRow_2 + 2] * pfFilter[12] + pfImage[iRow_2 + 3] * pfFilter[13] + pfImage[iRow_2 + 4] * pfFilter[14];
		fVal3 = pfImage[iRow_3] * pfFilter[15] + pfImage[iRow_3 + 1] * pfFilter[16] + pfImage[iRow_3 + 2] * pfFilter[17] + pfImage[iRow_3 + 3] * pfFilter[18] + pfImage[iRow_3 + 4] * pfFilter[19];
		fVal4 = pfImage[iRow_4] * pfFilter[20] + pfImage[iRow_4 + 1] * pfFilter[21] + pfImage[iRow_4 + 2] * pfFilter[22] + pfImage[iRow_4 + 3] * pfFilter[23] + pfImage[iRow_4 + 4] * pfFilter[24];

		i += iWidth * 5;
		pfFilter += (5 * 5);

		fAccum += fVal0 + fVal1 + fVal2 + fVal3 + fVal4;


		iRow_1 = i + iWidth;
		iRow_2 = i + iWidth * 2;
		iRow_3 = i + iWidth * 3;
		iRow_4 = i + iWidth * 4;

		fVal0 = pfImage[i] * pfFilter[0] + pfImage[i + 1] * pfFilter[1] + pfImage[i + 2] * pfFilter[2] + pfImage[i + 3] * pfFilter[3] + pfImage[i + 4] * pfFilter[4];
		fVal1 = pfImage[iRow_1] * pfFilter[5] + pfImage[iRow_1 + 1] * pfFilter[6] + pfImage[iRow_1 + 2] * pfFilter[7] + pfImage[iRow_1 + 3] * pfFilter[8] + pfImage[iRow_1 + 4] * pfFilter[9];
		fVal2 = pfImage[iRow_2] * pfFilter[10] + pfImage[iRow_2 + 1] * pfFilter[11] + pfImage[iRow_2 + 2] * pfFilter[12] + pfImage[iRow_2 + 3] * pfFilter[13] + pfImage[iRow_2 + 4] * pfFilter[14];
		fVal3 = pfImage[iRow_3] * pfFilter[15] + pfImage[iRow_3 + 1] * pfFilter[16] + pfImage[iRow_3 + 2] * pfFilter[17] + pfImage[iRow_3 + 3] * pfFilter[18] + pfImage[iRow_3 + 4] * pfFilter[19];
		fVal4 = pfImage[iRow_4] * pfFilter[20] + pfImage[iRow_4 + 1] * pfFilter[21] + pfImage[iRow_4 + 2] * pfFilter[22] + pfImage[iRow_4 + 3] * pfFilter[23] + pfImage[iRow_4 + 4] * pfFilter[24];

		i -= iWidth * 10;
		fAccum += fVal0 + fVal1 + fVal2 + fVal3 + fVal4;


		pfResponse[i] = fAccum;
	}

}



__global__ void Convolve_15x5SMTT(float* pfImage, float* pfFilter, float* pfResponse, int iHeight, int iWidth, int iFilterHeight, int iFilterWidth)
{
	__shared__ float rgFilter[15 * 5];

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = y * iWidth + x;

	float fVal0;
	float fVal1;
	float fVal2;
	float fVal3;
	float fVal4;
	float fAccum;

	int iRow_1;
	int iRow_2;
	int iRow_3;
	int iRow_4;


	int iBlockThreadId = threadIdx.y * blockDim.x + threadIdx.x; // thread id within block


	if (iBlockThreadId < 75)
	{
		rgFilter[iBlockThreadId] = pfFilter[iBlockThreadId];
	}


	__syncthreads();

	pfFilter = rgFilter;

	if ((x  < iWidth - iFilterWidth) && (y < iHeight - iFilterHeight))
	{
		iRow_1 = i + iWidth;
		iRow_2 = i + iWidth * 2;
		iRow_3 = i + iWidth * 3;
		iRow_4 = i + iWidth * 4;

		fVal0 = pfImage[i] * pfFilter[0] + pfImage[i + 1] * pfFilter[1] + pfImage[i + 2] * pfFilter[2] + pfImage[i + 3] * pfFilter[3] + pfImage[i + 4] * pfFilter[4];
		fVal1 = pfImage[iRow_1] * pfFilter[5] + pfImage[iRow_1 + 1] * pfFilter[6] + pfImage[iRow_1 + 2] * pfFilter[7] + pfImage[iRow_1 + 3] * pfFilter[8] + pfImage[iRow_1 + 4] * pfFilter[9];
		fVal2 = pfImage[iRow_2] * pfFilter[10] + pfImage[iRow_2 + 1] * pfFilter[11] + pfImage[iRow_2 + 2] * pfFilter[12] + pfImage[iRow_2 + 3] * pfFilter[13] + pfImage[iRow_2 + 4] * pfFilter[14];
		fVal3 = pfImage[iRow_3] * pfFilter[15] + pfImage[iRow_3 + 1] * pfFilter[16] + pfImage[iRow_3 + 2] * pfFilter[17] + pfImage[iRow_3 + 3] * pfFilter[18] + pfImage[iRow_3 + 4] * pfFilter[19];
		fVal4 = pfImage[iRow_4] * pfFilter[20] + pfImage[iRow_4 + 1] * pfFilter[21] + pfImage[iRow_4 + 2] * pfFilter[22] + pfImage[iRow_4 + 3] * pfFilter[23] + pfImage[iRow_4 + 4] * pfFilter[24];

		fAccum = fVal0 + fVal1 + fVal2 + fVal3 + fVal4;

		i += iWidth * 5;
		pfFilter += (5 * 5);

		iRow_1 = i + iWidth;
		iRow_2 = i + iWidth * 2;
		iRow_3 = i + iWidth * 3;
		iRow_4 = i + iWidth * 4;

		fVal0 = pfImage[i] * pfFilter[0] + pfImage[i + 1] * pfFilter[1] + pfImage[i + 2] * pfFilter[2] + pfImage[i + 3] * pfFilter[3] + pfImage[i + 4] * pfFilter[4];
		fVal1 = pfImage[iRow_1] * pfFilter[5] + pfImage[iRow_1 + 1] * pfFilter[6] + pfImage[iRow_1 + 2] * pfFilter[7] + pfImage[iRow_1 + 3] * pfFilter[8] + pfImage[iRow_1 + 4] * pfFilter[9];
		fVal2 = pfImage[iRow_2] * pfFilter[10] + pfImage[iRow_2 + 1] * pfFilter[11] + pfImage[iRow_2 + 2] * pfFilter[12] + pfImage[iRow_2 + 3] * pfFilter[13] + pfImage[iRow_2 + 4] * pfFilter[14];
		fVal3 = pfImage[iRow_3] * pfFilter[15] + pfImage[iRow_3 + 1] * pfFilter[16] + pfImage[iRow_3 + 2] * pfFilter[17] + pfImage[iRow_3 + 3] * pfFilter[18] + pfImage[iRow_3 + 4] * pfFilter[19];
		fVal4 = pfImage[iRow_4] * pfFilter[20] + pfImage[iRow_4 + 1] * pfFilter[21] + pfImage[iRow_4 + 2] * pfFilter[22] + pfImage[iRow_4 + 3] * pfFilter[23] + pfImage[iRow_4 + 4] * pfFilter[24];

		fAccum += fVal0 + fVal1 + fVal2 + fVal3 + fVal4;

		i += iWidth * 5;
		pfFilter += (5 * 5);

		iRow_1 = i + iWidth;
		iRow_2 = i + iWidth * 2;
		iRow_3 = i + iWidth * 3;
		iRow_4 = i + iWidth * 4;

		fVal0 = pfImage[i] * pfFilter[0] + pfImage[i + 1] * pfFilter[1] + pfImage[i + 2] * pfFilter[2] + pfImage[i + 3] * pfFilter[3] + pfImage[i + 4] * pfFilter[4];
		fVal1 = pfImage[iRow_1] * pfFilter[5] + pfImage[iRow_1 + 1] * pfFilter[6] + pfImage[iRow_1 + 2] * pfFilter[7] + pfImage[iRow_1 + 3] * pfFilter[8] + pfImage[iRow_1 + 4] * pfFilter[9];
		fVal2 = pfImage[iRow_2] * pfFilter[10] + pfImage[iRow_2 + 1] * pfFilter[11] + pfImage[iRow_2 + 2] * pfFilter[12] + pfImage[iRow_2 + 3] * pfFilter[13] + pfImage[iRow_2 + 4] * pfFilter[14];
		fVal3 = pfImage[iRow_3] * pfFilter[15] + pfImage[iRow_3 + 1] * pfFilter[16] + pfImage[iRow_3 + 2] * pfFilter[17] + pfImage[iRow_3 + 3] * pfFilter[18] + pfImage[iRow_3 + 4] * pfFilter[19];
		fVal4 = pfImage[iRow_4] * pfFilter[20] + pfImage[iRow_4 + 1] * pfFilter[21] + pfImage[iRow_4 + 2] * pfFilter[22] + pfImage[iRow_4 + 3] * pfFilter[23] + pfImage[iRow_4 + 4] * pfFilter[24];

		fAccum += fVal0 + fVal1 + fVal2 + fVal3 + fVal4;


		pfResponse[i - iWidth * 10] = fAccum;
	}

}



__device__ float ConvRow(float* pfImage, float* pfFilter)
{
	float a = pfImage[0] * pfFilter[0];
	float b = pfImage[1] * pfFilter[1];
	float c = pfImage[2] * pfFilter[2];
	float d = pfImage[3] * pfFilter[3];
	float e = pfImage[4] * pfFilter[4];

	return a + b + c + d + e;
}

__global__ void Convolve_15x5SM(float* pfImage, float* pfFilter, float* pfResponse, int iHeight, int iWidth, int iFilterHeight, int iFilterWidth)
{
	__shared__ float rgFilter[15 * 5];

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = y * iWidth + x;

	float fVal0;
	float fVal1;
	float fVal2;
	float fAccum;


	int iBlockThreadId = threadIdx.y * blockDim.x + threadIdx.x; // thread id within block


	if (iBlockThreadId < 75)
	{
		rgFilter[iBlockThreadId] = pfFilter[iBlockThreadId];
	}
	__syncthreads();


	pfFilter = rgFilter;
	pfImage += i;

	if ((x  < iWidth - iFilterWidth) && (y < iHeight - iFilterHeight))
	{
		fVal0 = ConvRow(pfImage, pfFilter);
		fVal1 = ConvRow(pfImage + iWidth, pfFilter + 5);
		fVal2 = ConvRow(pfImage + iWidth * 2, pfFilter + 10);

		fAccum = fVal0 + fVal1 + fVal2;

		pfImage += iWidth * 3;
		pfFilter += 15;

		fVal0 = ConvRow(pfImage, pfFilter);
		fVal1 = ConvRow(pfImage + iWidth, pfFilter + 5);
		fVal2 = ConvRow(pfImage + iWidth * 2, pfFilter + 10);

		fAccum += fVal0 + fVal1 + fVal2;

		pfImage += iWidth * 3;
		pfFilter += 15;

		fVal0 = ConvRow(pfImage, pfFilter);
		fVal1 = ConvRow(pfImage + iWidth, pfFilter + 5);
		fVal2 = ConvRow(pfImage + iWidth * 2, pfFilter + 10);

		fAccum += fVal0 + fVal1 + fVal2;


		pfImage += iWidth * 3;
		pfFilter += 15;

		fVal0 = ConvRow(pfImage, pfFilter);
		fVal1 = ConvRow(pfImage + iWidth, pfFilter + 5);
		fVal2 = ConvRow(pfImage + iWidth * 2, pfFilter + 10);

		fAccum += fVal0 + fVal1 + fVal2;


		pfImage += iWidth * 3;
		pfFilter += 15;

		fVal0 = ConvRow(pfImage, pfFilter);
		fVal1 = ConvRow(pfImage + iWidth, pfFilter + 5);
		fVal2 = ConvRow(pfImage + iWidth * 2, pfFilter + 10);

		fAccum += fVal0 + fVal1 + fVal2;

		pfResponse[i] = fAccum;

	}

	/*
	if( (x  < iWidth - iFilterWidth ) && (y < iHeight - iFilterHeight ) )
	{
	fVal0 = ConvRow( pfImage + i, pfFilter );
	fVal1 = ConvRow( pfImage + i + iWidth, pfFilter + 5 );
	fVal2 = ConvRow( pfImage + i + iWidth * 2, pfFilter + 10 );

	fAccum = fVal0 + fVal1 + fVal2;

	pfImage += iWidth * 3;
	pfFilter += 15;

	fVal0 = ConvRow( pfImage + i, pfFilter );
	fVal1 = ConvRow( pfImage + i + iWidth, pfFilter + 5 );
	fVal2 = ConvRow( pfImage + i + iWidth * 2, pfFilter + 10 );

	fAccum += fVal0 + fVal1 + fVal2;

	pfImage += iWidth * 3;
	pfFilter += 15;

	fVal0 = ConvRow( pfImage + i, pfFilter );
	fVal1 = ConvRow( pfImage + i + iWidth, pfFilter + 5 );
	fVal2 = ConvRow( pfImage + i + iWidth * 2, pfFilter + 10 );

	fAccum += fVal0 + fVal1 + fVal2;


	pfImage += iWidth * 3;
	pfFilter += 15;

	fVal0 = ConvRow( pfImage + i, pfFilter );
	fVal1 = ConvRow( pfImage + i + iWidth, pfFilter + 5 );
	fVal2 = ConvRow( pfImage + i + iWidth * 2, pfFilter + 10 );

	fAccum += fVal0 + fVal1 + fVal2;


	pfImage += iWidth * 3;
	pfFilter += 15;

	fVal0 = ConvRow( pfImage + i, pfFilter );
	fVal1 = ConvRow( pfImage + i + iWidth, pfFilter + 5 );
	fVal2 = ConvRow( pfImage + i + iWidth * 2, pfFilter + 10 );

	fAccum += fVal0 + fVal1 + fVal2;

	pfResponse[i] = fAccum;

	}
	*/

}




__global__ void Convolve_15x5SMW(float* pfImage, float* pfFilter, float* pfResponse, int iHeight, int iWidth, int iFilterHeight, int iFilterWidth)
{
	__shared__ float rgFilter[15 * 5];

	int iWarpThreadId;
	int x_filter;
	int y_filter;
	float fVal;
	unsigned int x = blockIdx.x * 4;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = y * iWidth + x;


	int iBlockThreadId = threadIdx.y * blockDim.x + threadIdx.x; // thread id within block
	iWarpThreadId = iBlockThreadId % 32;

	if (iBlockThreadId < 75)
	{
		rgFilter[iBlockThreadId] = pfFilter[iBlockThreadId];
	}
	if (iWarpThreadId == 0)
	{
		pfResponse[i] = 0;
		pfResponse[i + 1] = 0;
		pfResponse[i + 2] = 0;
		pfResponse[i + 3] = 0;
	}
	__syncthreads();


	pfFilter = rgFilter;

	if ((x  < iWidth - iFilterWidth) && (y < iHeight - iFilterHeight))
	{
		x_filter = iWarpThreadId%iFilterWidth;;
		y_filter = iWarpThreadId / iFilterWidth;
		fVal = pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];


		x_filter = (iWarpThreadId + 32) % iFilterWidth;;
		y_filter = (iWarpThreadId + 32) / iFilterWidth;
		fVal += pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];

		if (iWarpThreadId < 11)
		{
			x_filter = (iWarpThreadId + 64) % iFilterWidth;;
			y_filter = (iWarpThreadId + 64) / iFilterWidth;
			fVal += pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];
		}

		atomicAdd(&pfResponse[i], fVal);
	}



	x = blockIdx.x * 4 + 1;
	i = y * iWidth + x;

	pfFilter = rgFilter;

	if ((x  < iWidth - iFilterWidth) && (y < iHeight - iFilterHeight))
	{
		x_filter = iWarpThreadId%iFilterWidth;;
		y_filter = iWarpThreadId / iFilterWidth;
		fVal = pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];


		x_filter = (iWarpThreadId + 32) % iFilterWidth;;
		y_filter = (iWarpThreadId + 32) / iFilterWidth;
		fVal += pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];

		if (iWarpThreadId < 11)
		{
			x_filter = (iWarpThreadId + 64) % iFilterWidth;;
			y_filter = (iWarpThreadId + 64) / iFilterWidth;
			fVal += pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];
		}

		atomicAdd(&pfResponse[i], fVal);
	}


	x = blockIdx.x * 4 + 2;
	i = y * iWidth + x;

	pfFilter = rgFilter;

	if ((x  < iWidth - iFilterWidth) && (y < iHeight - iFilterHeight))
	{
		x_filter = iWarpThreadId%iFilterWidth;;
		y_filter = iWarpThreadId / iFilterWidth;
		fVal = pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];


		x_filter = (iWarpThreadId + 32) % iFilterWidth;;
		y_filter = (iWarpThreadId + 32) / iFilterWidth;
		fVal += pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];

		if (iWarpThreadId < 11)
		{
			x_filter = (iWarpThreadId + 64) % iFilterWidth;;
			y_filter = (iWarpThreadId + 64) / iFilterWidth;
			fVal += pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];
		}

		atomicAdd(&pfResponse[i], fVal);
	}


	x = blockIdx.x * 4 + 3;
	i = y * iWidth + x;

	pfFilter = rgFilter;

	if ((x  < iWidth - iFilterWidth) && (y < iHeight - iFilterHeight))
	{
		x_filter = iWarpThreadId%iFilterWidth;;
		y_filter = iWarpThreadId / iFilterWidth;
		fVal = pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];


		x_filter = (iWarpThreadId + 32) % iFilterWidth;;
		y_filter = (iWarpThreadId + 32) / iFilterWidth;
		fVal += pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];

		if (iWarpThreadId < 11)
		{
			x_filter = (iWarpThreadId + 64) % iFilterWidth;;
			y_filter = (iWarpThreadId + 64) / iFilterWidth;
			fVal += pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];
		}

		atomicAdd(&pfResponse[i], fVal);
	}


}





__global__ void Convolve_15x5SMWXX(float* pfImage, float* pfFilter, float* pfResponse, int iHeight, int iWidth, int iFilterHeight, int iFilterWidth)
{
	__shared__ float rgFilter[15 * 5];

	int iWarpThreadId;
	int x_filter;
	int y_filter;
	float fVal;
	unsigned int x = blockIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = y * iWidth + x;


	int iBlockThreadId = threadIdx.y * blockDim.x + threadIdx.x; // thread id within block
	iWarpThreadId = iBlockThreadId % 32;

	if (iBlockThreadId < 75)
	{
		rgFilter[iBlockThreadId] = pfFilter[iBlockThreadId];
	}
	if (iWarpThreadId == 0)
	{
		pfResponse[i] = 0;
	}
	__syncthreads();


	pfFilter = rgFilter;


	if ((x  < iWidth - iFilterWidth) && (y < iHeight - iFilterHeight))
	{
		x_filter = iWarpThreadId%iFilterWidth;;
		y_filter = iWarpThreadId / iFilterWidth;
		fVal = pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];


		x_filter = (iWarpThreadId + 32) % iFilterWidth;;
		y_filter = (iWarpThreadId + 32) / iFilterWidth;
		fVal += pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];

		if (iWarpThreadId < 11)
		{
			x_filter = (iWarpThreadId + 64) % iFilterWidth;;
			y_filter = (iWarpThreadId + 64) / iFilterWidth;
			fVal += pfImage[(y + y_filter) * iWidth + (x + x_filter)] * pfFilter[y_filter * iFilterWidth + x_filter];
		}

		atomicAdd(&pfResponse[i], fVal);
	}
}




float Convolve(float* pfImageFeatures, int iHeight, int iWidth, float* pfFilter, int iFilterHeight, int iFilterWidth, int iFilterDepth, int iLoc_y, int iLoc_x)
{
	float fVal;
	int x;
	int y;
	int z;

	fVal = 0;

	for (y = 0; y < iFilterHeight; y++)
	{
		for (x = 0; x < iFilterWidth; x++)
		{
			for (z = 0; z < iFilterDepth; z++)
			{
				fVal += pfImageFeatures[(y + iLoc_y) * iWidth * iFilterDepth + (x + iLoc_x) * iFilterDepth + z] * pfFilter[y * iFilterWidth * iFilterDepth + x * iFilterDepth + z];
			}
		}
	}

	return fVal;
}



int AllocateMemoryOnGPU(void** ppvMemory, int iSize)
{
	int iRet;
	cudaError_t cudaErr;

	cudaErr = cudaMalloc(ppvMemory, iSize);
	if (cudaSuccess != cudaErr)
	{
		assert(0);
		//LogMessage( LOGGER_SEVERITY_CRITICAL_ERROR, __FILE__, __FUNCTION__, __LINE__, "Error [%d] allocating memory", GetLastError() );
		iRet = -1;
		goto Exit;
	}

	cudaMemset(*ppvMemory, 0, iSize);

	iRet = 0;
Exit:

	return iRet;
}

void FreeMemoryOnGPU(void* pvMemory)
{
	cudaFree(pvMemory);
}

int CopyDataToGPU(void* pvGPU, void* pvHost, int iSize)
{
	int iRet;
	cudaError_t cudaErr;

	cudaErr = cudaMemcpy(pvGPU, pvHost, iSize, cudaMemcpyHostToDevice);
	if (cudaSuccess != cudaErr)
	{
		assert(0);
		iRet = -1;
		goto Exit;
	}

	iRet = 0;
Exit:
	return iRet;
}


int CopyDataFromGPU(void* pvHost, void* pvGPU, int iSize)
{
	int iRet;
	cudaError_t cudaErr;


	cudaErr = cudaMemcpy(pvHost, pvGPU, iSize, cudaMemcpyDeviceToHost);
	if (cudaSuccess != cudaErr)
	{
		assert(0);
		iRet = -1;
		goto Exit;
	}

	iRet = 0;
Exit:
	return iRet;
}

void ZeroMemoryOnGPU(void* pvMemory, int iSize)
{
	cudaMemset(pvMemory, 0, iSize);
}


int SetDevice(int iDevice)
{
	int iRet;
	cudaError_t cudaErr;

	cudaErr = cudaSetDevice(iDevice);
	if (cudaSuccess != cudaErr)
	{
		assert(0);
		iRet = -1;
		goto Exit;
	}

	iRet = 0;

Exit:
	return iRet;
}





void RunTests()
{
	int i;
	float* pfImage;
	float* pfFilter;
	int iHeight;
	int iWidth;
	int iFilterHeight;
	int iFilterWidth;
	float* pfResponsesCPU;
	float* pfResponsesGPU;
	int y;
	int x;
	dim3 dimGrid;
	dim3 dimBlock;

	iHeight = 1024;
	iWidth = 4096;
	iFilterHeight = 15;
	iFilterWidth = 5;

	pfImage = new float[iHeight * iWidth];
	pfFilter = new float[iFilterHeight * iFilterWidth];

	srand(1);
	for (i = 0; i < iFilterHeight * iFilterWidth; i++)
	{
		pfFilter[i] = rand() % 100 / 100.0f;
	}

	srand(2);
	for (i = 0; i < iHeight * iWidth; i++)
	{
		pfImage[i] = rand() % 100 / 100.0f;
	}

	
	pfResponsesCPU = new float[iHeight * iWidth];
	pfResponsesGPU = new float[iHeight * iWidth];
	memset(pfResponsesCPU, 0, sizeof(float) * iHeight * iWidth);

	for (y = 0; y < iHeight - iFilterHeight; y++)
	{
		for (x = 0; x < iWidth - iFilterWidth; x++)
		{
			pfResponsesCPU[y * iWidth + x] = Convolve(pfImage, iHeight, iWidth, pfFilter, iFilterHeight, iFilterWidth, 1, y, x);
		}
	}


	dimBlock.y = 6;
	dimBlock.x = 32;

	dimGrid.x = (int)ceil((float)iWidth / (float)dimBlock.x);
	dimGrid.y = (int)ceil((float)iHeight / (float)dimBlock.y);

	void* pvImage;
	void* pvFilter;
	void* pvResponse;

	AllocateMemoryOnGPU(&pvImage, sizeof(float) * iHeight * iWidth);
	AllocateMemoryOnGPU(&pvResponse, sizeof(float) * iHeight * iWidth);
	AllocateMemoryOnGPU(&pvFilter, sizeof(float) * iFilterHeight * iFilterWidth);

	CopyDataToGPU(pvImage, pfImage, sizeof(float) * iHeight * iWidth);
	CopyDataToGPU(pvFilter, pfFilter, sizeof(float) * iFilterHeight * iFilterWidth);


	//---------------------------------------------------------------------------
	// Run Convolve_15x5
	//---------------------------------------------------------------------------
	Convolve_15x5 << < dimGrid, dimBlock >> >((float*)pvImage, (float*)pvFilter, (float*)pvResponse, iHeight, iWidth, iFilterHeight, iFilterWidth);
	
	CopyDataFromGPU(pfResponsesGPU, pvResponse, sizeof(float) * iHeight * iWidth); // copy results back from GPU
	for (y = 0; y < iHeight - iFilterHeight; y++) // compare with CPU results
	{
		for (x = 0; x < iWidth - iFilterWidth; x++)
		{
			float fDiff = fabs(pfResponsesGPU[y * iWidth + x] - pfResponsesCPU[y * iWidth + x]);
			if (fDiff > 0.0001f)
			{
				printf("\r\nCPU and GPU results do not match");
			}
		}
	}




	//---------------------------------------------------------------------------
	// Run Convolve_15x5
	//---------------------------------------------------------------------------
	ZeroMemoryOnGPU(pvResponse, sizeof(float) * iHeight * iWidth);
	Convolve_15x5SM << < dimGrid, dimBlock >> >((float*)pvImage, (float*)pvFilter, (float*)pvResponse, iHeight, iWidth, iFilterHeight, iFilterWidth);
	CopyDataFromGPU(pfResponsesGPU, pvResponse, sizeof(float) * iHeight * iWidth); // copy results back from GPU

	for (y = 0; y < iHeight - iFilterHeight; y++) // compare with CPU results
	{
		for (x = 0; x < iWidth - iFilterWidth; x++)
		{
			float fDiff = fabs(pfResponsesGPU[y * iWidth + x] - pfResponsesCPU[y * iWidth + x]);
			if (fDiff > 0.0001f)
			{
				printf("\r\nCPU and GPU results do not match");
			}
		}
	}


	//---------------------------------------------------------------------------
	// Run Convolve_15x5SMTexture
	//---------------------------------------------------------------------------
	ZeroMemoryOnGPU(pvResponse, sizeof(float) * iHeight * iWidth);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMallocArray(&d_imageArray, &channelDesc, iWidth, iHeight);
	cudaBindTextureToArray(texImage, d_imageArray);

	cudaMemcpyToArray(d_imageArray, 0, 0, pfImage, iHeight * iWidth * sizeof(float), cudaMemcpyHostToDevice);

	Convolve_15x5SMTexture << < dimGrid, dimBlock >> >((float*)pvFilter, (float*)pvResponse, iHeight, iWidth, iFilterHeight, iFilterWidth);

	CopyDataFromGPU(pfResponsesGPU, pvResponse, sizeof(float) * iHeight * iWidth); // copy results back from GPU

	for (y = 0; y < iHeight - iFilterHeight; y++) // compare with CPU results
	{
		for (x = 0; x < iWidth - iFilterWidth; x++)
		{
			float fDiff = fabs(pfResponsesGPU[y * iWidth + x] - pfResponsesCPU[y * iWidth + x]);
			if (fDiff > 0.0001f)
			{
				printf("\r\nCPU and GPU results do not match");
			}
		}
	}


	//---------------------------------------------------------------------------
	// Run Convolve_15x5SMTexturePitched
	//---------------------------------------------------------------------------
	ZeroMemoryOnGPU(pvResponse, sizeof(float) * iHeight * iWidth);
	void* pvPitchMem;
	size_t pitch;

	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMallocPitch(&pvPitchMem, &pitch, iWidth * sizeof(float), iHeight);
	cudaBindTexture2D(0, texImage2, pvPitchMem, channelDesc2, iWidth, iHeight, pitch);

	cudaMemcpy2D(pvPitchMem, pitch, pfImage, iWidth * sizeof(float), iWidth * sizeof(float), iHeight, cudaMemcpyHostToDevice);

	cudaMemset(pvResponse, 0, sizeof(float) * iHeight * iWidth);

	Convolve_15x5SMTexturePitched << < dimGrid, dimBlock >> >((float*)pvFilter, (float*)pvResponse, iHeight, iWidth, iFilterHeight, iFilterWidth);

	CopyDataFromGPU(pfResponsesGPU, pvResponse, sizeof(float) * iHeight * iWidth); // copy results back from GPU

	for (y = 0; y < iHeight - iFilterHeight; y++) // compare with CPU results
	{
		for (x = 0; x < iWidth - iFilterWidth; x++)
		{
			float fDiff = fabs(pfResponsesGPU[y * iWidth + x] - pfResponsesCPU[y * iWidth + x]);
			if (fDiff > 0.0001f)
			{
				printf("\r\nCPU and GPU results do not match");
			}
		}
	}

}