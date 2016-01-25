#include <windows.h>
#include <assert.h>
#include "periscope_common.h"
#include "feature_descriptors.h"
//#include "actionrecognizer2.h"

__global__ void RenderFastBicubic3P( unsigned char* pchResizedImage, int width, int height, float scale, float fInvScale, int iScaledHeight, int iScaledWidth, int iScaledPitch );
__global__ void RenderFastBicubic2P( unsigned char* pchResizedImage, int width, int height, float scale, float fInvScale, int iScaledHeight, int iScaledWidth, int iScaledPitch );
__global__ void RenderFastBicubic1P( unsigned char* pchResizedImage, int width, int height, float scale, float fInvScale, int iScaledHeight, int iScaledWidth, int iScaledPitch );

__global__ void RenderFastBicubic1PF( float* pfResizedImage, int width, int height, float scale, float fInvScale, int iScaledHeight, int iScaledWidth, int iScaledPitch );
__global__ void RenderFastBicubic2PF( float* pfResizedImage, int width, int height, float scale, float fInvScale, int iScaledHeight, int iScaledWidth, int iScaledPitch );
__global__ void RenderFastBicubic3PF( float* pfResizedImage, int width, int height, float scale, float fInvScale, int iScaledHeight, int iScaledWidth, int iScaledPitch );


texture <unsigned char, 2, cudaReadModeNormalizedFloat> texImage0;
texture <unsigned char, 2, cudaReadModeNormalizedFloat> texImage1;
texture <unsigned char, 2, cudaReadModeNormalizedFloat> texImage2;
cudaArray* d_imageArray[3];

texture <float, 2, cudaReadModeElementType> texImageF0;
texture <float, 2, cudaReadModeElementType> texImageF1;
texture <float, 2, cudaReadModeElementType> texImageF2;
cudaArray* d_imageArrayF[3];

texture <unsigned char, 2, cudaReadModeNormalizedFloat> texImageP0;
texture <unsigned char, 2, cudaReadModeNormalizedFloat> texImageP1;
texture <unsigned char, 2, cudaReadModeNormalizedFloat> texImageP2;
void* d_imageArrayP[3];

texture <float, 2, cudaReadModeElementType> texImagePF0;
texture <float, 2, cudaReadModeElementType> texImagePF1;
texture <float, 2, cudaReadModeElementType> texImagePF2;
void* d_imageArrayPF[3];



bool g_InitTexture = false;
bool g_InitTextureF = false;
bool g_InitTextureP = false;
bool g_InitTexturePF = false;

int g_Pitch;
int g_PitchF;
int g_PitchP;
int g_PitchPF;

/*
void InitResizeTextures( int iHeight, int iWidth, int iDepth )
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

	cudaMallocArray( &d_imageArray[0], &channelDesc, iWidth, iHeight );
	cudaMallocArray( &d_imageArray[1], &channelDesc, iWidth, iHeight );
	cudaMallocArray( &d_imageArray[2], &channelDesc, iWidth, iHeight );

	texImage0.filterMode = cudaFilterModeLinear;
	texImage1.filterMode = cudaFilterModeLinear;
	texImage2.filterMode = cudaFilterModeLinear;

	cudaBindTextureToArray( texImage0, d_imageArray[0] );
	cudaBindTextureToArray( texImage1, d_imageArray[1] );
	cudaBindTextureToArray( texImage2, d_imageArray[2] );
}


void InitResizeTexturesF( int iHeight, int iWidth, int iDepth )
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaMallocArray( &d_imageArrayF[0], &channelDesc, iWidth, iHeight );
	cudaMallocArray( &d_imageArrayF[1], &channelDesc, iWidth, iHeight );
	cudaMallocArray( &d_imageArrayF[2], &channelDesc, iWidth, iHeight );

	texImageF0.filterMode = cudaFilterModeLinear;
	texImageF1.filterMode = cudaFilterModeLinear;
	texImageF2.filterMode = cudaFilterModeLinear;

	cudaBindTextureToArray( texImageF0, d_imageArrayF[0] );
	cudaBindTextureToArray( texImageF1, d_imageArrayF[1] );
	cudaBindTextureToArray( texImageF2, d_imageArrayF[2] );
}
*/

void InitResizeTexturesP( int iHeight, int iWidth, int iDepth, int* piWidthStride )
{
	size_t pitch;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

	cudaMallocPitch( &d_imageArrayP[0], &pitch, iWidth, iHeight );
	cudaMallocPitch( &d_imageArrayP[1], &pitch, iWidth, iHeight );
	cudaMallocPitch( &d_imageArrayP[2], &pitch, iWidth, iHeight );

	*piWidthStride = pitch;
	g_PitchP = pitch;	

	texImageP0.filterMode = cudaFilterModeLinear;
	texImageP1.filterMode = cudaFilterModeLinear;
	texImageP2.filterMode = cudaFilterModeLinear;

	cudaBindTexture2D( 0, texImageP0, d_imageArrayP[0], channelDesc, iWidth, iHeight, pitch ); 
	cudaBindTexture2D( 0, texImageP1, d_imageArrayP[1], channelDesc, iWidth, iHeight, pitch ); 
	cudaBindTexture2D( 0, texImageP2, d_imageArrayP[2], channelDesc, iWidth, iHeight, pitch ); 

}



void InitResizeTexturesPF( int iHeight, int iWidth, int iDepth, int* piWidthStride )
{
	size_t pitch;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaMallocPitch( &d_imageArrayPF[0], &pitch, sizeof(float) * iWidth, iHeight );
	cudaMallocPitch( &d_imageArrayPF[1], &pitch, sizeof(float) * iWidth, iHeight );
	cudaMallocPitch( &d_imageArrayPF[2], &pitch, sizeof(float) * iWidth, iHeight );

	*piWidthStride = pitch;
	g_PitchPF = pitch;

	texImagePF0.filterMode = cudaFilterModeLinear;
	texImagePF1.filterMode = cudaFilterModeLinear;
	texImagePF2.filterMode = cudaFilterModeLinear;

	cudaBindTexture2D( 0, texImagePF0, d_imageArrayPF[0], channelDesc, iWidth, iHeight, pitch ); 
	cudaBindTexture2D( 0, texImagePF1, d_imageArrayPF[1], channelDesc, iWidth, iHeight, pitch ); 
	cudaBindTexture2D( 0, texImagePF2, d_imageArrayPF[2], channelDesc, iWidth, iHeight, pitch ); 

	
}



void DeleteResizeTexturesP()
{
	cudaUnbindTexture( texImageP0 ); 
	cudaUnbindTexture( texImageP1 ); 
	cudaUnbindTexture( texImageP2 ); 

	cudaFree( d_imageArrayP[0] );
	cudaFree( d_imageArrayP[1] );
	cudaFree( d_imageArrayP[2] );
}



void DeleteResizeTexturesPF()
{
	cudaUnbindTexture( texImagePF0 ); 
	cudaUnbindTexture( texImagePF1 ); 
	cudaUnbindTexture( texImagePF2 ); 

	cudaFree( d_imageArrayPF[0] );
	cudaFree( d_imageArrayPF[1] );
	cudaFree( d_imageArrayPF[2] );
}




//------------------------------------------------------------
// 
// Old school resize, using pre-Kepler textures and 
// CUDA array
//
//------------------------------------------------------------

int ResizeFermi( unsigned char* pchSplitImage, void* pvScaledImage, int iHeight, int iWidth, int iPitch, int iScaledHeight, int iScaledWidth, int iScaledPitch, int iDepth )
{
	int iRet;
	float fScale;

	dim3 dimGrid;
	dim3 dimBlock;

	/*
	if( !g_InitTextureP )
	{
		InitResizeTexturesP( iHeight, iWidth, iDepth, &g_PitchP );
		g_InitTextureP = true;
	}
	*/

	fScale = (float)iScaledHeight/(float)iHeight;

	dimBlock.y = 6;
	dimBlock.x = 32;


	dimGrid.x = (int)ceil((float)iScaledWidth/(float)dimBlock.x);
	dimGrid.y = (int)ceil((float)iScaledHeight/(float)dimBlock.y);


	if( iDepth == 1 )
	{
		cudaMemcpy2D( d_imageArrayP[0], g_PitchP, pchSplitImage, iPitch, iWidth, iHeight, cudaMemcpyHostToDevice );
		RenderFastBicubic1P<<<dimGrid, dimBlock>>>( (unsigned char*)pvScaledImage, iWidth, iHeight, fScale, 1.0f/fScale, iScaledHeight, iScaledWidth, iScaledPitch );
	}
	else
	{
		if( iDepth == 3 )
		{
			cudaMemcpy2D( d_imageArrayP[0], g_PitchP, pchSplitImage, iPitch, iWidth, iHeight, cudaMemcpyHostToDevice );
			cudaMemcpy2D( d_imageArrayP[1], g_PitchP, pchSplitImage + iHeight * iPitch, iWidth, iWidth, iHeight, cudaMemcpyHostToDevice );
			cudaMemcpy2D( d_imageArrayP[2], g_PitchP, pchSplitImage + iHeight * iPitch * 2, iPitch, iWidth, iHeight, cudaMemcpyHostToDevice );
			
			RenderFastBicubic3P<<<dimGrid, dimBlock>>>( (unsigned char*)pvScaledImage, iWidth * iDepth, iHeight, fScale, 1.0f/fScale, iScaledHeight, iScaledWidth, iScaledPitch );
		}
		else
		{
			assert(0); // only support grey scale and RGB
			iRet = -1;
			goto Exit;
		}
	}

	iRet = 0;
Exit:
	return iRet;


}


int ResizeFermiPF( void* pvSplitImage, void* pvScaledImage, int iHeight, int iWidth, int iPitch, int iScaledHeight, int iScaledWidth, int iScaledPitch, int iDepth )
{
	int iRet;
	float fScale;

	dim3 dimGrid;
	dim3 dimBlock;

	/*
	if( !g_InitTexturePF )
	{
		InitResizeTexturesPF( iHeight, iWidth, iDepth, &g_PitchPF );
		g_InitTexturePF = true;
	}
	*/

	fScale = (float)iScaledHeight/(float)iHeight;

	dimBlock.y = 6;
	dimBlock.x = 32;


	dimGrid.x = (int)ceil((float)iScaledWidth/(float)dimBlock.x);
	dimGrid.y = (int)ceil((float)iScaledHeight/(float)dimBlock.y);


	if( iDepth == 1 )
	{
		cudaMemset2D( d_imageArrayPF[0], g_PitchPF, 0x0, g_PitchPF, MAX_FRAME_HEIGHT );

		cudaMemcpy2D( d_imageArrayPF[0], g_PitchPF, pvSplitImage, iPitch, sizeof(float) * iWidth, iHeight, cudaMemcpyDeviceToDevice );
		RenderFastBicubic1PF<<<dimGrid, dimBlock>>>( (float*)pvScaledImage, iWidth * iDepth, iHeight, fScale, 1.0f/fScale, iScaledHeight, iScaledWidth, iScaledPitch/sizeof(float) );
	}
	else
	{
		if( iDepth == 2 )
		{
			cudaMemset2D( d_imageArrayPF[0], g_PitchPF, 0x0, g_PitchPF, MAX_FRAME_HEIGHT );
			cudaMemset2D( d_imageArrayPF[1], g_PitchPF, 0x0, g_PitchPF, MAX_FRAME_HEIGHT );

			cudaMemcpy2D( d_imageArrayPF[0], g_PitchPF, pvSplitImage, iPitch, sizeof(float) * iWidth, iHeight, cudaMemcpyDeviceToDevice );
			cudaMemcpy2D( d_imageArrayPF[1], g_PitchPF, (char*)pvSplitImage + iHeight * iPitch, iPitch, sizeof(float) * iWidth, iHeight, cudaMemcpyDeviceToDevice );

			RenderFastBicubic2PF<<<dimGrid, dimBlock>>>( (float*)pvScaledImage, iWidth * iDepth, iHeight, fScale, 1.0f/fScale, iScaledHeight, iScaledWidth, iScaledPitch/sizeof(float) );
		}
		else
		{
			if( iDepth == 3 )
			{
				cudaMemcpy2D( d_imageArrayPF[0], g_PitchPF, pvSplitImage, sizeof(float) * iWidth, sizeof(float) * iWidth, iHeight, cudaMemcpyDeviceToDevice );
				cudaMemcpy2D( d_imageArrayPF[1], g_PitchPF, (float*)pvSplitImage + iHeight * iWidth, sizeof(float) * iWidth, sizeof(float) * iWidth, iHeight, cudaMemcpyDeviceToDevice );
				cudaMemcpy2D( d_imageArrayPF[2], g_PitchPF, (float*)pvSplitImage + iHeight * iWidth * 2, sizeof(float) * iWidth, sizeof(float) * iWidth, iHeight, cudaMemcpyDeviceToDevice );

				RenderFastBicubic3PF<<<dimGrid, dimBlock>>>( (float*)pvScaledImage, iWidth * iDepth, iHeight, fScale, 1.0f/fScale, iScaledHeight, iScaledWidth, iScaledPitch );
			}
			else
			{
				assert(0);
				iRet = -1;
				goto Exit;
			}
		}
	}

	iRet = 0;
Exit:
	return iRet;


}










typedef unsigned char uchar;

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__
float w0(float a)
{
//    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

__host__ __device__
float w1(float a)
{
//    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__host__ __device__
float w2(float a)
{
//    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__host__ __device__
float w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
}


// g0 and g1 are the two amplitude functions
__device__ float g0(float a)
{
    return w0(a) + w1(a);
}

__device__ float g1(float a)
{
    return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
__device__ float h0(float a)
{
    // note +0.5 offset to compensate for CUDA linear filtering convention
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

__device__ float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}


__device__ float tex2DFastBicubic( const texture<uchar, 2, cudaReadModeNormalizedFloat>texref, float x, float y )
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    // note: we could store these functions in a lookup table texture, but maths is cheap
    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);


	float r = g0(fy) * ( g0x * tex2D(texref, px + h0x, py + h0y) + g1x * tex2D(texref, px + h1x, py + h0y) ) +
			  g1(fy) * ( g0x * tex2D(texref, px + h0x, py + h1y) + g1x * tex2D(texref, px + h1x, py + h1y) );
    
	return r;
}


__device__ float tex2DFastBicubic( const texture<float, 2, cudaReadModeElementType>texref, float x, float y )
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    // note: we could store these functions in a lookup table texture, but maths is cheap
    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);


	float r = g0(fy) * ( g0x * tex2D(texref, px + h0x, py + h0y) + g1x * tex2D(texref, px + h1x, py + h0y) ) +
			  g1(fy) * ( g0x * tex2D(texref, px + h0x, py + h1y) + g1x * tex2D(texref, px + h1x, py + h1y) );
    
	return r;
}


__global__ void RenderFastBicubic3P( unsigned char* pchResizedImage, int width, int height, float scale, float fInvScale, int iScaledHeight, int iScaledWidth, int iScaledPitch )
{
	float c;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i =  y * iScaledPitch * 3 + x * 3;

	float u = x * fInvScale;
    float v = y * fInvScale;


	if( (x  < iScaledWidth ) && (y < iScaledHeight ) ) 
	{
        c = tex2DFastBicubic( texImageP0, u, v );
		pchResizedImage[i++] = c * 0xff;

        c = tex2DFastBicubic( texImageP1, u, v );
		pchResizedImage[i++] = c * 0xff;

        c = tex2DFastBicubic( texImageP2, u, v );
		pchResizedImage[i] = c * 0xff;
    }
}


__global__ void RenderFastBicubic2P( unsigned char* pchResizedImage, int width, int height, float scale, float fInvScale, int iScaledHeight, int iScaledWidth, int iScaledPitch )
{
	float c;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i =  y * iScaledPitch * 2 + x * 2;

	float u = x * fInvScale;
    float v = y * fInvScale;


	if( (x  < iScaledWidth ) && (y < iScaledHeight ) ) 
	{
        c = tex2DFastBicubic( texImageP0, u, v );
		pchResizedImage[i++] = c * 0xff;

        c = tex2DFastBicubic( texImageP1, u, v );
		pchResizedImage[i] = c * 0xff;
    }
}


__global__ void RenderFastBicubic1P( unsigned char* pchResizedImage, int width, int height, float scale, float fInvScale, int iScaledHeight, int iScaledWidth, int iScaledPitch )
{
	float c;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i =  y * iScaledPitch + x;

	float u = x * fInvScale;
    float v = y * fInvScale;


	if( (x  < iScaledWidth ) && (y < iScaledHeight ) ) 
	{
        c = tex2DFastBicubic( texImageP0, u, v );
		pchResizedImage[i] = c * 0xff;
    }
}



__global__ void RenderFastBicubic1PF( float* pfResizedImage, int width, int height, float scale, float fInvScale, int iScaledHeight, int iScaledWidth, int iScaledPitch )
{
	float c;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i =  y * iScaledPitch + x;
	

	float u = x * fInvScale;
    float v = y * fInvScale;


	if( (x  < iScaledWidth ) && (y < iScaledHeight ) ) 
	{
        c = tex2DFastBicubic( texImagePF0, u, v );
		pfResizedImage[i] = c;
    }


}

__global__ void RenderFastBicubic2PF( float* pfResizedImage, int width, int height, float scale, float fInvScale, int iScaledHeight, int iScaledWidth, int iScaledPitch )
{
	float c;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i =  y * iScaledPitch + x;
	
	int iBigStride = iScaledHeight * iScaledPitch;

	float u = x * fInvScale;
    float v = y * fInvScale;


	if( (x  < iScaledWidth ) && (y < iScaledHeight ) ) 
	{
        c = tex2DFastBicubic( texImagePF0, u, v );
		pfResizedImage[i] = c;

        c = tex2DFastBicubic( texImagePF1, u, v );
		pfResizedImage[i + iBigStride] = c;
    }

}

__global__ void RenderFastBicubic3PF( float* pfResizedImage, int width, int height, float scale, float fInvScale, int iScaledHeight, int iScaledWidth, int iScaledPitch )
{
	float c;
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i =  y * iScaledPitch + x;
	
	int iBigStride = iScaledHeight * iScaledPitch;

	float u = x * fInvScale;
    float v = y * fInvScale;


	if( (x  < iScaledWidth ) && (y < iScaledHeight ) ) 
	{
        c = tex2DFastBicubic( texImagePF0, u, v );
		pfResizedImage[i] = c;

        c = tex2DFastBicubic( texImagePF1, u, v );
		pfResizedImage[i + iBigStride] = c;

        c = tex2DFastBicubic( texImagePF2, u, v );
		pfResizedImage[i + 2 * iBigStride] = c;
    }

}