#ifndef CUSOBEL_H
#define CUSOBEL_H

//---------------------------------------------------------------------------------
//
// Sobel functions with unsigned char* source, float* destination 
// Src stride = Dst stride
//
//---------------------------------------------------------------------------------
void CuSobelDxUF( void* pvSrc, void* pvDst, int iHeight, int iWidth, int iWidthStep, float fScale );
void CuSobelDyUF( void* pvSrc, void* pvDst, int iHeight, int iWidth, int iWidthStep, float fScale );

void CuSobel3DxUF( void* pvSrc, void* pvDst, int iHeight, int iWidth, int iWidthStep, float fScale );
void CuSobel3DyUF( void* pvSrc, void* pvDst, int iHeight, int iWidth, int iWidthStep, float fScale );


//---------------------------------------------------------------------------------
//
// Sobel functions with  float* source, float* destination 
// Src stride = Dst stride
//
//---------------------------------------------------------------------------------
void CuSobelDxFF( void* pvSrc, void* pvDst, int iHeight, int iWidth, int iWidthStep, float fScale );
void CuSobelDyFF( void* pvSrc, void* pvDst, int iHeight, int iWidth, int iWidthStep, float fScale );

#endif // CUSOBEL_H