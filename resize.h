#ifndef RESIZE_H
#define RESIZE_H


void InitResizeTexturesP( int iHeight, int iWidth, int iDepth, int* piPitch );
void InitResizeTexturesPF( int iHeight, int iWidth, int iDepth, int* piPitch );
void DeleteResizeTexturesP();
void DeleteResizeTexturesPF();



int ResizeFermi( unsigned char* pchSplitImage, void* pvResizeBuffer, int iHeight, int iWidth, int iPitch, int iScaledHeight, int iScaledWidth, int iScaledPitch, int iDepth );
int ResizeFermiPF( void* pvSplitImage, void* pvScaledImage, int iHeight, int iWidth, int iPitch, int iScaledHeight, int iScaledWidth, int iScaledPitch, int iDepth ); // float version of Device->Device resize

#endif // RESIZE_H