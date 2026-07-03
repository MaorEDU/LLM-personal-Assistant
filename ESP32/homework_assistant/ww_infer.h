// Pure-C 'hey pip' inference: log-mel front-end + tiny CNN. Matches the Python
// training pipeline exactly (verified by host parity test). No external deps.
#pragma once
#include <math.h>
#include "hey_pip_model.h"
#ifndef WW_PI
#define WW_PI 3.14159265358979323846f
#endif

// ---- in-place radix-2 complex FFT, N=WW_NFFT (512) ----
static void ww_fft(float* re, float* im){
  const int N=WW_NFFT;
  for(int i=1,j=0;i<N;i++){
    int bit=N>>1;
    for(;j&bit;bit>>=1) j^=bit;
    j^=bit;
    if(i<j){ float t=re[i];re[i]=re[j];re[j]=t; t=im[i];im[i]=im[j];im[j]=t; }
  }
  for(int len=2;len<=N;len<<=1){
    float ang=-2.0f*WW_PI/(float)len;
    float wr=cosf(ang), wi=sinf(ang);
    for(int i=0;i<N;i+=len){
      float cwr=1.0f,cwi=0.0f;
      for(int k=0;k<len/2;k++){
        int a=i+k, b=a+len/2;
        float vr=re[b]*cwr-im[b]*cwi, vi=re[b]*cwi+im[b]*cwr;
        float ur=re[a], ui=im[a];
        re[a]=ur+vr; im[a]=ui+vi; re[b]=ur-vr; im[b]=ui-vi;
        float n=cwr*wr-cwi*wi; cwi=cwr*wi+cwi*wr; cwr=n;
      }
    }
  }
}

// log-mel of WW_CLIP samples -> feat[WW_FRAMES*WW_MELS]
static void ww_logmel(const float* sig, float* feat){
  static float re[WW_NFFT], im[WW_NFFT];
  for(int f=0; f<WW_FRAMES; f++){
    int s=f*WW_FRAME_STEP;
    for(int n=0;n<WW_FRAME_LEN;n++){ re[n]=sig[s+n]*WW_HANN[n]; im[n]=0.0f; }
    for(int n=WW_FRAME_LEN;n<WW_NFFT;n++){ re[n]=0.0f; im[n]=0.0f; }
    ww_fft(re,im);
    for(int m=0;m<WW_MELS;m++){
      const float* fb=&WW_MELFB[m*WW_BINS];
      float acc=0.0f;
      for(int b=0;b<WW_BINS;b++){ float p=re[b]*re[b]+im[b]*im[b]; acc+=fb[b]*p; }
      feat[f*WW_MELS+m]=logf(acc+1e-6f);
    }
  }
}

// conv2d 'valid' stride2 + ReLU. in[IH*IW*IC] kernel Keras layout (kh,kw,ic,oc).
static void ww_conv(const float* in,int IH,int IW,int IC,
                    const float* W,const float* B,float* out,int OC){
  int OH=(IH-WW_K)/2+1, OW=(IW-WW_K)/2+1;
  for(int oh=0;oh<OH;oh++) for(int ow=0;ow<OW;ow++) for(int oc=0;oc<OC;oc++){
    float acc=B[oc];
    for(int kh=0;kh<WW_K;kh++) for(int kw=0;kw<WW_K;kw++){
      int ih=oh*2+kh, iw=ow*2+kw;
      const float* ip=&in[(ih*IW+iw)*IC];
      const float* wp=&W[((kh*WW_K+kw)*IC)*OC+oc];
      for(int ic=0;ic<IC;ic++) acc+=ip[ic]*wp[ic*OC];
    }
    out[(oh*OW+ow)*OC+oc]= acc>0.0f?acc:0.0f;
  }
}

// Full forward: raw sig (WW_CLIP float) -> prob[WW_NCLASS]. Needs scratch bufs.
static void ww_infer(const float* sig,float* feat,float* b1,float* b2,float* b3,float* prob){
  ww_logmel(sig,feat);
#ifdef WW_CMN
  // Per-band cepstral mean normalization — matches feat.py logmel(). Input gain
  // adds a constant to every log-mel cell and a stationary spectral tilt adds a
  // per-band constant; subtracting each band's mean over the frames removes both,
  // making the score independent of absolute mic level. Defined by the model
  // header when (and only when) the model was trained on CMN features.
  for(int m=0;m<WW_MELS;m++){
    float mu=0.0f;
    for(int f=0;f<WW_FRAMES;f++) mu+=feat[f*WW_MELS+m];
    mu/=(float)WW_FRAMES;
    for(int f=0;f<WW_FRAMES;f++) feat[f*WW_MELS+m]-=mu;
  }
#endif
  for(int i=0;i<WW_FRAMES*WW_MELS;i++) feat[i]=(feat[i]-WW_MEAN)/WW_STD;
  ww_conv(feat,WW_FRAMES,WW_MELS,1,        WW_C1W,WW_C1B,b1,WW_C1_OUT);
  int H1=(WW_FRAMES-WW_K)/2+1, W1=(WW_MELS-WW_K)/2+1;
  ww_conv(b1,H1,W1,WW_C1_OUT,              WW_C2W,WW_C2B,b2,WW_C2_OUT);
  int H2=(H1-WW_K)/2+1, W2=(W1-WW_K)/2+1;
  ww_conv(b2,H2,W2,WW_C2_OUT,              WW_C3W,WW_C3B,b3,WW_C3_OUT);
  int H3=(H2-WW_K)/2+1, W3=(W2-WW_K)/2+1;
  float gap[WW_C3_OUT];
  for(int c=0;c<WW_C3_OUT;c++){ float a=0; for(int i=0;i<H3*W3;i++) a+=b3[i*WW_C3_OUT+c]; gap[c]=a/(H3*W3); }
  float logit[WW_NCLASS], mx=-1e30f;
  for(int c=0;c<WW_NCLASS;c++){ float a=WW_DB[c]; for(int k=0;k<WW_GAP;k++) a+=gap[k]*WW_DW[k*WW_NCLASS+c]; logit[c]=a; if(a>mx)mx=a; }
  float sum=0; for(int c=0;c<WW_NCLASS;c++){ logit[c]=expf(logit[c]-mx); sum+=logit[c]; }
  for(int c=0;c<WW_NCLASS;c++) prob[c]=logit[c]/sum;
}
