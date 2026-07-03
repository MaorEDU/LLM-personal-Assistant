import numpy as np
SR=16000; CLIP=16000
FRAME_LEN=400; FRAME_STEP=160; NFFT=512
N_MEL=40; FMIN=20.0; FMAX=8000.0
N_FRAMES=1+(CLIP-FRAME_LEN)//FRAME_STEP   # 98
N_BINS=NFFT//2+1                          # 257

def hz_to_mel(f): return 2595.0*np.log10(1.0+f/700.0)
def mel_to_hz(m): return 700.0*(10.0**(m/2595.0)-1.0)

def build_mel_fb():
    mmin,mmax=hz_to_mel(FMIN),hz_to_mel(FMAX)
    mpts=np.linspace(mmin,mmax,N_MEL+2); hz=mel_to_hz(mpts)
    bins=np.floor((NFFT+1)*hz/SR).astype(int)
    fb=np.zeros((N_MEL,N_BINS),dtype=np.float32)
    for m in range(1,N_MEL+1):
        l,c,r=bins[m-1],bins[m],bins[m+1]
        c=max(c,l+1); r=max(r,c+1)
        for k in range(l,c):
            if 0<=k<N_BINS: fb[m-1,k]=(k-l)/(c-l)
        for k in range(c,r):
            if 0<=k<N_BINS: fb[m-1,k]=(r-k)/(r-c)
    return fb
MEL_FB=build_mel_fb()
HANN=(0.5-0.5*np.cos(2*np.pi*np.arange(FRAME_LEN)/(FRAME_LEN-1))).astype(np.float32)

def logmel(sig):
    x=np.zeros(CLIP,dtype=np.float32); n=min(len(sig),CLIP); x[:n]=sig[:n]
    fr=np.lib.stride_tricks.sliding_window_view(x,FRAME_LEN)[::FRAME_STEP]
    fr=fr*HANN
    spec=np.abs(np.fft.rfft(fr,NFFT,axis=1))**2
    mel=spec@MEL_FB.T
    lm=np.log(mel+1e-6).astype(np.float32)
    # Per-band cepstral mean normalization (CMN): input gain g adds a constant
    # 2*log(g) to every log-mel cell, and a stationary channel tilt (partly covered
    # mic, codec gain changes) adds a per-band constant — subtracting each band's
    # mean over the 98 frames removes both. This makes the model level-invariant,
    # so the device no longer has to hit an exact absolute input level.
    # MUST match ww_infer.h (WW_CMN) exactly — parity.py gates this.
    lm-=lm.mean(axis=0,keepdims=True)
    return lm
