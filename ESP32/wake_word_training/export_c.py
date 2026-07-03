import numpy as np, tensorflow as tf, feat
m=tf.keras.models.load_model('/tmp/ww/model.keras')
nz=np.load('/tmp/ww/norm.npz'); MEAN=float(nz['mean']); STD=float(nz['std'])
W=[l.get_weights() for l in m.layers if l.get_weights()]
# layers with weights in order: conv1,conv2,conv3,dense
(c1w,c1b),(c2w,c2b),(c3w,c3b),(dw,db)=W
for nm,a in [('c1',c1w),('c2',c2w),('c3',c3w),('dense',dw)]:
    print(nm, a.shape)
def _f(v):
    t="%.8g"%v
    if not any(c in t for c in ".eEnN"): t+=".0"
    return t+"f"
def arr(name,a):
    flat=a.astype(np.float32).ravel()
    return "static const float %s[%d]={"%(name,flat.size)+",".join(_f(v) for v in flat)+"};\n"
H=open('/tmp/ww/out/hey_pip_model.h','w')
H.write("// Auto-generated: 'hey pip' wake-word model (log-mel + tiny CNN). Do not edit.\n#pragma once\n")
H.write("#define WW_FRAMES %d\n#define WW_MELS %d\n#define WW_FRAME_LEN %d\n#define WW_FRAME_STEP %d\n#define WW_NFFT %d\n#define WW_BINS %d\n"%(
    feat.N_FRAMES,feat.N_MEL,feat.FRAME_LEN,feat.FRAME_STEP,feat.NFFT,feat.N_BINS))
H.write("#define WW_SR %d\n#define WW_CLIP %d\n"%(feat.SR,feat.CLIP))
H.write("static const float WW_MEAN=%s;\nstatic const float WW_STD=%s;\n"%(_f(MEAN),_f(STD)))
H.write("#define WW_CMN 1\n")   # model trained on per-band mean-normalized log-mel (see feat.py)
# conv dims
H.write("#define WW_C1_OUT %d\n#define WW_C2_OUT %d\n#define WW_C3_OUT %d\n"%(c1w.shape[3],c2w.shape[3],c3w.shape[3]))
H.write("#define WW_C1_IN %d\n#define WW_C2_IN %d\n#define WW_C3_IN %d\n"%(c1w.shape[2],c2w.shape[2],c3w.shape[2]))
H.write("#define WW_K 3\n")
H.write(arr("WW_HANN",feat.HANN))
H.write(arr("WW_MELFB",feat.MEL_FB))   # [40*257] row-major (mel,bin)
H.write(arr("WW_C1W",c1w)); H.write(arr("WW_C1B",c1b))
H.write(arr("WW_C2W",c2w)); H.write(arr("WW_C2B",c2b))
H.write(arr("WW_C3W",c3w)); H.write(arr("WW_C3B",c3b))
H.write(arr("WW_DW",dw));   H.write(arr("WW_DB",db))
H.write("#define WW_GAP %d\n#define WW_NCLASS %d\n"%(dw.shape[0],dw.shape[1]))
H.close()
import os; print("header bytes", os.path.getsize('/tmp/ww/out/hey_pip_model.h'))
