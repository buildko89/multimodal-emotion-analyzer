"""Versioned, integer-defined grayscale and exact resize for FER parity."""
PROFILE='integer-gray-linear-exact-v1'


def preprocess_face(face,target_size):
    import cv2
    import numpy as np
    if face.dtype!=np.uint8 or face.ndim not in (2,3):raise ValueError('expected_uint8_face')
    if face.ndim==3:
        if face.shape[2]!=3:raise ValueError('expected_bgr_face')
        values=face.astype(np.uint32)
        # Define coefficients and round-half-up explicitly; no platform color kernels.
        gray=((values[:,:,0]*1868+values[:,:,1]*9617+values[:,:,2]*4899+8192)>>14).astype(np.uint8)
    else:gray=face
    resized=cv2.resize(gray,tuple(target_size),interpolation=cv2.INTER_LINEAR_EXACT)
    normalized=resized.astype(np.float32)/np.float32(255)
    normalized=(normalized-np.float32(.5))*np.float32(2)
    return normalized[None,:,:,None]
