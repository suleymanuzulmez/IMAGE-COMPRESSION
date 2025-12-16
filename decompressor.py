import os
import yaml
from PIL import Image
import numpy as np
import yaml
import matplotlib.pyplot as plt
import cv2
import json
import struct
from dahuffman import HuffmanCodec
import os


########---------- QUANTIZATION MATRICES ----------########
# Matrices are for 8x8 DCT blocks, if block size changes, then matrices need to be rearranged

Q_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float32)

Q_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)

Q_LUMA_AGGR = np.array([
    [ 29,  20,  18,  29,  43,  72,  92, 110],
    [ 22,  22,  25,  34,  47, 104, 108,  99],
    [ 25,  23,  29,  43,  72, 103, 124, 101],
    [ 25,  31,  40,  52,  92, 157, 144, 112],
    [ 32,  40,  67, 101, 122, 196, 185, 139],
    [ 43,  63,  99, 115, 146, 187, 203, 166],
    [ 88, 115, 140, 157, 185, 218, 216, 182],
    [130, 166, 171, 176, 202, 180, 185, 178],
], dtype=np.float32)

Q_CHROMA_AGGR = np.array([
    [ 26,  27,  36,  70, 148, 148, 148, 148],
    [ 27,  32,  39,  99, 148, 148, 148, 148],
    [ 36,  39,  84, 148, 148, 148, 148, 148],
    [ 70,  99, 148, 148, 148, 148, 148, 148],
    [148, 148, 148, 148, 148, 148, 148, 148],
    [148, 148, 148, 148, 148, 148, 148, 148],
    [148, 148, 148, 148, 148, 148, 148, 148],
    [148, 148, 148, 148, 148, 148, 148, 148],
], dtype=np.float32)

Q_V_HSV = np.array([
    [ 11,   8,   7,  16,  24,  40,  66,  79],
    [  8,   8,  14,  19,  26,  75,  78,  72],
    [ 10,  13,  16,  24,  52,  74,  90,  73],
    [ 14,  17,  22,  38,  66, 113, 104,  81],
    [ 18,  22,  48,  73,  88, 142, 134, 100],
    [ 24,  46,  72,  83, 105, 135, 147, 120],
    [ 64,  83, 101, 113, 134, 157, 156, 131],
    [ 94, 120, 124, 127, 146, 130, 134, 129],
], dtype=np.float32)

Q_HS_HSV = np.array([
    [ 17,  18,  24,  61, 129, 129, 158, 158],
    [ 18,  21,  34,  86, 129, 158, 158, 158],
    [ 24,  34,  73, 129, 158, 158, 158, 158],
    [ 61,  86, 129, 158, 158, 158, 158, 158],
    [129, 129, 158, 158, 158, 158, 158, 158],
    [129, 158, 158, 158, 158, 158, 158, 158],
    [158, 158, 158, 158, 158, 158, 158, 158],
    [158, 158, 158, 158, 158, 158, 158, 158],
], dtype=np.float32)

######## -------- BINARY FILE READER ------- ##############

def read_binary_file(path):
        
    with open(path, "rb") as f:
        # First info is header size info kept in first 4 bytes
        header_len_bytes = f.read(4)
        header_len = struct.unpack(">I", header_len_bytes)[0]
        # We read header bytes
        header_bytes = f.read(header_len)
        header_str = header_bytes.decode("utf-8")
        header = json.loads(header_str)
        # Then read compressed data
        compressed_data = f.read()
    
    y_len = header["Y_len"]
    cb_len = header["Cb_len"]
    cr_len = header["Cr_len"]

    # We use dahuffman decoding using the codebook in the header file
    codebook = header["huffman_codebook"]
    codec = HuffmanCodec(code_table=codebook, check=True)
    decoded = codec.decode(compressed_data)

    # Return sequences seperately
    Y_seq = decoded[0:y_len]
    Cb_seq = decoded[y_len:y_len+cb_len]
    Cr_seq = decoded[y_len+cb_len:y_len+cb_len+cr_len]

    return header, Y_seq, Cb_seq, Cr_seq

###### ------ SEQUENCE TO MATRIX FUNCTION ------ ######

def seq_to_matrix(seq, shape):
    H, W = shape
    mat = np.zeros((H, W), dtype=np.float32)

    idx = 0

    # We apply same pattern in the sequencer funstion from 0 to (H-1)+(W-1)-1
    for i in range(0, H - 1 + W - 1):
        if i == 0:
            mat[0, 0] = seq[idx]
            idx += 1

        # When row+col even, move upward
        elif i % 2 == 0:
            x_idx = min(i, H - 1)
            y_idx = i - x_idx
            while (x_idx >= 0) and (y_idx < W):
                mat[x_idx, y_idx] = seq[idx]
                idx += 1
                x_idx -= 1
                y_idx += 1

        # When row+col odd move downward
        else:
            y_idx = min(i, W - 1)
            x_idx = i - y_idx
            while (y_idx >= 0) and (x_idx < H):
                mat[x_idx, y_idx] = seq[idx]
                idx += 1
                y_idx -= 1
                x_idx += 1

    # Final element was add at the end
    mat[H - 1, W - 1] = seq[idx]

    return mat

###### ------ DE-QUANTISATION FUNCTION ------ ######

def dequantize_matrix(mat_q, mode, is_it_luma):
    
    mat = np.zeros_like(mat_q, dtype=np.float32)

    if mode == "soft":
        quantization_matrix = Q_LUMA if is_it_luma else Q_CHROMA
        mat = mat_q * quantization_matrix
    elif mode == "agressive":
        quantization_matrix = Q_LUMA_AGGR if is_it_luma else Q_CHROMA_AGGR
        mat = mat_q * quantization_matrix
    elif mode == "hsv-oriented":
        quantization_matrix = Q_V_HSV if is_it_luma else Q_HS_HSV
        mat = mat_q * quantization_matrix
    else:
        raise ValueError(f"Unsupported quantization mode: {mode}")

    return mat

####### ------ UPSAMPLING FUNTIONS ------ ##########

def upsample_444(C_sub):
    # This format does not apply upsampling
    return C_sub.astype(np.float32)

def upsample_422(C_sub, out_shape, method="bilinear"):
    H, W = out_shape
    C_sub = C_sub.astype(np.float32)

    # Nearest nighbor interpolation
    C = np.repeat(C_sub, 2, axis=1)  # creating replicas of nearest terms to increase horizontal size

    # Applies bilinear interpolation to intermediate replicas
    if method == "bilinear":
        C2 = C.copy()
        C2[:, 1:-1:2] = 0.5 * (C2[:, 0:-2:2] + C2[:, 2::2])
        C = C2

    # Below lines handle for size mismatch
    if C.shape[1] > W:
        C = C[:, :W]
    elif C.shape[1] < W:
        pad = np.repeat(C[:, -1:], W - C.shape[1], axis=1)
        C = np.concatenate([C, pad], axis=1)

    if C.shape[0] > H:
        C = C[:H, :]
    elif C.shape[0] < H:
        pad = np.repeat(C[-1:, :], H - C.shape[0], axis=0)
        C = np.concatenate([C, pad], axis=0)

    return C

def upsample_420(C_sub, out_shape, method="bilinear"):
    H, W = out_shape
    C_sub = C_sub.astype(np.float32)

    # Nearest nighbor interpolation BUT for both dimensions
    C = np.repeat(np.repeat(C_sub, 2, axis=0), 2, axis=1)

    # Applies bilinear interpolation to intermediate replicas of EACH dimension
    if method == "bilinear":
        C2 = C.copy()
        # Horizontal interpolation
        C2[:, 1:-1:2] = 0.5 * (C2[:, 0:-2:2] + C2[:, 2::2])
        # Vertical interpolation
        C2[1:-1:2, :] = 0.5 * (C2[0:-2:2, :] + C2[2::2, :])
        C = C2

    # Below lines handle for size mismatch
    C = C[:H, :W]
    if C.shape[0] < H:
        pad = np.repeat(C[-1:, :], H - C.shape[0], axis=0)
        C = np.concatenate([C, pad], axis=0)
    if C.shape[1] < W:
        pad = np.repeat(C[:, -1:], W - C.shape[1], axis=1)
        C = np.concatenate([C, pad], axis=1)

    return C

def chroma_upsample(Cb_sub, Cr_sub, out_shape, mode="4:2:0", method="bilinear"):
    if mode == "4:4:4":
        Cb = upsample_444(Cb_sub)
        Cr = upsample_444(Cr_sub)
    elif mode == "4:2:2":
        Cb = upsample_422(Cb_sub, out_shape, method)
        Cr = upsample_422(Cr_sub, out_shape, method)
    elif mode == "4:2:0":
        Cb = upsample_420(Cb_sub, out_shape, method)
        Cr = upsample_420(Cr_sub, out_shape, method)
    else:
        raise ValueError(f"Unsupported chroma upsampling mode: {mode}")

    return Cb, Cr

######## -------- MAIN FUNCTION ------- ##############
# It decompresses a file using absolute path, it must be considered while making experimentation

def decompressor(path):
    # Reading config file
    with open(r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\config.yml", "r") as f:
        config = yaml.safe_load(f)

    block_size = config["decompression"]["idct_block_size"]
    quantization_mode = config["decompression"]["quantization_mode"]
    downsampling_mode = config["decompression"]["downsampling_rate"]
    interpolation = config["decompression"]["interpolation_mode"]
    out_path = config["decompression"]["out_location"]
    
    # We first read binary file
    header, Y_seq, Cb_seq, Cr_seq = read_binary_file(path)

    # We get QUANTIZED matrices for Y, Cb, and Cr
    Y_dct_mat = seq_to_matrix(Y_seq, header["Y_shape"])
    Cb_dct_mat = seq_to_matrix(Cb_seq, header["Cb_shape"])
    Cr_dct_mat = seq_to_matrix(Cr_seq, header["Cr_shape"])

    # We then DE-QUANTIZE the matrices and apply IDCT 
    # NOTE: This is applied BLOCK-WISE
    Y_mat  = np.zeros_like(Y_dct_mat, dtype=np.float32)
    Cb_mat = np.zeros_like(Cb_dct_mat, dtype=np.float32)
    Cr_mat = np.zeros_like(Cr_dct_mat, dtype=np.float32)

    # Filling IgAge matrices block-wise
    M, N = Y_mat.shape
    K, L = Cb_mat.shape
    Q, R = Cr_mat.shape
    for i in range(0, M-block_size+1,block_size):
        for j in range(0, N-block_size+1, block_size):
            block = dequantize_matrix(Y_dct_mat[i:i+block_size, j:j+block_size], quantization_mode, True)
            Y_mat[i:i+block_size, j:j+block_size] = cv2.idct(block) +128                                               # This shift comes from -128 shift taht is applied in compression
    
    for i in range(0, K-block_size+1,block_size):
        for j in range(0, L-block_size+1, block_size):
            block = dequantize_matrix(Cb_dct_mat[i:i+block_size, j:j+block_size], quantization_mode, False)
            Cb_mat[i:i+block_size, j:j+block_size] = cv2.idct(block) +128                                               # This shift comes from -128 shift taht is applied in compression
 
    for i in range(0, Q-block_size+1,block_size):
        for j in range(0, R-block_size+1, block_size):
            block = dequantize_matrix(Cr_dct_mat[i:i+block_size, j:j+block_size], quantization_mode, False)
            Cr_mat[i:i+block_size, j:j+block_size] = cv2.idct(block) +128                                               # This shift comes from -128 shift taht is applied in compression

    # UP-SAMPLING/Interpolating the Cb and Cr
    out_shape = [M, N]
    Cb_mat_u, Cr_mat_u = chroma_upsample(Cb_mat, Cr_mat, out_shape, downsampling_mode, interpolation)

    # Converting matrices to uint8
    Y_u8  = np.clip(Y_mat,  0, 255).astype(np.uint8)
    Cb_u8 = np.clip(Cb_mat_u, 0, 255).astype(np.uint8)
    Cr_u8 = np.clip(Cr_mat_u, 0, 255).astype(np.uint8)

    # Merging them and creating YCbCr image back
    Y_img  = Image.fromarray(Y_u8,  mode="L")
    Cb_img = Image.fromarray(Cb_u8, mode="L")
    Cr_img = Image.fromarray(Cr_u8, mode="L")

    img_ycbcr = Image.merge("YCbCr", (Y_img, Cb_img, Cr_img))

    # Converting back to RGB
    img_rgb = img_ycbcr.convert("RGB")

    base_name = os.path.basename(path)
    name_root, _ = os.path.splitext(base_name)
    out_filename = name_root + ".tiff"

    os.makedirs(out_path, exist_ok=True)  # klasör yoksa oluştur
    out_path = os.path.join(out_path, out_filename)
    img_rgb.save(out_path)


if __name__ == "__main__":
    decompressor()