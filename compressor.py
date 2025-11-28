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

########---------- DOWNSAMPLING FUNCTIONS ----------########

def downsample_444(C):
    # This format does not apply downsampling
    return C

def downsample_422(C):
    H, W = C.shape

    # Below lines crops the final column if width is odd but not practically possible, images are power of 2 in general
    if W % 2 == 1:
        C = C[:, :-1]
        H, W = C.shape

    # We basically average width values to decrease size by half
    C_even = C[:, 0::2]
    C_odd  = C[:, 1::2]
    C_sub  = ((C_even + C_odd) / 2.0).astype(C.dtype)

    return C_sub

def downsample_420(C):
    H, W = C.shape

    # Below lines crops the final column/row if width/height is odd but not practically possible, images are power of 2 in general
    if H % 2 == 1:
        C = C[:-1, :]
        H, W = C.shape
    if W % 2 == 1:
        C = C[:, :-1]
        H, W = C.shape

    # Doing the same thing but for both dimensions
    C_even_even = C[0::2, 0::2]
    C_even_odd = C[0::2, 1::2]
    C_odd_even = C[1::2, 0::2]
    C_odd_odd = C[1::2, 1::2]

    C_sub = ((C_even_even + C_even_odd + C_odd_even + C_odd_odd) / 4.0).astype(C.dtype)
    return C_sub

def chroma_downsample(Cb, Cr, mode="4:2:0"):
    if mode == "4:4:4":
        Cb_sub = downsample_444(Cb)
        Cr_sub = downsample_444(Cr)
    elif mode == "4:2:2":
        Cb_sub = downsample_422(Cb)
        Cr_sub = downsample_422(Cr)
    elif mode == "4:2:0":
        Cb_sub = downsample_420(Cb)
        Cr_sub = downsample_420(Cr)
    else:
        raise ValueError(f"Unsupported chroma downsampling mode: {mode}")

    return Cb_sub, Cr_sub

######-------- QUANTIZATION FUNCTION --------######

def quantize_matrix(mat, mode="soft", is_it_luma = False):
    mat_q = np.zeros_like(mat, dtype=np.float32)

    if mode == "soft":
        quantization_matrix = Q_LUMA if is_it_luma else Q_CHROMA
        mat_q = np.round(mat / quantization_matrix)
    elif mode == "agressive":
        pass
    elif mode == "hsv-oriented":
        pass
    else:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    
    return mat_q

######-------- SEQUENCER FUNCTION --------######
#Below function applies zigzag scan on !!! WHOLE IMAGE !!!!

def sequencer(mat):

    H, W = mat.shape
    seq = []

    # We look for ro+coloumn number from 0 to (H-1) + (W-1) -1 to look whetger it is even or odd (-1 comes from last element which is appended at the end)
    for i in range(0, H-1+W-1):
        if i == 0:
            x_idx = 0
            y_idx = 0
            seq.append(mat[x_idx, y_idx])
        # If row+coloumn number is even, zigzag moves upward
        elif i % 2 == 0:
            x_idx = min(i, H-1)
            y_idx = i - x_idx
            while (x_idx >= 0) and (y_idx < W):
                seq.append(mat[x_idx, y_idx])
                x_idx = x_idx - 1
                y_idx = y_idx + 1
        # If row+coloumn number is odd, zigzag moves downward
        else:
            y_idx = min(i, W-1)
            x_idx = i - y_idx
            while (y_idx >= 0) and (x_idx < H):
                seq.append(mat[x_idx, y_idx])
                y_idx = y_idx - 1
                x_idx = x_idx + 1

    seq.append(mat[H-1, W-1])
    return np.array(seq, dtype=np.float32)

####### ------- ENCODING FUNCTION ---------- #######

def encode_image_coeffs(seq_Y, seq_Cb, seq_Cr, sizes, config) -> bytes:
    
    # We first encode sequences using Huffman coding
    symbols = list(seq_Y) + list(seq_Cb) + list(seq_Cr)

    # Huffman encoding using dahuffman library
    codec = HuffmanCodec.from_data(symbols)
    compressed_bytes = codec.encode(symbols)                               # NOTE: symbols_rec = codec.decode(compressed_bytes) converts back to original symbols
    codebook = codec.get_code_table()                                        # NOTE: codebook is get from header, this must be used for decompression    
    codebook_repr = {
        str(sym): [int(n_bits), int(val)]
        for sym, (n_bits, val) in codebook.items()
    }

    # We define header file to reconstuct image during decompression
    Y_H, Y_W = sizes[0]
    Cb_H, Cb_W = sizes[1]
    Cr_H, Cr_W = sizes[2]

    header = {
    # Matrix dimensions fo Y, Cb, and Cr
    "Y_shape":  [Y_H, Y_W],
    "Cb_shape": [Cb_H, Cb_W],
    "Cr_shape": [Cr_H, Cr_W],

    # Compression parameters
    "block_size": config["compression"]["dct_block_size"],
    "downsampling_rate": config["compression"]["downsampling_rate"],
    "quantization_mode": config["compression"]["quantization_mode"],

    # Lengths of sequences
    "Y_len": int(len(seq_Y)),
    "Cb_len": int(len(seq_Cb)),
    "Cr_len": int(len(seq_Cr)),

    # Entropy coding information
    "entropy_coder": "dahuffman",

    # Huffman codebook (symbol → code length etc., will explain shortly)
    "huffman_codebook": codebook_repr,
    }

    # Converting header to bytes
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = len(header_bytes)

    file_bytes = struct.pack(">I", header_len) + header_bytes + compressed_bytes

    return file_bytes 

########---------- MAIN FUNCTION -----------########

def main():
    # Reading config file
    with open(r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\config.yml", "r") as f:
        config = yaml.safe_load(f)

    downsampling_rate = config["compression"]["downsampling_rate"]
    quantization_mode = config["compression"]["quantization_mode"]
    out_path = config["compression"]["out_location"]

    # Loading the image
    path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\data\4.2.06.tiff"
    img = Image.open(path)

    # Converting image format to YCbCr
    img_ycbcr = img.convert("YCbCr")

    # Changing channels into matrices (Y matrix, Cb matrix, Cr matrix)
    Y_pil, Cb_pil, Cr_pil = img_ycbcr.split()
    Y_mat  = np.array(Y_pil).astype(np.float32)
    Cb_mat = np.array(Cb_pil).astype(np.float32)
    Cr_mat = np.array(Cr_pil).astype(np.float32)

    # Downsampling Cb and Cr channels
    Cb_mat , Cr_mat = chroma_downsample(Cb_mat, Cr_mat, mode=downsampling_rate)

    #####----- APPLYING DISCRETE COSINE TRANSFORM (DCT) ON 8x8 BLOCKS -----#####
    block_size = config["compression"]["dct_block_size"] # 8 by default
    # Creating empty DCT matrices
    Y_mat_dct  = np.zeros_like(Y_mat, dtype=np.float32)
    Cb_mat_dct = np.zeros_like(Cb_mat, dtype=np.float32)
    Cr_mat_dct = np.zeros_like(Cr_mat, dtype=np.float32)

    # Filling DCT matrices block by block                                      # NOTE: While dividing into 8x8 blocks, we assume that matrix dimensions are multiples of 8/block_size
    M, N = Y_mat.shape                                                         # This must be kept in mind. (Data in me are multiples of 8, powers of 2 actually, 512x512 or 1024x1024)
    K, L = Cb_mat.shape
    Q, R = Cr_mat.shape
    for i in range(0, M-block_size+1,block_size):
        for j in range(0, N-block_size+1, block_size):
            block = Y_mat[i:i+block_size, j:j+block_size] - 128.0              # NOTE: We shift image values by 128 to decrase DC component 
            Y_mat_dct[i:i+block_size, j:j+block_size] = cv2.dct(block)         # This shift must be reverted during decompression

            # Quantization is applied block by block with quantization matrices
            Y_mat_dct[i:i+block_size, j:j+block_size] = quantize_matrix(Y_mat_dct[i:i+block_size, j:j+block_size], quantization_mode, True)

    for i in range(0, K-block_size+1,block_size):
        for j in range(0, L-block_size+1, block_size):
            block = Cb_mat[i:i+block_size, j:j+block_size] - 128.0              # NOTE: We shift image values by 128 to decrase DC component 
            Cb_mat_dct[i:i+block_size, j:j+block_size] = cv2.dct(block)
            # Quantization is applied block by block with quantization matrices
            Cb_mat_dct[i:i+block_size, j:j+block_size] = quantize_matrix(Cb_mat_dct[i:i+block_size, j:j+block_size], quantization_mode, False)

    for i in range(0, Q-block_size+1,block_size):
        for j in range(0, R-block_size+1, block_size):
            block = Cr_mat[i:i+block_size, j:j+block_size] - 128.0              # NOTE: We shift image values by 128 to decrase DC component 
            Cr_mat_dct[i:i+block_size, j:j+block_size] = cv2.dct(block)
            # Quantization is applied block by block with quantization matrices
            Cr_mat_dct[i:i+block_size, j:j+block_size] = quantize_matrix(Cr_mat_dct[i:i+block_size, j:j+block_size], quantization_mode, False)

    # The last part is to change matrices into sequences and convert them into a binary file
    seq_Y  = sequencer(Y_mat_dct)
    seq_Cb = sequencer(Cb_mat_dct)
    seq_Cr = sequencer(Cr_mat_dct)

    binary_data = encode_image_coeffs(seq_Y, seq_Cb, seq_Cr, [[M,N], [K,L], [Q,R]], config)

    # Writing to output binary file
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "compressed_image.bin"), "wb") as f:
        f.write(binary_data)


if __name__ == "__main__":
    main()