import os
import yaml
from compressor import compressor
from decompressor import decompressor
from PIL import Image
import numpy as np
import shutil
import time
from datetime import datetime

def mse_cal(path1, path2):
    img1 = Image.open(path1)
    img2 = Image.open(path2)

    arr1 = np.asarray(img1, dtype=np.float64)
    arr2 = np.asarray(img2, dtype=np.float64)

    diff = arr1 - arr2
    mse = np.mean(diff ** 2)

    return mse

def main():

    lines = [
        "---------------TESTS------------------",
        f"Time = {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}"
    ]

    text_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\reports\test_results.txt"
    with open(text_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    # We first get all image paths to give later the compressor function
    data_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\data"
    entries = os.listdir(data_path)
    files = sorted([
    os.path.join(data_path, name)
    for name in entries
    if os.path.isfile(os.path.join(data_path, name))
    ])
    
    # Config file will be changed during experimentation
    config_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\config.yml"
    
    #####-------- Chroma subsampling tests ---------#########
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["compression"]["downsampling_rate"] = '4:4:4'
    config["decompression"]["downsampling_rate"] = '4:4:4'

    with open(config_path, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            default_flow_style=False
        )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    index = 1
    for loc in files:
        compressor(loc)
        print(f"{index} of {len(files)} images is compressed.")
        index += 1
    
    comp_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\compressed_outputs"
    entries = os.listdir(comp_path)
    dec_files = sorted([
    os.path.join(comp_path, name)
    for name in entries
    if os.path.isfile(os.path.join(comp_path, name))
    ])

    index = 1
    for loc in dec_files:
        decompressor(loc)
        print(f"{index} of {len(files)} images is decompressed.")
        index += 1

    
    src_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs\17.tiff"
    dst_dir  = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\reports\444"
    os.makedirs(dst_dir, exist_ok=True)
    file_name = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, file_name)
    shutil.copy2(src_path, dst_path)
    
    total_bits = 0
    for name in os.listdir(comp_path):
        if name.lower().endswith(".bin"):
            file_path = os.path.join(comp_path, name)
            if os.path.isfile(file_path):
                size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                size_bits = size_bytes * 8                # bit'e çevir
                total_bits += size_bits

    total_bits_original = 0
    for name in os.listdir(data_path):
        if name.lower().endswith(".tiff"):
            file_path = os.path.join(data_path, name)
            if os.path.isfile(file_path):
                size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                size_bits = size_bytes * 8                # bit'e çevir
                total_bits_original += size_bits   
    
    comp_rate = total_bits_original/total_bits

    print("-----------------------")
    print(f"CHROMA 4:4:4 ---- Average compression rate is: {comp_rate}")

    with open(text_path, "a", encoding="utf-8") as f:
        f.write(f"CHROMA 4:4:4 ---- Average compression rate is: {comp_rate}\n")

    out_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs"
    entries = os.listdir(out_path)
    out_files = sorted([
    os.path.join(out_path, name)
    for name in entries
    if os.path.isfile(os.path.join(out_path, name))
    ])

    mse = 0
    for i in range(len(files)):
        mse += mse_cal(files[i],out_files[i])
    
    mse = mse/len(files)
    print("-----------------------")
    print(f"CHROMA 4:4:4 ---- Average MSE is: {mse}")   

    with open(text_path, "a", encoding="utf-8") as f:
        f.write(f"CHROMA 4:4:4 ---- Average MSE is: {mse}\n")


    # ---- 4:2:2 -----

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["compression"]["downsampling_rate"] = '4:2:2'
    config["decompression"]["downsampling_rate"] = '4:2:2'

    with open(config_path, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            default_flow_style=False
        )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    index = 1
    for loc in files:
        compressor(loc)
        print(f"{index} of {len(files)} images is compressed.")
        index += 1
    
    comp_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\compressed_outputs"
    entries = os.listdir(comp_path)
    dec_files = sorted([
    os.path.join(comp_path, name)
    for name in entries
    if os.path.isfile(os.path.join(comp_path, name))
    ])

    index = 1
    for loc in dec_files:
        decompressor(loc)
        print(f"{index} of {len(files)} images is decompressed.")
        index += 1

    src_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs\17.tiff"
    dst_dir  = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\reports\422"
    os.makedirs(dst_dir, exist_ok=True)
    file_name = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, file_name)
    shutil.copy2(src_path, dst_path)
    
    total_bits = 0
    for name in os.listdir(comp_path):
        if name.lower().endswith(".bin"):
            file_path = os.path.join(comp_path, name)
            if os.path.isfile(file_path):
                size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                size_bits = size_bytes * 8                # bit'e çevir
                total_bits += size_bits

    total_bits_original = 0
    for name in os.listdir(data_path):
        if name.lower().endswith(".tiff"):
            file_path = os.path.join(data_path, name)
            if os.path.isfile(file_path):
                size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                size_bits = size_bytes * 8                # bit'e çevir
                total_bits_original += size_bits   
    
    comp_rate = total_bits_original/total_bits

    print("-----------------------")
    print(f"CHROMA 4:2:2 ---- Average compression rate is: {comp_rate}")    

    with open(text_path, "a", encoding="utf-8") as f:
        f.write(f"CHROMA 4:2:2 ---- Average compression rate is: {comp_rate}\n")

    out_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs"
    entries = os.listdir(out_path)
    out_files = sorted([
    os.path.join(out_path, name)
    for name in entries
    if os.path.isfile(os.path.join(out_path, name))
    ])

    mse = 0
    for i in range(len(files)):
        mse += mse_cal(files[i],out_files[i])
    
    mse = mse/len(files)
    print("-----------------------")
    print(f"CHROMA 4:2:2 ---- Average MSE is: {mse}")  

    with open(text_path, "a", encoding="utf-8") as f:
        f.write(f"CHROMA 4:2:2 ---- Average MSE is: {mse}\n")

    # ---- 4:2:0 -----

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["compression"]["downsampling_rate"] = '4:2:0'
    config["decompression"]["downsampling_rate"] = '4:2:0'

    with open(config_path, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            default_flow_style=False
        )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    index = 1
    for loc in files:
        compressor(loc)
        print(f"{index} of {len(files)} images is compressed.")
        index += 1
    
    comp_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\compressed_outputs"
    entries = os.listdir(comp_path)
    dec_files = sorted([
    os.path.join(comp_path, name)
    for name in entries
    if os.path.isfile(os.path.join(comp_path, name))
    ])

    index = 1
    for loc in dec_files:
        decompressor(loc)
        print(f"{index} of {len(files)} images is decompressed.")
        index += 1
    
    src_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs\17.tiff"
    dst_dir  = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\reports\420"
    os.makedirs(dst_dir, exist_ok=True)
    file_name = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, file_name)
    shutil.copy2(src_path, dst_path)
    
    total_bits = 0
    for name in os.listdir(comp_path):
        if name.lower().endswith(".bin"):
            file_path = os.path.join(comp_path, name)
            if os.path.isfile(file_path):
                size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                size_bits = size_bytes * 8                # bit'e çevir
                total_bits += size_bits

    total_bits_original = 0
    for name in os.listdir(data_path):
        if name.lower().endswith(".tiff"):
            file_path = os.path.join(data_path, name)
            if os.path.isfile(file_path):
                size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                size_bits = size_bytes * 8                # bit'e çevir
                total_bits_original += size_bits   
    
    comp_rate = total_bits_original/total_bits

    print("-----------------------")
    print(f"CHROMA 4:2:0 ---- Average compression rate is: {comp_rate}")

    with open(text_path, "a", encoding="utf-8") as f:
        f.write(f"CHROMA 4:2:0 ---- Average compression rate is: {comp_rate}\n")

    out_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs"
    entries = os.listdir(out_path)
    out_files = sorted([
    os.path.join(out_path, name)
    for name in entries
    if os.path.isfile(os.path.join(out_path, name))
    ])

    mse = 0
    for i in range(len(files)):
        mse += mse_cal(files[i],out_files[i])
    
    mse = mse/len(files)
    print("-----------------------")
    print(f"CHROMA 4:2:0 ---- Average MSE is: {mse}")  
    
    with open(text_path, "a", encoding="utf-8") as f:
        f.write(f"CHROMA 4:2:0 ---- Average MSE is: {mse}\n")
    

    #####-------- QUANTIZATION EXPERIMENTS ---------#########

    # ---- 4:2:2 SOFT -----

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["compression"]["downsampling_rate"] = '4:2:2'
    config["decompression"]["downsampling_rate"] = '4:2:2'
    config["compression"]["quantization_mode"] = "soft"
    config["decompression"]["quantization_mode"] = "soft"

    with open(config_path, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            default_flow_style=False
        )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    index = 1
    for loc in files:
        compressor(loc)
        print(f"{index} of {len(files)} images is compressed.")
        index += 1
    
    comp_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\compressed_outputs"
    entries = os.listdir(comp_path)
    dec_files = sorted([
    os.path.join(comp_path, name)
    for name in entries
    if os.path.isfile(os.path.join(comp_path, name))
    ])

    index = 1
    for loc in dec_files:
        decompressor(loc)
        print(f"{index} of {len(files)} images is decompressed.")
        index += 1
    
    src_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs\7.tiff"
    dst_dir  = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\reports\soft"
    os.makedirs(dst_dir, exist_ok=True)
    file_name = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, file_name)
    shutil.copy2(src_path, dst_path)

    total_bits = 0
    for name in os.listdir(comp_path):
        if name.lower().endswith(".bin"):
            file_path = os.path.join(comp_path, name)
            if os.path.isfile(file_path):
                size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                size_bits = size_bytes * 8                # bit'e çevir
                total_bits += size_bits

    total_bits_original = 0
    for name in os.listdir(data_path):
        if name.lower().endswith(".tiff"):
            file_path = os.path.join(data_path, name)
            if os.path.isfile(file_path):
                size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                size_bits = size_bytes * 8                # bit'e çevir
                total_bits_original += size_bits   
    
    comp_rate = total_bits_original/total_bits

    print("-----------------------")
    print(f"Quantization SOFT ---- Average compression rate is: {comp_rate}")

    with open(text_path, "a", encoding="utf-8") as f:
        f.write(f"Quantization SOFT ---- Average compression rate is: {comp_rate}\n")

    out_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs"
    entries = os.listdir(out_path)
    out_files = sorted([
    os.path.join(out_path, name)
    for name in entries
    if os.path.isfile(os.path.join(out_path, name))
    ])

    mse = 0
    for i in range(len(files)):
        mse += mse_cal(files[i],out_files[i])
    
    mse = mse/len(files)
    print("-----------------------")
    print(f"Quantization SOFT ---- Average MSE is: {mse}")  
    with open(text_path, "a", encoding="utf-8") as f:
        f.write(f"Quantization SOFT ---- Average MSE is: {mse}\n")

    
    # ---- 4:2:2 AGGRESSIVE -----

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["compression"]["quantization_mode"] = "agressive"
    config["decompression"]["quantization_mode"] = "agressive"

    with open(config_path, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            default_flow_style=False
        )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    index = 1
    for loc in files:
        compressor(loc)
        print(f"{index} of {len(files)} images is compressed.")
        index += 1
    
    comp_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\compressed_outputs"
    entries = os.listdir(comp_path)
    dec_files = sorted([
    os.path.join(comp_path, name)
    for name in entries
    if os.path.isfile(os.path.join(comp_path, name))
    ])

    index = 1
    for loc in dec_files:
        decompressor(loc)
        print(f"{index} of {len(files)} images is decompressed.")
        index += 1
    
    src_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs\7.tiff"
    dst_dir  = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\reports\agressive"
    os.makedirs(dst_dir, exist_ok=True)
    file_name = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, file_name)
    shutil.copy2(src_path, dst_path)
    
    total_bits = 0
    for name in os.listdir(comp_path):
        if name.lower().endswith(".bin"):
            file_path = os.path.join(comp_path, name)
            if os.path.isfile(file_path):
                size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                size_bits = size_bytes * 8                # bit'e çevir
                total_bits += size_bits

    total_bits_original = 0
    for name in os.listdir(data_path):
        if name.lower().endswith(".tiff"):
            file_path = os.path.join(data_path, name)
            if os.path.isfile(file_path):
                size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                size_bits = size_bytes * 8                # bit'e çevir
                total_bits_original += size_bits   
    
    comp_rate = total_bits_original/total_bits

    print("-----------------------")
    print(f"Quantization AGRESSIVE ---- Average compression rate is: {comp_rate}")
    with open(text_path, "a", encoding="utf-8") as f:
        f.write(f"Quantization AGRESSIVE ---- Average compression rate is: {comp_rate}\n")

    out_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs"
    entries = os.listdir(out_path)
    out_files = sorted([
    os.path.join(out_path, name)
    for name in entries
    if os.path.isfile(os.path.join(out_path, name))
    ])

    mse = 0
    for i in range(len(files)):
        mse += mse_cal(files[i],out_files[i])
    
    mse = mse/len(files)
    print("-----------------------")
    print(f"Quantization AGRESSIVE ---- Average MSE is: {mse}")
    with open(text_path, "a", encoding="utf-8") as f:
        f.write(f"Quantization AGRESSIVE ---- Average MSE is: {mse}\n")  


    # ---- 4:2:2 HSV oriented -----

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["compression"]["quantization_mode"] = "hsv-oriented"
    config["decompression"]["quantization_mode"] = "hsv-oriented"

    with open(config_path, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            default_flow_style=False
        )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    index = 1
    for loc in files:
        compressor(loc)
        print(f"{index} of {len(files)} images is compressed.")
        index += 1
    
    comp_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\compressed_outputs"
    entries = os.listdir(comp_path)
    dec_files = sorted([
    os.path.join(comp_path, name)
    for name in entries
    if os.path.isfile(os.path.join(comp_path, name))
    ])

    index = 1
    for loc in dec_files:
        decompressor(loc)
        print(f"{index} of {len(files)} images is decompressed.")
        index += 1

    src_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs\7.tiff"
    dst_dir  = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\reports\hsv-oriented"
    os.makedirs(dst_dir, exist_ok=True)
    file_name = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, file_name)
    shutil.copy2(src_path, dst_path)
    
    total_bits = 0
    for name in os.listdir(comp_path):
        if name.lower().endswith(".bin"):
            file_path = os.path.join(comp_path, name)
            if os.path.isfile(file_path):
                size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                size_bits = size_bytes * 8                # bit'e çevir
                total_bits += size_bits

    total_bits_original = 0
    for name in os.listdir(data_path):
        if name.lower().endswith(".tiff"):
            file_path = os.path.join(data_path, name)
            if os.path.isfile(file_path):
                size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                size_bits = size_bytes * 8                # bit'e çevir
                total_bits_original += size_bits   
    
    comp_rate = total_bits_original/total_bits

    print("-----------------------")
    print(f"Quantization HSV-Oriented ---- Average compression rate is: {comp_rate}")
    with open(text_path, "a", encoding="utf-8") as f:
        f.write(f"Quantization HSV-Oriented ---- Average compression rate is: {comp_rate}\n")  

    out_path = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs"
    entries = os.listdir(out_path)
    out_files = sorted([
    os.path.join(out_path, name)
    for name in entries
    if os.path.isfile(os.path.join(out_path, name))
    ])

    mse = 0
    for i in range(len(files)):
        mse += mse_cal(files[i],out_files[i])
    
    mse = mse/len(files)
    print("-----------------------")
    print(f"Quantization HSV-Oriented ---- Average MSE is: {mse}")
    with open(text_path, "a", encoding="utf-8") as f:
        f.write(f"Quantization HSV-Oriented ---- Average MSE is: {mse}\n")    


    ###### ------ Noise part -------- #########

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["compression"]["downsampling_rate"] = '4:2:2'
    config["decompression"]["downsampling_rate"] = '4:2:2'
    config["compression"]["quantization_mode"] = "hsv-oriented"
    config["decompression"]["quantization_mode"] = "hsv-oriented"


    with open(config_path, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            default_flow_style=False
        )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    def add_gaussian_noise(img_u8, sigma):
        x = img_u8.astype(np.float32)
        n = np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)
        y = np.clip(x + n, 0, 255).astype(np.uint8)
        return y

    np.random.seed(0)

    input_dir =  r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\data"
    noisy_dir = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\gaussian_noised_input_data"
    os.makedirs(noisy_dir, exist_ok=True)

    sigmas = [0, 5, 10, 20, 30]

    for sigma in sigmas:
        for p in [r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\compressed_outputs",r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs"]:
            for name in os.listdir(p):
                fp = os.path.join(p, name)
                if os.path.isdir(fp):
                    shutil.rmtree(fp)
                else:
                    os.remove(fp)

        mse = 0
        total_bits = 0
        total_bits_original = 0
        for fn in os.listdir(input_dir):
            if not fn.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                continue

            in_path = os.path.join(input_dir, fn)
            img = np.array(Image.open(in_path).convert("RGB"), dtype=np.uint8)

            noisy = add_gaussian_noise(img, sigma)
            name_root, _ = os.path.splitext(fn)
            noisy_name = f"{name_root}_sigma{sigma}.tiff"
            noisy_path = os.path.join(noisy_dir, noisy_name)
            Image.fromarray(noisy).save(noisy_path)

            compressor(noisy_path)

            com_name = f"{name_root}_sigma{sigma}.bin"
            com_dir = r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\compressed_outputs"
            com_path = os.path.join(com_dir, com_name)
            decompressor(com_path)

            mse += mse_cal(fr"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs\{name_root}_sigma{sigma}.tiff",fr"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\data\{name_root}.tiff")

        total_bits = 0
        for name in os.listdir(r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\compressed_outputs"):
            if name.lower().endswith(".bin"):
                file_path = os.path.join(comp_path, name)
                if os.path.isfile(file_path):
                    size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                    size_bits = size_bytes * 8                # bit'e çevir
                    total_bits += size_bits

        total_bits_original = 0
        for name in os.listdir(r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\data"):
            if name.lower().endswith(".tiff"):
                file_path = os.path.join(r"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\data", name)
                if os.path.isfile(file_path):
                    size_bytes = os.path.getsize(file_path)   # dosya boyutu (byte)
                    size_bits = size_bytes * 8                # bit'e çevir
                    total_bits_original += size_bits   
            
        comp_rate = total_bits_original/total_bits
            
        mse = mse/len(files)
        print(" ************* ")
        print(f" MSE for sigma of {sigma} is : {mse}")
        with open(text_path, "a", encoding="utf-8") as f:
            f.write(f" MSE for sigma of {sigma} is : {mse}\n")   
        
        print("-----------------------")
        print(f"Noise with sigma = {sigma} ---- Average compression rate is: {comp_rate}")
        with open(text_path, "a", encoding="utf-8") as f:
            f.write(f"Noise with sigma = {sigma} ---- Average compression rate is: {comp_rate}\n")  

        src_path = fr"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\decompressed_outputs\20_sigma{sigma}.tiff"
        dst_dir  = fr"C:\0_bilgekagan\DERSLER\7. Donem\EE 473\Project\reports\sigma_{sigma}"
        os.makedirs(dst_dir, exist_ok=True)
        file_name = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, file_name)
        shutil.copy2(src_path, dst_path)
    
    with open(text_path, "a", encoding="utf-8") as f:
        f.write("------------------------------------------------------------------\n")

    return 0

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    elapsed_sec = end - start
    elapsed_min = elapsed_sec / 60.0
    print(f"Elapsed time: {elapsed_min:.2f} minutes ({elapsed_sec:.1f} seconds)\n\n")