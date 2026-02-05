import os
import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import sys
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# Path jaha tumhari kharab images rakhi hain (Validation set)
INPUT_DIR = "../data/raw_downloads/underwater-trash/train/images" 
# Path jaha enhanced images save hongi (temp folder)
OUTPUT_DIR = "../data/evaluation_results"
# Tumhara GAN Model Weights
GAN_WEIGHTS = "../ml-engine/weights/funie_generator.pth"

# Limit: Kitni images par test karna hai (Zyada mat rakho, 50 kaafi hain)
NUM_IMAGES_TO_TEST = 50

# --- SETUP PATHS ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# Import your Model (Path adjust kar lena agar alag ho)
try:
    from ml_engine.core.models import GeneratorFunieGAN
except ImportError:
    # Fallback for colab structure
    sys.path.append(str(PROJECT_ROOT / "ml-engine"))
    from img_enhancement.funie_gan.nets.funiegan import GeneratorFunieGAN

# --- UIQM CALCULATION FUNCTION ---
def calculate_uiqm(img):
    """
    Simplified UIQM approximation focused on Contrast and Sharpness 
    (Complete UIQM is complex, this is a standard fast variant for evaluation)
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img)
    
    # 1. UICM (Colorfulness)
    rg = a.astype(float) - b.astype(float)
    uicm = -0.0268 * np.sqrt(np.mean(rg**2) + np.mean(l**2)) + 0.1586
    
    # 2. UISM (Sharpness)
    sobelx = cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(l, cv2.CV_64F, 0, 1, ksize=3)
    uism = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    # 3. UIConM (Contrast)
    uiconm = np.std(l)
    
    # Weights for Underwater (Paper standard)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uiqm = c1 * uicm + c2 * uism + c3 * uiconm
    return uiqm

# --- MAIN EVALUATION ---
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Running Evaluation on {device}...")

    # 1. Load GAN
    model = GeneratorFunieGAN().to(device)
    model.load_state_dict(torch.load(GAN_WEIGHTS, map_location=device))
    model.eval()

    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Input directory not found: {INPUT_DIR}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    images = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.png'))]
    images = images[:NUM_IMAGES_TO_TEST]
    
    total_psnr = 0
    total_ssim = 0
    total_uiqm_raw = 0
    total_uiqm_enhanced = 0
    
    print(f"🚀 Testing on {len(images)} images...")

    for img_name in tqdm(images):
        img_path = os.path.join(INPUT_DIR, img_name)
        
        # Read & Preprocess
        raw_img = cv2.imread(img_path)
        if raw_img is None: continue
        
        # Keep a copy of raw for metrics
        raw_resized = cv2.resize(raw_img, (256, 256))
        
        # Prepare for GAN
        img_rgb = cv2.cvtColor(raw_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)
        img_tensor = (img_tensor - 127.5) / 127.5
        
        # Inference
        with torch.no_grad():
            enhanced_tensor = model(img_tensor)
            
        # Post-process
        enhanced_tensor = (enhanced_tensor + 1.0) / 2.0
        enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_uint8 = (enhanced_np * 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)
        
        # --- METRICS CALCULATION ---
        
        # 1. PSNR & SSIM (Comparison with Raw - Note: In real research, you compare with Ground Truth. 
        # Since we don't have GT, comparing with Raw tells us "How much did it change?")
        # NOTE: High PSNR vs Raw means "It didn't change much". Low PSNR means "It changed a lot".
        # For this script, we will calculate UIQM mostly as it's No-Reference.
        
        p_val = psnr(raw_resized, enhanced_bgr)
        s_val = ssim(cv2.cvtColor(raw_resized, cv2.COLOR_BGR2GRAY), 
                     cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY))
        
        # 2. UIQM (The Real Test)
        uiqm_r = calculate_uiqm(raw_resized)
        uiqm_e = calculate_uiqm(enhanced_bgr)
        
        total_psnr += p_val
        total_ssim += s_val
        total_uiqm_raw += uiqm_r
        total_uiqm_enhanced += uiqm_e
        
        # Save comparison for visual check
        comparison = np.hstack((raw_resized, enhanced_bgr))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"eval_{img_name}"), comparison)

    # --- FINAL REPORT ---
    avg_psnr = total_psnr / len(images)
    avg_ssim = total_ssim / len(images)
    avg_uiqm_raw = total_uiqm_raw / len(images)
    avg_uiqm_enhanced = total_uiqm_enhanced / len(images)
    
    print("\n" + "="*40)
    print(" 📢  JAL-DRISHTI PIPELINE REPORT CARD")
    print("="*40)
    print(f"✅ Images Tested: {len(images)}")
    print("-" * 30)
    print(f"🔹 Avg PSNR (vs Raw): {avg_psnr:.2f} dB  (Shows structural fidelity)")
    print(f"🔹 Avg SSIM (vs Raw): {avg_ssim:.4f}     (1.0 is identical to raw)")
    print("-" * 30)
    print("🌊 UNDERWATER QUALITY (UIQM) - The Main Event")
    print(f"🔴 Raw Image Score:    {avg_uiqm_raw:.2f}")
    print(f"🟢 Enhanced Image Score: {avg_uiqm_enhanced:.2f}")
    
    improvement = ((avg_uiqm_enhanced - avg_uiqm_raw) / avg_uiqm_raw) * 100
    print(f"\n🚀 VISIBILITY IMPROVEMENT: +{improvement:.1f}%")
    print("="*40)
    print(f"📂 Visual results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    evaluate()