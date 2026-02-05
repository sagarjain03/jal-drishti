import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import sys
from pathlib import Path

# --- SETUP PATHS ---
# Ensure we can import run_funie from ml-engine/img_enhancement
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
IMG_ENHANCEMENT_DIR = CURRENT_DIR / "img_enhancement"
sys.path.append(str(IMG_ENHANCEMENT_DIR))

# --- IMPORT YOUR GENERATION SCRIPT ---
# This allows us to run the generation logic and read its config variables
try:
    import run_funie
except ImportError:
    print("❌ Error: Could not import 'run_funie.py'. Make sure it is in the same folder or path correct.")
    sys.exit(1)

# --- UIQM METRIC FUNCTION ---
def calculate_uiqm(img):
    """Calculates Underwater Image Quality Measure (UIQM)"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img)
    rg = a.astype(float) - b.astype(float)
    uicm = -0.0268 * np.sqrt(np.mean(rg**2) + np.mean(l**2)) + 0.1586
    sobelx = cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(l, cv2.CV_64F, 0, 1, ksize=3)
    uism = np.mean(np.sqrt(sobelx**2 + sobely**2))
    uiconm = np.std(l)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uiqm = c1 * uicm + c2 * uism + c3 * uiconm
    return uiqm

def evaluate_metrics(raw_dir, enhanced_dir):
    print(f"\n📊 Starting Metric Evaluation...")
    print(f"   🔹 Comparing Raw:      {os.path.abspath(raw_dir)}")
    print(f"   🔹 With Enhanced:      {os.path.abspath(enhanced_dir)}")

    if not os.path.exists(raw_dir):
        print(f"❌ Error: Raw directory not found: {raw_dir}")
        return
    if not os.path.exists(enhanced_dir):
        print(f"❌ Error: Enhanced directory not found: {enhanced_dir}")
        return

    raw_files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    enhanced_files = sorted([f for f in os.listdir(enhanced_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    # Find common files
    common_files = [f for f in raw_files if f in enhanced_files]
    
    if len(common_files) == 0:
        print("❌ Error: No matching filenames found. Did generation fail?")
        return

    print(f"🚀 Grading {len(common_files)} images...")

    total_psnr = 0
    total_ssim = 0
    total_uiqm_raw = 0
    total_uiqm_enhanced = 0

    for filename in tqdm(common_files):
        # Load Images
        path_raw = os.path.join(raw_dir, filename)
        path_enh = os.path.join(enhanced_dir, filename)

        img_raw = cv2.imread(path_raw)
        img_enh = cv2.imread(path_enh)

        if img_raw is None or img_enh is None: continue

        # Resize Raw to match Enhanced (640x640) for pixel-perfect comparison
        # run_funie outputs 640x640
        h, w = img_enh.shape[:2]
        img_raw = cv2.resize(img_raw, (w, h))

        # 1. PSNR
        total_psnr += psnr(img_raw, img_enh)
        
        # 2. SSIM (Gray)
        gray_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        gray_enh = cv2.cvtColor(img_enh, cv2.COLOR_BGR2GRAY)
        total_ssim += ssim(gray_raw, gray_enh)

        # 3. UIQM
        total_uiqm_raw += calculate_uiqm(img_raw)
        total_uiqm_enhanced += calculate_uiqm(img_enh)

    # --- FINAL REPORT ---
    count = len(common_files)
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_uiqm_raw = total_uiqm_raw / count
    avg_uiqm_enh = total_uiqm_enhanced / count
    improvement = ((avg_uiqm_enh - avg_uiqm_raw) / avg_uiqm_raw) * 100

    print("\n" + "="*50)
    print(" 📢  JAL-DRISHTI FINAL REPORT CARD")
    print("="*50)
    print(f"✅ Images Graded: {count}")
    print("-" * 35)
    print(f"🔹 Structural Integrity (SSIM): {avg_ssim:.4f}  (Target: >0.85)")
    print(f"🔹 Noise Control (PSNR):       {avg_psnr:.2f} dB")
    print("-" * 35)
    print("🌊 VISIBILITY SCORES (UIQM)")
    print(f"🔴 Raw Score:    {avg_uiqm_raw:.2f}")
    print(f"🟢 Hybrid Score: {avg_uiqm_enh:.2f}")
    print(f"\n🚀 TOTAL IMPROVEMENT: +{improvement:.2f}%")
    print("="*50)

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("⚙️  PHASE 1: RUNNING HYBRID PIPELINE (GAN + CLAHE)")
    print("#"*60)
    
    # 1. CALL THE GENERATION SCRIPT
    # This will use the arguments passed to this script, because run_funie uses argparse!
    # If we want to force specific defaults if no args are passed, we rely on run_funie's defaults.
    # Note: run_funie.main() will verify weights existence etc.
    try:
        # Hack: Inject arguments for run_funie so it doesn't process 5000 images
        # Check if user already provided arguments. If not, inject defaults for evaluation.
        if len(sys.argv) == 1:
            # No args provided, let's inject our test config
            sys.argv.extend(["--limit", "50"])
            print("⚡ Auto-injecting '--limit 50' for evaluation speed.")
        
        run_funie.main()
    except SystemExit as e:
        if e.code != 0:
            print(f"❌ Phase 1 failed with exit code {e.code}")
            sys.exit(e.code)
        # If exit code is 0 (e.g. help), just exit
        sys.exit(0)
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        sys.exit(1)
    
    print("\n" + "#"*60)
    print("⚙️  PHASE 2: EVALUATING RESULTS")
    print("#"*60)
    
    # 2. RUN EVALUATION
    # We read the folder paths directly from run_funie so you don't have to edit them twice
    evaluate_metrics(run_funie.INPUT_DIR, run_funie.OUTPUT_DIR)