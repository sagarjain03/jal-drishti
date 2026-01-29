from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from datetime import datetime

# Configuration
INPUT_DIR = "../data/enhancement/paired"
OUTPUT_DIR = "yolo/outputs"
MODEL_NAME = "yolo/weights/yolo8v"  # Auto-download model
CONF_THRESHOLD = 0.4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all image files
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
image_files = [f for f in Path(INPUT_DIR).iterdir() if f.suffix.lower() in image_extensions]
total_images = len(image_files)

print(f"🔄 Found {total_images} images to process...")
print(f"Model: {MODEL_NAME}")
print(f"Processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load model
model = YOLO(MODEL_NAME)

start_time = datetime.now()
processed = 0
failed = 0
total_detections = 0

# Process each image
for idx, image_path in enumerate(image_files, 1):
    try:
        # Run inference
        results = model(str(image_path), conf=CONF_THRESHOLD, verbose=False)
        
        # Load image for visualization
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠️  [{idx}/{total_images}] Failed to read: {image_path.name}")
            failed += 1
            continue
        
        # Draw detections
        detections_count = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"Anomaly {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                detections_count += 1
        
        total_detections += detections_count
        
        # Save output
        output_path = os.path.join(OUTPUT_DIR, image_path.name)
        cv2.imwrite(output_path, image)
        
        processed += 1
        if idx % 50 == 0 or idx == total_images:
            elapsed = (datetime.now() - start_time).total_seconds()
            avg_time = elapsed / idx
            eta_remaining = avg_time * (total_images - idx)
            print(f"✅ [{idx}/{total_images}] {image_path.name} ({detections_count} detections) | Avg: {avg_time:.2f}s/img | ETA: {eta_remaining:.1f}s")
    
    except Exception as e:
        print(f"❌ [{idx}/{total_images}] Error processing {image_path.name}: {str(e)}")
        failed += 1

end_time = datetime.now()
total_time = (end_time - start_time).total_seconds()

print(f"\n{'='*60}")
print(f"📊 DETECTION COMPLETE")
print(f"{'='*60}")
print(f"✅ Successfully processed: {processed} images")
print(f"⚠️  Failed: {failed} images")
print(f"🎯 Total detections found: {total_detections}")
print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"⏱️  Average time per image: {total_time/total_images:.2f} seconds")
print(f"📁 Output saved to: {os.path.abspath(OUTPUT_DIR)}")
print(f"⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
