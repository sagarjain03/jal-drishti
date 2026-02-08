"""
Training Data Harvester for Jal-Drishti

MILESTONE-2: Unknown Object Handling & Human-in-the-Loop Learning

This module implements the "Strategic Evolution Layer" - harvesting human-verified
samples for future model retraining.

Key Concepts (Dual-Speed Learning):
1. TACTICAL (Fast): SQLite overrides for immediate use (handled by tactical_db.py)
2. STRATEGIC (Slow): This module - saves data for post-mission retraining

Design Principles:
- Collect ONLY human-verified samples
- Organize by label for direct training use
- Include metadata for data quality tracking
- No live training - harvest only

*** FOR ML ENGINEER ***
The harvested data is saved to: mission_data/retrain_queue/{label}/
You can use this folder directly as a training dataset after the mission.
"""

import os
import json
import time
import base64
import logging
import hashlib
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class HarvestedSample:
    """Metadata for a harvested training sample."""
    image_path: str
    label: str
    timestamp: str
    operator_id: str
    mission_id: str
    track_id: int
    original_confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    sensor_context: Dict  # Sonar/IR readings at capture time


# ============================================================
# TRAINING HARVESTER
# ============================================================

class TrainingHarvester:
    """
    Harvests training data from operator-tagged objects.
    
    Workflow:
    1. Operator tags unknown object via UI → add_override() in tactical_db
    2. ML engine calls harvest_sample() with cropped image
    3. Image saved to retrain_queue/{label}/image_{id}.jpg
    4. Metadata saved alongside for quality tracking
    5. Post-mission: ML engineer uses this data for fine-tuning
    
    *** FOR ML ENGINEER ***
    Call harvest_sample() after operator tagging to save training data.
    The cropped image should be extracted from the detection bbox.
    """
    
    def __init__(self, base_path: str = "mission_data/retrain_queue"):
        """
        Initialize the training harvester.
        
        Args:
            base_path: Root folder for harvested samples
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
        self._sample_count = 0
        self._manifest_path = os.path.join(base_path, "manifest.json")
        self._load_manifest()
        
        logger.info(f"[TrainingHarvester] Initialized at {base_path}")
        logger.info(f"[TrainingHarvester] {self._sample_count} samples in queue")
    
    def _load_manifest(self):
        """Load or create the manifest file."""
        if os.path.exists(self._manifest_path):
            try:
                with open(self._manifest_path, 'r') as f:
                    manifest = json.load(f)
                    self._sample_count = manifest.get('total_samples', 0)
            except:
                self._sample_count = 0
        else:
            self._save_manifest()
    
    def _save_manifest(self):
        """Save manifest with current stats."""
        manifest = {
            'total_samples': self._sample_count,
            'last_updated': datetime.now().isoformat(),
            'labels': self._get_label_counts()
        }
        with open(self._manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _get_label_counts(self) -> Dict[str, int]:
        """Count samples per label."""
        counts = {}
        if os.path.exists(self.base_path):
            for label_dir in os.listdir(self.base_path):
                label_path = os.path.join(self.base_path, label_dir)
                if os.path.isdir(label_path):
                    # Count .jpg files
                    count = len([f for f in os.listdir(label_path) if f.endswith('.jpg')])
                    if count > 0:
                        counts[label_dir] = count
        return counts
    
    def harvest_sample(self,
                      image_data: bytes,
                      label: str,
                      track_id: int,
                      bbox: List[int],
                      original_confidence: float = 0.0,
                      operator_id: str = "operator_1",
                      mission_id: str = "unknown",
                      sensor_context: Optional[Dict] = None) -> Optional[str]:
        """
        Save a cropped detection as a training sample.
        
        *** FOR ML ENGINEER: Call this when operator tags an object ***
        
        Args:
            image_data: Cropped image bytes (JPEG) or base64 string
            label: Operator-assigned label
            track_id: YOLO track ID
            bbox: Bounding box [x1, y1, x2, y2]
            original_confidence: YOLO confidence at tagging
            operator_id: Who tagged it
            mission_id: Current mission ID
            sensor_context: Optional sensor readings at capture
            
        Returns:
            Path to saved image, or None on failure
        """
        try:
            # Sanitize label for filesystem
            safe_label = self._sanitize_label(label)
            label_dir = os.path.join(self.base_path, safe_label)
            os.makedirs(label_dir, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_hash = hashlib.md5(f"{track_id}_{time.time()}".encode()).hexdigest()[:8]
            filename = f"sample_{timestamp}_{sample_hash}.jpg"
            image_path = os.path.join(label_dir, filename)
            
            # Decode if base64
            if isinstance(image_data, str):
                # Handle data URI format
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
            
            # Save image
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            # Save metadata alongside
            metadata = HarvestedSample(
                image_path=image_path,
                label=label,
                timestamp=datetime.now().isoformat(),
                operator_id=operator_id,
                mission_id=mission_id,
                track_id=track_id,
                original_confidence=original_confidence,
                bbox=bbox,
                sensor_context=sensor_context or {}
            )
            
            metadata_path = image_path.replace('.jpg', '_meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            # Update counts
            self._sample_count += 1
            self._save_manifest()
            
            logger.info(f"[TrainingHarvester] Saved sample: {filename} ({label})")
            return image_path
            
        except Exception as e:
            logger.error(f"[TrainingHarvester] Error saving sample: {e}")
            return None
    
    def _sanitize_label(self, label: str) -> str:
        """Make label safe for filesystem."""
        # Replace spaces and special chars
        safe = label.lower().strip()
        safe = safe.replace(' ', '_')
        safe = ''.join(c for c in safe if c.isalnum() or c == '_')
        return safe or "unlabeled"
    
    def get_samples_for_label(self, label: str) -> List[HarvestedSample]:
        """Get all samples for a specific label."""
        safe_label = self._sanitize_label(label)
        label_dir = os.path.join(self.base_path, safe_label)
        
        samples = []
        if os.path.exists(label_dir):
            for filename in os.listdir(label_dir):
                if filename.endswith('_meta.json'):
                    meta_path = os.path.join(label_dir, filename)
                    try:
                        with open(meta_path, 'r') as f:
                            data = json.load(f)
                            samples.append(HarvestedSample(**data))
                    except:
                        pass
        
        return samples
    
    def get_stats(self) -> Dict:
        """Get harvester statistics."""
        return {
            'total_samples': self._sample_count,
            'samples_by_label': self._get_label_counts(),
            'base_path': self.base_path
        }
    
    def export_for_training(self, output_path: Optional[str] = None) -> str:
        """
        Export all harvested data as a training-ready package.
        
        *** FOR ML ENGINEER: Call this post-mission ***
        
        Returns:
            Path to export folder/archive
        """
        if output_path is None:
            output_path = os.path.join(self.base_path, 'export')
        
        os.makedirs(output_path, exist_ok=True)
        
        # Create training manifest
        export_manifest = {
            'created': datetime.now().isoformat(),
            'total_samples': self._sample_count,
            'labels': self._get_label_counts(),
            'format': 'yolo_classification',
            'instructions': """
            To use this data for YOLO fine-tuning:
            1. Each subfolder is a class
            2. Images are human-verified samples
            3. Metadata files contain sensor context
            4. Run: yolo train data=retrain_queue classify
            """
        }
        
        manifest_path = os.path.join(output_path, 'training_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(export_manifest, f, indent=2)
        
        logger.info(f"[TrainingHarvester] Exported training data to {output_path}")
        return output_path


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_harvester_instance: Optional[TrainingHarvester] = None


def get_training_harvester() -> TrainingHarvester:
    """Get or create the singleton TrainingHarvester instance."""
    global _harvester_instance
    if _harvester_instance is None:
        _harvester_instance = TrainingHarvester()
    return _harvester_instance
