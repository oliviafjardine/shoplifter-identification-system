#!/usr/bin/env python3
"""
Dataset Manager for Shoplifting Detection Training
Downloads and manages the UCF Anomaly Detection dataset, specifically shoplifting videos
"""

import os
import cv2
import numpy as np
import kagglehub
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import shutil


class ShopliftingDatasetManager:
    """
    Manages the shoplifting video dataset for training and evaluation
    """

    def __init__(self, base_dir: str = "training_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        self.dataset_path = None
        self.shoplifting_videos_path = None
        self.processed_data_path = self.base_dir / "processed"
        self.processed_data_path.mkdir(exist_ok=True)

        self.video_metadata = {}

    def download_dataset(self) -> str:
        """Download the UCF Anomaly Detection dataset"""
        print("📥 Downloading UCF Anomaly Detection dataset...")

        try:
            # Download latest version
            path = kagglehub.dataset_download(
                "minhajuddinmeraj/anomalydetectiondatasetucf")
            print(f"✅ Dataset downloaded to: {path}")

            self.dataset_path = Path(path)

            # Find the shoplifting videos folder
            shoplifting_folder = self.dataset_path / \
                "Anomaly-Videos-Part-4" / "Anomaly-Videos-Part-4Shoplifting"

            if shoplifting_folder.exists():
                self.shoplifting_videos_path = shoplifting_folder
                print(f"✅ Found shoplifting videos at: {shoplifting_folder}")

                # List available videos
                video_files = list(shoplifting_folder.glob("*.mp4"))
                print(f"📹 Found {len(video_files)} shoplifting videos")

                return str(shoplifting_folder)
            else:
                print("❌ Shoplifting videos folder not found!")
                print("Available folders:")
                for folder in self.dataset_path.rglob("*"):
                    if folder.is_dir():
                        print(f"  - {folder}")
                return None

        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return None

    def analyze_videos(self) -> Dict:
        """Analyze the shoplifting videos to understand their characteristics"""
        if not self.shoplifting_videos_path:
            print("❌ No shoplifting videos path set. Run download_dataset() first.")
            return {}

        print("🔍 Analyzing shoplifting videos...")

        video_files = list(self.shoplifting_videos_path.glob("*.mp4"))
        analysis_results = {
            'total_videos': len(video_files),
            'videos': {},
            'summary': {
                'total_duration': 0,
                'total_frames': 0,
                'avg_fps': 0,
                'resolutions': {},
                'file_sizes': []
            }
        }

        for video_file in video_files:
            print(f"  📹 Analyzing {video_file.name}...")

            try:
                cap = cv2.VideoCapture(str(video_file))

                if not cap.isOpened():
                    print(f"    ❌ Could not open {video_file.name}")
                    continue

                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                file_size = video_file.stat().st_size / (1024 * 1024)  # MB

                video_info = {
                    'filename': video_file.name,
                    'duration_seconds': duration,
                    'fps': fps,
                    'frame_count': frame_count,
                    'resolution': f"{width}x{height}",
                    'width': width,
                    'height': height,
                    'file_size_mb': file_size
                }

                analysis_results['videos'][video_file.name] = video_info

                # Update summary
                analysis_results['summary']['total_duration'] += duration
                analysis_results['summary']['total_frames'] += frame_count
                analysis_results['summary']['file_sizes'].append(file_size)

                resolution_key = f"{width}x{height}"
                if resolution_key in analysis_results['summary']['resolutions']:
                    analysis_results['summary']['resolutions'][resolution_key] += 1
                else:
                    analysis_results['summary']['resolutions'][resolution_key] = 1

                cap.release()

                print(
                    f"    ✅ {video_file.name}: {duration:.1f}s, {fps:.1f}fps, {width}x{height}")

            except Exception as e:
                print(f"    ❌ Error analyzing {video_file.name}: {e}")

        # Calculate averages
        if analysis_results['total_videos'] > 0:
            total_fps = sum(v['fps']
                            for v in analysis_results['videos'].values())
            analysis_results['summary']['avg_fps'] = total_fps / \
                analysis_results['total_videos']

        # Save analysis results
        analysis_file = self.processed_data_path / "video_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)

        print(f"📊 Analysis complete! Results saved to {analysis_file}")
        print(f"📈 Summary:")
        print(f"   Total videos: {analysis_results['total_videos']}")
        print(
            f"   Total duration: {analysis_results['summary']['total_duration']:.1f} seconds")
        print(
            f"   Total frames: {analysis_results['summary']['total_frames']}")
        print(f"   Average FPS: {analysis_results['summary']['avg_fps']:.1f}")
        print(f"   Resolutions: {analysis_results['summary']['resolutions']}")

        self.video_metadata = analysis_results
        return analysis_results

    def extract_frames(self, max_videos: Optional[int] = None, frame_interval: int = 30) -> str:
        """
        Extract frames from shoplifting videos for training

        Args:
            max_videos: Maximum number of videos to process (None for all)
            frame_interval: Extract every Nth frame
        """
        if not self.shoplifting_videos_path:
            print("❌ No shoplifting videos path set. Run download_dataset() first.")
            return None

        print(f"🎬 Extracting frames from shoplifting videos...")
        print(f"   Frame interval: every {frame_interval} frames")

        frames_dir = self.processed_data_path / "frames"
        frames_dir.mkdir(exist_ok=True)

        video_files = list(self.shoplifting_videos_path.glob("*.mp4"))
        if max_videos:
            video_files = video_files[:max_videos]

        extraction_log = {
            'timestamp': datetime.now().isoformat(),
            'frame_interval': frame_interval,
            'videos_processed': 0,
            'total_frames_extracted': 0,
            'videos': {}
        }

        for i, video_file in enumerate(video_files):
            print(
                f"  📹 Processing {video_file.name} ({i+1}/{len(video_files)})...")

            video_frames_dir = frames_dir / video_file.stem
            video_frames_dir.mkdir(exist_ok=True)

            try:
                cap = cv2.VideoCapture(str(video_file))

                if not cap.isOpened():
                    print(f"    ❌ Could not open {video_file.name}")
                    continue

                frame_count = 0
                extracted_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Extract frame at specified interval
                    if frame_count % frame_interval == 0:
                        frame_filename = video_frames_dir / \
                            f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(str(frame_filename), frame)
                        extracted_count += 1

                    frame_count += 1

                cap.release()

                extraction_log['videos'][video_file.name] = {
                    'total_frames': frame_count,
                    'extracted_frames': extracted_count,
                    'frames_dir': str(video_frames_dir)
                }

                extraction_log['total_frames_extracted'] += extracted_count
                extraction_log['videos_processed'] += 1

                print(
                    f"    ✅ Extracted {extracted_count} frames from {frame_count} total frames")

            except Exception as e:
                print(f"    ❌ Error processing {video_file.name}: {e}")

        # Save extraction log
        log_file = self.processed_data_path / "frame_extraction_log.json"
        with open(log_file, 'w') as f:
            json.dump(extraction_log, f, indent=2)

        print(f"✅ Frame extraction complete!")
        print(f"   Videos processed: {extraction_log['videos_processed']}")
        print(
            f"   Total frames extracted: {extraction_log['total_frames_extracted']}")
        print(f"   Frames saved to: {frames_dir}")
        print(f"   Log saved to: {log_file}")

        return str(frames_dir)

    def create_training_annotations(self) -> str:
        """Create training annotations for the extracted frames"""
        print("📝 Creating training annotations...")

        annotations_file = self.processed_data_path / "training_annotations.json"

        # Since these are all shoplifting videos, we'll mark them as positive examples
        # In a real scenario, you'd have manual annotations or more sophisticated labeling

        frames_dir = self.processed_data_path / "frames"
        if not frames_dir.exists():
            print("❌ No frames directory found. Run extract_frames() first.")
            return None

        annotations = {
            'dataset_info': {
                'name': 'UCF Anomaly Detection - Shoplifting',
                'description': 'Shoplifting videos from UCF Anomaly Detection dataset',
                'created': datetime.now().isoformat(),
                'total_videos': 0,
                'total_frames': 0
            },
            'annotations': {}
        }

        for video_dir in frames_dir.iterdir():
            if video_dir.is_dir():
                video_name = video_dir.name
                frame_files = list(video_dir.glob("*.jpg"))

                annotations['annotations'][video_name] = {
                    'video_name': video_name,
                    'label': 'shoplifting',  # All videos in this folder are shoplifting
                    'is_anomaly': True,
                    'frame_count': len(frame_files),
                    'frames': []
                }

                for frame_file in frame_files:
                    frame_info = {
                        'filename': frame_file.name,
                        'path': str(frame_file.relative_to(self.base_dir)),
                        'label': 'shoplifting',
                        'is_anomaly': True,
                        'confidence': 1.0  # High confidence since these are labeled shoplifting videos
                    }
                    annotations['annotations'][video_name]['frames'].append(
                        frame_info)

                annotations['dataset_info']['total_videos'] += 1
                annotations['dataset_info']['total_frames'] += len(frame_files)

        # Save annotations
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)

        print(f"✅ Training annotations created!")
        print(
            f"   Videos annotated: {annotations['dataset_info']['total_videos']}")
        print(
            f"   Frames annotated: {annotations['dataset_info']['total_frames']}")
        print(f"   Annotations saved to: {annotations_file}")

        return str(annotations_file)

    def get_dataset_summary(self) -> Dict:
        """Get a summary of the processed dataset"""
        summary = {
            'dataset_path': str(self.dataset_path) if self.dataset_path else None,
            'shoplifting_videos_path': str(self.shoplifting_videos_path) if self.shoplifting_videos_path else None,
            'processed_data_path': str(self.processed_data_path),
            'video_metadata': self.video_metadata
        }

        # Check for processed files
        frames_dir = self.processed_data_path / "frames"
        annotations_file = self.processed_data_path / "training_annotations.json"

        summary['frames_extracted'] = frames_dir.exists()
        summary['annotations_created'] = annotations_file.exists()

        if frames_dir.exists():
            video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
            summary['processed_videos'] = len(video_dirs)

            total_frames = 0
            for video_dir in video_dirs:
                total_frames += len(list(video_dir.glob("*.jpg")))
            summary['total_processed_frames'] = total_frames

        return summary


def main():
    """Example usage of the dataset manager"""
    print("🎯 Shoplifting Dataset Manager")
    print("=" * 50)

    # Initialize dataset manager
    manager = ShopliftingDatasetManager()

    # Download dataset
    dataset_path = manager.download_dataset()
    if not dataset_path:
        print("❌ Failed to download dataset")
        return

    # Analyze videos
    analysis = manager.analyze_videos()

    # Extract frames (limit to first 3 videos for initial testing)
    print("\n🎬 Extracting frames for training...")
    # Every 60 frames to reduce data size
    frames_path = manager.extract_frames(max_videos=3, frame_interval=60)

    # Create annotations
    annotations_path = manager.create_training_annotations()

    # Get summary
    summary = manager.get_dataset_summary()
    print("\n📊 Dataset Summary:")
    for key, value in summary.items():
        if key != 'video_metadata':
            print(f"   {key}: {value}")

    print("\n✅ Dataset preparation complete!")
    print("Next steps:")
    print("1. Run 'python training/model_trainer.py' to train the model")
    print("2. Use 'python training/evaluate_model.py' to test performance")
    print("3. Apply trained model to improve detection accuracy")


if __name__ == "__main__":
    main()
