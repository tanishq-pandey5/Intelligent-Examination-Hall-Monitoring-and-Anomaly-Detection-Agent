#!/usr/bin/env python3
"""
==============================================================================
AI Examination Hall Monitoring - Model Training Pipeline
==============================================================================
CLI script to train/fine-tune models on classroom CCTV videos present in
the 'videos/' directory.

Usage:
    python3 train.py
    python3 train.py --videos videos/ --epochs 5 --mode all
    python3 train.py --mode behavior  # Train ML anomaly detector only
    python3 train.py --mode yolo --epochs 10  # Train YOLO detector only
==============================================================================
"""

import os
import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TrainCLI")

from src.trainer import ExamModelTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train AI Examination Hall Detection & Anomaly Models on video files."
    )
    parser.add_argument(
        "--videos",
        type=str,
        default="videos",
        help="Path to directory containing input classroom videos (default: videos/)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dataset",
        help="Path to output dataset directory (default: data/dataset)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="models",
        help="Directory to save trained model artifacts (default: models/)"
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default="outputs/training",
        help="Directory to save training metrics and logs (default: outputs/training)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs for YOLO fine-tuning (default: 5)"
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
        help="Frame sampling interval in seconds (default: 0.5s -> ~2 FPS)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "yolo", "behavior"],
        default="all",
        help="Training mode: 'all' (both YOLO & ML Anomaly Model), 'yolo', or 'behavior' (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('mps', 'cuda', 'cpu'). Auto-selects if omitted."
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🎓 AI EXAMINATION MONITORING - MODEL TRAINING PIPELINE")
    print(f"📁 Video Directory: {args.videos}")
    print(f"🎯 Training Mode:   {args.mode.upper()}")
    if args.mode in ["all", "yolo"]:
        print(f"🔄 YOLO Epochs:     {args.epochs}")
    print("=" * 70)

    if not os.path.isdir(args.videos):
        logger.error(f"Video directory '{args.videos}' does not exist!")
        sys.exit(1)

    trainer = ExamModelTrainer(
        videos_dir=args.videos,
        dataset_dir=args.dataset,
        models_dir=args.models,
        outputs_dir=args.outputs,
        device=args.device
    )

    train_yolo = args.mode in ["all", "yolo"]
    train_behavior = args.mode in ["all", "behavior"]

    try:
        report = trainer.run_full_pipeline(
            yolo_epochs=args.epochs,
            sample_interval_sec=args.sample_interval,
            train_yolo=train_yolo,
            train_behavior=train_behavior
        )

        print("\n" + "=" * 70)
        print("✅ TRAINING SUCCESSFULLY COMPLETED!")
        print(f"⏱️  Total Duration: {report['total_duration_sec']} seconds")
        print(f"📊 Dataset Extracted: {report['dataset_summary']['total_frames']} frames")
        print(f"📈 Behavioral Samples: {report['dataset_summary']['behavioral_samples']}")
        
        if yolo_res := report.get("yolo_training"):
            print(f"🎯 Fine-Tuned YOLO Weights: {yolo_res['trained_weights_path']}")
        
        if beh_res := report.get("behavior_anomaly_training"):
            print(f"🧠 Behavioral Anomaly Model: {beh_res['model_path']}")
            print(f"   Median Anomaly Score: {beh_res['anomaly_score_quantiles']['q50_median']}")
            print(f"   95th Percentile:      {beh_res['anomaly_score_quantiles']['q95']}")

        print("=" * 70)

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
