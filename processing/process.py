import os
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import random
import argparse

# Import LightGlue
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
from lightglue import viz2d

# Setup logging
def setup_logging(log_dir="logs", debug=False):
    """Set up logging to both file and console with timestamps"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"lightglue_matching_{timestamp}.log")
    
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

# Calculate homography error
def calculate_homography_error(points1, points2, H_gt, threshold=3.0):
    """Calculate the percentage of matches consistent with the ground truth homography"""
    if len(points1) == 0:
        return 0.0, 0, 0
    
    # Convert points to homogeneous coordinates
    p1_homogeneous = np.hstack((points1, np.ones((len(points1), 1))))
    
    # Project points using ground truth homography
    p1_projected = np.dot(H_gt, p1_homogeneous.T).T
    p1_projected[:, 0] /= p1_projected[:, 2]
    p1_projected[:, 1] /= p1_projected[:, 2]
    
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((p1_projected[:, :2] - points2) ** 2, axis=1))
    
    # Calculate percentage of inliers
    inliers = np.sum(distances < threshold)
    return inliers / len(points1) * 100, inliers, len(points1)

def parse_arguments():
    parser = argparse.ArgumentParser(description='LightGlue HPatches Evaluation')
    parser.add_argument('--dataset_path', type=str, default='data/hpatches-sequences-release',
                        help='Path to HPatches dataset')
    parser.add_argument('--extractor', type=str, default='superpoint',
                        choices=['superpoint', 'disk', 'sift', 'aliked'],
                        help='Feature extractor to use')
    parser.add_argument('--max_keypoints', type=int, default=2048,
                        help='Maximum number of keypoints to detect')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with limited samples')
    parser.add_argument('--debug_samples', type=int, default=3,
                        help='Number of sequences to process in debug mode')
    parser.add_argument('--debug_pairs', type=int, default=2,
                        help='Number of image pairs per sequence in debug mode')
    parser.add_argument('--skip_compile', action='store_true',
                        help='Skip pytorch compilation even if available')
    parser.add_argument('--visualize_frequency', type=int, default=10,
                        help='Frequency of visualization (1 means every pair)')
    parser.add_argument('--view_only', action='store_true',
                        help='Process only viewpoint (v_) sequences')
    parser.add_argument('--intensity_only', action='store_true',
                        help='Process only illumination (i_) sequences')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize logging with debug mode if specified
    log_file = setup_logging(debug=args.debug)
    logging.info("Starting LightGlue matching on HPatches dataset")
    if args.debug:
        logging.debug("Running in DEBUG mode")
        logging.debug(f"Processing {args.debug_samples} sequences with {args.debug_pairs} pairs each")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Initialize feature extractor and matcher
    logging.info("Initializing feature extractor and matcher")
    
    # Configuration parameters
    extractor_type = args.extractor
    max_keypoints = args.max_keypoints
    
    # Set up the extractor based on the selected type
    if extractor_type == "superpoint":
        extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    elif extractor_type == "disk":
        extractor = DISK(max_num_keypoints=max_keypoints).eval().to(device)
    elif extractor_type == "sift":
        extractor = SIFT(max_num_keypoints=max_keypoints).eval().to(device)
    elif extractor_type == "aliked":
        extractor = ALIKED(max_num_keypoints=max_keypoints).eval().to(device)
    
    # Initialize the matcher with the corresponding feature type
    matcher = LightGlue(features=extractor_type).eval().to(device)
    logging.info(f"Using {extractor_type} extractor with max {max_keypoints} keypoints")
    
    # Try to compile for speed if torch version supports it and not in debug mode
    if not args.skip_compile and hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
        try:
            logging.info("Using torch.compile for optimization")
            matcher = torch.compile(matcher, mode='reduce-overhead')
        except Exception as e:
            logging.warning(f"Failed to compile matcher: {e}")
    
    # Set up dataset path
    dataset_path = args.dataset_path
    logging.info(f"Using dataset at: {dataset_path}")
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"results_{extractor_type}_{timestamp}"
    if args.debug:
        results_dir = f"debug_{results_dir}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)
    logging.info(f"Saving results to: {results_dir}")
    
    # Load dataset pairs
    logging.info("Loading HPatches dataset pairs")
    validation_data_base = []
    
    for dirs in os.listdir(dataset_path):
        # Filter by sequence type if specified
        if args.view_only and not dirs.startswith("v_"):
            continue
        if args.intensity_only and not dirs.startswith("i_"):
            continue
        
        current_set = []
        img_path1 = os.path.join(dataset_path, dirs, "1.ppm")
        
        if not os.path.exists(img_path1):
            logging.warning(f"Reference image not found: {img_path1}")
            continue
            
        for i in range(2, 7):
            img_path2 = os.path.join(dataset_path, dirs, f"{i}.ppm")
            transformation_path = os.path.join(dataset_path, dirs, f"H_1_{i}")
            
            if os.path.exists(img_path2) and os.path.exists(transformation_path):
                current_set.append((img_path1, img_path2, transformation_path))
            else:
                logging.warning(f"Missing files for pair: {dirs}/1.ppm - {dirs}/{i}.ppm")
                
        if current_set:
            validation_data_base.append(current_set)
    
    total_pairs = sum(len(s) for s in validation_data_base)
    logging.info(f"Found {len(validation_data_base)} sequences with {total_pairs} image pairs")
    
    # In debug mode, limit the number of sequences and pairs
    if args.debug:
        # Take a subset of sequences
        validation_data_base = validation_data_base[:args.debug_samples]
        # Take a subset of pairs from each sequence
        for i in range(len(validation_data_base)):
            validation_data_base[i] = validation_data_base[i][:args.debug_pairs]
        
        debug_pairs = sum(len(s) for s in validation_data_base)
        logging.debug(f"In debug mode, processing {len(validation_data_base)} sequences with {debug_pairs} image pairs")
    
    # Optional: shuffle sequences to get more varied visualizations during progress
    random.shuffle(validation_data_base)
    
    # Process all sequences
    all_results = []
    
    with torch.no_grad():
        for seq_idx, image_set in enumerate(tqdm(validation_data_base, desc="Processing sequences")):
            seq_name = os.path.basename(os.path.dirname(image_set[0][0]))
            logging.info(f"Processing sequence {seq_idx+1}/{len(validation_data_base)}: {seq_name}")
            
            # Process each image pair in the sequence
            for pair_idx, (img1_path, img2_path, transformation_path) in enumerate(tqdm(image_set, desc=f"Pairs in {seq_name}", leave=False)):
                try:
                    if args.debug:
                        logging.debug(f"Processing pair: {img1_path} - {img2_path}")
                    
                    # Load images using LightGlue's utility
                    image1 = load_image(img1_path).to(device)
                    image2 = load_image(img2_path).to(device)
                    
                    # Load ground truth homography
                    H_gt = np.loadtxt(transformation_path)
                    
                    # Extract keypoints and descriptors
                    start_time = time.time()
                    feats1 = extractor.extract(image1)
                    feats2 = extractor.extract(image2)
                    extract_time = time.time() - start_time
                    
                    if args.debug:
                        # Debug information about extracted features
                        logging.debug(f"Features 1: {feats1.keys()}")
                        logging.debug(f"Features 2: {feats2.keys()}")
                        logging.debug(f"Keypoints 1: {feats1['keypoints'].shape}")
                        logging.debug(f"Keypoints 2: {feats2['keypoints'].shape}")
                    
                    # Match features
                    start_time = time.time()
                    matches = matcher({"image0": feats1, "image1": feats2})
                    match_time = time.time() - start_time
                    
                    if args.debug:
                        # Debug information about matches
                        logging.debug(f"Matches: {matches.keys()}")
                        for k, v in matches.items():
                            if isinstance(v, torch.Tensor):
                                logging.debug(f"{k}: {v.shape} (tensor)")
                            else:
                                logging.debug(f"{k}: {type(v)}")
                    
                    # Get details about matches (remove batch dimension)
                    feats1, feats2, matches = [rbd(x) for x in [feats1, feats2, matches]]
                    
                    # Extract matched keypoints
                    kpts1 = feats1["keypoints"].cpu().numpy()
                    kpts2 = feats2["keypoints"].cpu().numpy()
                    match_indices = matches["matches"].cpu().numpy()
                    
                    num_kpts1 = len(kpts1)
                    num_kpts2 = len(kpts2)
                    num_matches = len(match_indices)
                    
                    if num_matches > 0:
                        matched_kpts1 = kpts1[match_indices[:, 0]]
                        matched_kpts2 = kpts2[match_indices[:, 1]]
                        
                        # Calculate accuracy using ground truth homography
                        accuracy, num_inliers, total_matches = calculate_homography_error(
                            matched_kpts1, matched_kpts2, H_gt
                        )
                    else:
                        matched_kpts1 = np.array([])
                        matched_kpts2 = np.array([])
                        accuracy, num_inliers, total_matches = 0, 0, 0
                    
                    # Get early-stopping layer if available
                    # Fix for the 'int' object has no attribute 'item' error
                    stop_layer = None
                    if "stop" in matches:
                        if isinstance(matches["stop"], torch.Tensor):
                            stop_layer = matches["stop"].item()
                        else:
                            stop_layer = matches["stop"]  # Already an int
                    
                    # Record results
                    pair_result = {
                        "sequence": seq_name,
                        "pair": f"{os.path.basename(img1_path)}-{os.path.basename(img2_path)}",
                        "num_keypoints1": num_kpts1,
                        "num_keypoints2": num_kpts2,
                        "num_matches": num_matches,
                        "num_inliers": num_inliers,
                        "inlier_ratio": accuracy / 100.0,
                        "extract_time": extract_time,
                        "match_time": match_time,
                        "total_time": extract_time + match_time,
                        "stop_layer": stop_layer
                    }
                    all_results.append(pair_result)
                    
                    # Log per-pair results
                    logging.info(f"  Pair {pair_idx+1}/{len(image_set)}: "
                                f"{os.path.basename(img1_path)}-{os.path.basename(img2_path)} - "
                                f"Keypoints: {num_kpts1}/{num_kpts2}, "
                                f"Matches: {num_matches}, "
                                f"Inliers: {num_inliers}/{num_matches if num_matches > 0 else 0} "
                                f"({accuracy:.2f}%), "
                                f"Time: {(extract_time + match_time)*1000:.2f}ms"
                                f"{f', Stopped at layer: {stop_layer}' if stop_layer else ''}")
                    
                    # Create visualizations for some pairs 
                    # Adjust frequency as needed
                    if (seq_idx * len(image_set) + pair_idx) % args.visualize_frequency == 0:
                        # Convert to numpy for visualization
                        img1_np = image1.cpu().numpy().transpose(1, 2, 0)
                        img2_np = image2.cpu().numpy().transpose(1, 2, 0)
                        
                        # Create directory for sequence
                        seq_vis_dir = os.path.join(results_dir, "visualizations", seq_name)
                        os.makedirs(seq_vis_dir, exist_ok=True)
                        
                        # Visualize matches
                        plt.figure(figsize=(15, 10))
                        
                        # Plot matched images
                        axes = viz2d.plot_images([img1_np, img2_np])
                        if num_matches > 0:
                            viz2d.plot_matches(matched_kpts1, matched_kpts2, color="lime", lw=0.2)
                        
                        # Add title and information
                        plt.suptitle(f"{seq_name}: {os.path.basename(img1_path)} to {os.path.basename(img2_path)}", fontsize=16)
                        viz2d.add_text(0, f'Matches: {num_matches}, Inliers: {num_inliers} ({accuracy:.2f}%)', fs=12)
                        if stop_layer:
                            viz2d.add_text(0, f'Stopped at layer: {stop_layer}/9', fs=12, pos=(0.02, 0.08))
                        
                        # Save the visualization
                        vis_path = os.path.join(seq_vis_dir, f"{os.path.basename(img1_path).split('.')[0]}_{os.path.basename(img2_path).split('.')[0]}_matches.png")
                        plt.savefig(vis_path, bbox_inches='tight')
                        plt.close()
                        
                        # Visualize keypoint pruning if available
                        if "prune0" in matches and "prune1" in matches:
                            plt.figure(figsize=(15, 10))
                            viz2d.plot_images([img1_np, img2_np])
                            
                            kpc0 = viz2d.cm_prune(matches["prune0"].cpu().numpy())
                            kpc1 = viz2d.cm_prune(matches["prune1"].cpu().numpy())
                            viz2d.plot_keypoints([kpts1, kpts2], colors=[kpc0, kpc1], ps=6)
                            
                            plt.suptitle(f"{seq_name}: Keypoint Pruning", fontsize=16)
                            
                            # Save keypoint pruning visualization
                            vis_path = os.path.join(seq_vis_dir, f"{os.path.basename(img1_path).split('.')[0]}_{os.path.basename(img2_path).split('.')[0]}_pruning.png")
                            plt.savefig(vis_path, bbox_inches='tight')
                            plt.close()
                    
                except Exception as e:
                    logging.error(f"Error processing pair {img1_path} - {img2_path}: {str(e)}")
                    if args.debug:
                        # In debug mode, print the full stack trace
                        import traceback
                        logging.debug(traceback.format_exc())
                    continue
    
    # Save overall results
    logging.info("Saving final results")
    
    # Save detailed results as numpy array
    results_file = os.path.join(results_dir, "results.npz")
    np.savez_compressed(results_file, results=all_results)
    
    # Also save as readable text
    with open(os.path.join(results_dir, "results_summary.txt"), "w") as f:
        f.write(f"LightGlue Matching Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Feature extractor: {extractor_type}, Max keypoints: {max_keypoints}\n")
        f.write(f"Dataset: {dataset_path}\n\n")
        
        # Calculate statistics
        if all_results:
            num_matches = [r["num_matches"] for r in all_results]
            num_inliers = [r["num_inliers"] for r in all_results]
            inlier_ratios = [r["inlier_ratio"] for r in all_results]
            extract_times = [r["extract_time"] * 1000 for r in all_results]  # convert to ms
            match_times = [r["match_time"] * 1000 for r in all_results]  # convert to ms
            total_times = [r["total_time"] * 1000 for r in all_results]  # convert to ms
            
            f.write("Overall Statistics:\n")
            f.write(f"Total pairs processed: {len(all_results)}\n")
            f.write(f"Average keypoints per image: {np.mean([r['num_keypoints1'] for r in all_results]):.2f}\n")
            f.write(f"Average matches per pair: {np.mean(num_matches):.2f}\n")
            f.write(f"Average inliers per pair: {np.mean(num_inliers):.2f}\n")
            f.write(f"Average inlier ratio: {np.mean(inlier_ratios):.4f}\n")
            f.write(f"Average extraction time: {np.mean(extract_times):.2f} ms\n")
            f.write(f"Average matching time: {np.mean(match_times):.2f} ms\n")
            f.write(f"Average total time: {np.mean(total_times):.2f} ms\n\n")
            
            # Add sequence-specific statistics
            f.write("Per-Sequence Statistics:\n")
            sequences = set(r["sequence"] for r in all_results)
            for seq in sorted(sequences):
                seq_results = [r for r in all_results if r["sequence"] == seq]
                seq_inlier_ratio = np.mean([r["inlier_ratio"] for r in seq_results])
                seq_matches = np.mean([r["num_matches"] for r in seq_results])
                seq_time = np.mean([r["total_time"] * 1000 for r in seq_results])
                
                f.write(f"  {seq}: {len(seq_results)} pairs, "
                       f"Avg matches: {seq_matches:.1f}, "
                       f"Avg inlier ratio: {seq_inlier_ratio:.4f}, "
                       f"Avg time: {seq_time:.2f} ms\n")
            
            # Create summary plots
            logging.info("Creating summary plots")
            plt.figure(figsize=(20, 15))
            
            plt.subplot(2, 2, 1)
            plt.hist(num_matches, bins=30, alpha=0.7)
            plt.title("Number of Matches Distribution")
            plt.xlabel("Matches")
            plt.ylabel("Count")
            plt.grid(alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.hist(inlier_ratios, bins=30, alpha=0.7)
            plt.title("Inlier Ratio Distribution")
            plt.xlabel("Inlier Ratio")
            plt.ylabel("Count")
            plt.grid(alpha=0.3)
            
            plt.subplot(2, 2, 3)
            plt.scatter(num_matches, num_inliers, alpha=0.5)
            plt.title("Matches vs. Inliers")
            plt.xlabel("Number of Matches")
            plt.ylabel("Number of Inliers")
            plt.grid(alpha=0.3)
            
            plt.subplot(2, 2, 4)
            plt.hist(total_times, bins=30, alpha=0.7)
            plt.title("Total Processing Time (ms)")
            plt.xlabel("Time (ms)")
            plt.ylabel("Count")
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "summary_plots.png"))
            plt.close()
            
            # If we have stop_layer information, create a plot for it
            stop_layers = [r["stop_layer"] for r in all_results if r["stop_layer"] is not None]
            if stop_layers:
                plt.figure(figsize=(10, 6))
                plt.hist(stop_layers, bins=range(1, 11), alpha=0.7)
                plt.title("Early Stopping Layer Distribution")
                plt.xlabel("Layer")
                plt.ylabel("Count")
                plt.xticks(range(1, 10))
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(results_dir, "stop_layer_distribution.png"))
                plt.close()
    
    logging.info(f"All results saved to {results_dir}")
    logging.info("Processing complete!")

if __name__ == "__main__":
    main()