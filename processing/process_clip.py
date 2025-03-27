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
from PIL import Image
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.spatial.distance import cdist

# Import CLIP
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Setup logging
def setup_logging(log_dir="logs"):
    """Set up logging to both file and console with timestamps"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"clip_similarity_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def load_clip_model(model_name="ViT-B/32", device=None):
    """
    Load a pre-trained CLIP model
    
    Args:
        model_name: The CLIP model variant to use (e.g., "ViT-B/32", "ViT-B/16", "RN50")
        device: The torch device to use
        
    Returns:
        model: The CLIP model
        preprocess: The CLIP preprocessing pipeline
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"Loading CLIP model: {model_name} on {device}")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()  # Set to evaluation mode
    
    logging.info(f"CLIP model loaded successfully")
    return model, preprocess

def extract_clip_embedding(image_path, model, preprocess, device=None):
    """
    Extract CLIP embeddings from an image
    
    Args:
        image_path: Path to the image file
        model: The CLIP model
        preprocess: The CLIP preprocessing pipeline
        device: The torch device to use
        
    Returns:
        embedding: The normalized CLIP embedding vector
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            
        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    except Exception as e:
        logging.error(f"Error extracting embedding from {image_path}: {e}")
        return None

def compute_similarity_metrics(embedding1, embedding2):
    """
    Compute various similarity metrics between two embeddings
    
    Args:
        embedding1, embedding2: Embedding vectors
        
    Returns:
        dict: Dictionary of similarity metrics
    """
    # Convert to numpy if they are torch tensors
    if isinstance(embedding1, torch.Tensor):
        embedding1 = embedding1.cpu().numpy()
    if isinstance(embedding2, torch.Tensor):
        embedding2 = embedding2.cpu().numpy()
    
    # Reshape if needed
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()
    
    # Cosine similarity (higher is more similar)
    cosine_sim = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    # Euclidean distance (lower is more similar)
    euclidean_dist = np.linalg.norm(embedding1 - embedding2)
    
    # Manhattan/L1 distance (lower is more similar)
    manhattan_dist = np.sum(np.abs(embedding1 - embedding2))
    
    # Dot product (higher is more similar)
    dot_product = np.dot(embedding1, embedding2)
    
    return {
        "cosine_similarity": float(cosine_sim),
        "euclidean_distance": float(euclidean_dist),
        "manhattan_distance": float(manhattan_dist),
        "dot_product": float(dot_product)
    }

def check_homography_consistency(metrics, H_gt, threshold=0.7):
    """
    Determine if the similarity metrics are consistent with the ground truth homography
    
    Args:
        metrics: Dictionary of similarity metrics
        H_gt: Ground truth homography matrix
        threshold: Similarity threshold for consistency
        
    Returns:
        is_consistent: Boolean indicating if the metrics are consistent with the homography
    """
    # This is a simplified heuristic:
    # - For small homography changes (viewpoint/intensity), similarity should be high
    # - For large homography changes, similarity can be lower
    
    # Calculate homography "difficulty" - how much the image changes
    # We use the determinant as a simple measure
    h_difficulty = np.abs(np.linalg.det(H_gt) - 1.0)
    
    # Adjust threshold based on difficulty
    adjusted_threshold = threshold - min(0.3, h_difficulty * 0.5)
    
    # Check if cosine similarity is above the adjusted threshold
    return metrics["cosine_similarity"] > adjusted_threshold

def plot_similarity_distribution(similarities, output_path):
    """
    Create a plot of similarity metric distributions
    
    Args:
        similarities: List of similarity metric dictionaries
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Extract different metrics
    cosine_sims = [s["cosine_similarity"] for s in similarities]
    euclidean_dists = [s["euclidean_distance"] for s in similarities]
    manhattan_dists = [s["manhattan_distance"] for s in similarities]
    
    # Plot distributions
    plt.subplot(2, 2, 1)
    plt.hist(cosine_sims, bins=30, alpha=0.7)
    plt.title(f"Cosine Similarity Distribution\nMean: {np.mean(cosine_sims):.4f}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(euclidean_dists, bins=30, alpha=0.7)
    plt.title(f"Euclidean Distance Distribution\nMean: {np.mean(euclidean_dists):.4f}")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.hist(manhattan_dists, bins=30, alpha=0.7)
    plt.title(f"Manhattan Distance Distribution\nMean: {np.mean(manhattan_dists):.4f}")
    plt.xlabel("Manhattan Distance")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    
    # Plot correlation between metrics
    plt.subplot(2, 2, 4)
    plt.scatter(cosine_sims, euclidean_dists, alpha=0.5)
    plt.title("Cosine Similarity vs Euclidean Distance")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Euclidean Distance")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_similarity_pairs(image_pairs, similarities, metrics_key, output_path, n_samples=10):
    """
    Create a visualization of image pairs sorted by similarity
    
    Args:
        image_pairs: List of (image1_path, image2_path) tuples
        similarities: List of similarity metric dictionaries
        metrics_key: Which metric to use for sorting (e.g., "cosine_similarity")
        output_path: Path to save the visualization
        n_samples: Number of pairs to visualize
    """
    # Sort pairs by the specified metric
    if metrics_key in ["cosine_similarity", "dot_product"]:
        # For these metrics, higher values mean more similar
        sorted_indices = np.argsort([-s[metrics_key] for s in similarities])
    else:
        # For distance metrics, lower values mean more similar
        sorted_indices = np.argsort([s[metrics_key] for s in similarities])
    
    # Select samples (most similar, middle, least similar)
    step = len(sorted_indices) // (n_samples - 1) if n_samples > 1 else 1
    sample_indices = sorted_indices[::step][:n_samples]
    
    # Create visualization grid
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
    
    for i, idx in enumerate(sample_indices):
        img1_path, img2_path = image_pairs[idx]
        similarity = similarities[idx][metrics_key]
        
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Display images
        axes[i, 0].imshow(img1)
        axes[i, 0].set_title(f"Image 1: {os.path.basename(img1_path)}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img2)
        axes[i, 1].set_title(f"Image 2: {os.path.basename(img2_path)}\n{metrics_key}: {similarity:.4f}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_retrieval_performance(image_pairs, similarities, sequence_info):
    """
    Evaluate how well the similarity metrics perform for image retrieval
    
    Args:
        image_pairs: List of (image1_path, image2_path) tuples
        similarities: List of similarity dictionaries
        sequence_info: List of sequence names for each pair
        
    Returns:
        retrieval_metrics: Dictionary of retrieval performance metrics
    """
    # Extract sequences and metrics
    sequences = [seq for seq in sequence_info]
    cosine_sims = [s["cosine_similarity"] for s in similarities]
    
    # Create ground truth matrix - 1 if same sequence, 0 otherwise
    unique_sequences = list(set(sequences))
    seq_to_idx = {seq: i for i, seq in enumerate(unique_sequences)}
    seq_indices = [seq_to_idx[seq] for seq in sequences]
    
    n = len(seq_indices)
    ground_truth = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Images in the same sequence should be similar
            if seq_indices[i] == seq_indices[j]:
                ground_truth[i, j] = 1
    
    # Create similarity matrix
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = compute_similarity_metrics(
                np.array(similarities[i]["embedding"]), 
                np.array(similarities[j]["embedding"])
            )["cosine_similarity"]
    
    # Flatten for precision-recall calculation (ignore diagonals)
    y_true = ground_truth[~np.eye(n, dtype=bool)].flatten()
    y_scores = sim_matrix[~np.eye(n, dtype=bool)].flatten()
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    
    return {
        "average_precision": average_precision,
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds
    }

def main():
    # Initialize logging
    log_file = setup_logging()
    logging.info("Starting CLIP similarity evaluation on HPatches dataset")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Initialize CLIP model
    model_name = "ViT-B/32"  # Smaller transformer as requested
    model, preprocess = load_clip_model(model_name, device)
    
    # Set up dataset path
    dataset_path = "data/hpatches-sequences-release"
    logging.info(f"Using dataset at: {dataset_path}")
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"clip_results_{model_name.replace('/', '_')}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)
    logging.info(f"Saving results to: {results_dir}")
    
    # Load dataset pairs
    logging.info("Loading HPatches dataset pairs")
    validation_data_base = []
    
    for dirs in os.listdir(dataset_path):
        current_set = []
        # You can filter by sequence type (v_ for viewpoint changes, i_ for intensity changes)
        # if dirs.startswith("i_"): continue
        
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
    
    # Optional: limit sequences for faster processing during development
    # validation_data_base = validation_data_base[:10]  # Uncomment to process only 10 sequences
    
    # Process all sequences
    all_results = []
    all_image_pairs = []
    all_similarities = []
    all_sequence_info = []
    
    with torch.no_grad():
        for seq_idx, image_set in enumerate(tqdm(validation_data_base, desc="Processing sequences")):
            seq_name = os.path.basename(os.path.dirname(image_set[0][0]))
            logging.info(f"Processing sequence {seq_idx+1}/{len(validation_data_base)}: {seq_name}")
            
            # Process each image pair in the sequence
            for pair_idx, (img1_path, img2_path, transformation_path) in enumerate(tqdm(image_set, desc=f"Pairs in {seq_name}", leave=False)):
                try:
                    # Extract CLIP embeddings
                    start_time = time.time()
                    
                    embedding1 = extract_clip_embedding(img1_path, model, preprocess, device)
                    embedding2 = extract_clip_embedding(img2_path, model, preprocess, device)
                    
                    if embedding1 is None or embedding2 is None:
                        logging.warning(f"Failed to extract embeddings for {img1_path} - {img2_path}")
                        continue
                    
                    extract_time = time.time() - start_time
                    
                    # Compute similarity metrics
                    start_time = time.time()
                    similarity_metrics = compute_similarity_metrics(embedding1, embedding2)
                    # Store the embeddings for later retrieval evaluation
                    similarity_metrics["embedding"] = embedding1.tolist()  # Store the first image's embedding
                    compute_time = time.time() - start_time
                    
                    # Load ground truth homography
                    H_gt = np.loadtxt(transformation_path)
                    
                    # Check if similarity is consistent with homography
                    is_consistent = check_homography_consistency(similarity_metrics, H_gt)
                    
                    # Record results
                    pair_result = {
                        "sequence": seq_name,
                        "pair": f"{os.path.basename(img1_path)}-{os.path.basename(img2_path)}",
                        "similarity_metrics": similarity_metrics,
                        "extract_time": extract_time,
                        "compute_time": compute_time,
                        "total_time": extract_time + compute_time,
                        "is_consistent": is_consistent,
                        "h_det": float(np.linalg.det(H_gt))  # Determinant of homography as a measure of transformation difficulty
                    }
                    all_results.append(pair_result)
                    
                    # Store for later visualization
                    all_image_pairs.append((img1_path, img2_path))
                    all_similarities.append(similarity_metrics)
                    all_sequence_info.append(seq_name)
                    
                    # Log per-pair results
                    logging.info(f"  Pair {pair_idx+1}/{len(image_set)}: "
                                f"{os.path.basename(img1_path)}-{os.path.basename(img2_path)} - "
                                f"Cosine Similarity: {similarity_metrics['cosine_similarity']:.4f}, "
                                f"Euclidean Distance: {similarity_metrics['euclidean_distance']:.4f}, "
                                f"Time: {(extract_time + compute_time)*1000:.2f}ms, "
                                f"Consistent: {is_consistent}")
                    
                    # Create visualizations for some pairs (adjust frequency as needed)
                    if (seq_idx * len(image_set) + pair_idx) % 20 == 0:
                        # Create directory for sequence
                        seq_vis_dir = os.path.join(results_dir, "visualizations", seq_name)
                        os.makedirs(seq_vis_dir, exist_ok=True)
                        
                        # Load images for visualization
                        img1 = cv2.imread(img1_path)
                        img2 = cv2.imread(img2_path)
                        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                        
                        # Create visualization
                        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                        
                        axes[0].imshow(img1)
                        axes[0].set_title(f"Image 1: {os.path.basename(img1_path)}")
                        axes[0].axis('off')
                        
                        axes[1].imshow(img2)
                        axes[1].set_title(f"Image 2: {os.path.basename(img2_path)}")
                        axes[1].axis('off')
                        
                        plt.suptitle(f"Cosine Similarity: {similarity_metrics['cosine_similarity']:.4f}, "
                                    f"Euclidean Distance: {similarity_metrics['euclidean_distance']:.4f}\n"
                                    f"Consistent with Homography: {is_consistent}", 
                                    fontsize=14)
                        
                        vis_path = os.path.join(seq_vis_dir, f"{os.path.basename(img1_path).split('.')[0]}_{os.path.basename(img2_path).split('.')[0]}_similarity.png")
                        plt.savefig(vis_path, bbox_inches='tight')
                        plt.close()
                    
                except Exception as e:
                    logging.error(f"Error processing pair {img1_path} - {img2_path}: {str(e)}")
                    continue
    
    # Save overall results
    logging.info("Saving final results")
    
    if len(all_image_pairs) > 10:
        retrieval_metrics = evaluate_retrieval_performance(
            all_image_pairs, 
            all_similarities,
            all_sequence_info
        )
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(retrieval_metrics["recall"], retrieval_metrics["precision"], marker='.')
        plt.title(f"Precision-Recall Curve\nAverage Precision: {retrieval_metrics['average_precision']:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(results_dir, "precision_recall_curve.png"))
        plt.close()

    
    # Save detailed results as numpy array
    # Remove the embeddings before saving to reduce file size
    for result in all_results:
        if "embedding" in result["similarity_metrics"]:
            del result["similarity_metrics"]["embedding"]
            
    results_file = os.path.join(results_dir, "clip_similarity_results.npz")
    np.savez_compressed(results_file, results=all_results)
    
    # Create distribution plots
    plot_similarity_distribution(all_similarities, os.path.join(results_dir, "similarity_distributions.png"))
    
    # Create visualizations of most/least similar pairs
    visualize_similarity_pairs(
        all_image_pairs, 
        all_similarities, 
        "cosine_similarity", 
        os.path.join(results_dir, "cosine_similarity_samples.png")
    )
    
    visualize_similarity_pairs(
        all_image_pairs, 
        all_similarities, 
        "euclidean_distance", 
        os.path.join(results_dir, "euclidean_distance_samples.png")
    )
    
    # Evaluate retrieval performance if we have enough data
    # if len(all_image_pairs) > 10:
    #     retrieval_metrics = evaluate_retrieval_performance(
    #         all_image_pairs, 
    #         all_similarities,
    #         all_sequence_info
    #     )
        
    #     # Plot precision-recall curve
    #     plt.figure(figsize=(10, 8))
    #     plt.plot(retrieval_metrics["recall"], retrieval_metrics["precision"], marker='.')
    #     plt.title(f"Precision-Recall Curve\nAverage Precision: {retrieval_metrics['average_precision']:.4f}")
    #     plt.xlabel("Recall")
    #     plt.ylabel("Precision")
    #     plt.grid(alpha=0.3)
    #     plt.savefig(os.path.join(results_dir, "precision_recall_curve.png"))
    #     plt.close()
    
    # Save results summary
    with open(os.path.join(results_dir, "clip_similarity_summary.txt"), "w") as f:
        f.write(f"CLIP Similarity Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CLIP model: {model_name}\n")
        f.write(f"Dataset: {dataset_path}\n\n")
        
        # Calculate statistics
        if all_results:
            cosine_similarities = [r["similarity_metrics"]["cosine_similarity"] for r in all_results]
            euclidean_distances = [r["similarity_metrics"]["euclidean_distance"] for r in all_results]
            manhattan_distances = [r["similarity_metrics"]["manhattan_distance"] for r in all_results]
            consistencies = [1 if r["is_consistent"] else 0 for r in all_results]
            extract_times = [r["extract_time"] * 1000 for r in all_results]  # convert to ms
            compute_times = [r["compute_time"] * 1000 for r in all_results]  # convert to ms
            total_times = [r["total_time"] * 1000 for r in all_results]  # convert to ms
            
            f.write("Overall Statistics:\n")
            f.write(f"Total pairs processed: {len(all_results)}\n")
            f.write(f"Average cosine similarity: {np.mean(cosine_similarities):.4f}\n")
            f.write(f"Average euclidean distance: {np.mean(euclidean_distances):.4f}\n")
            f.write(f"Average manhattan distance: {np.mean(manhattan_distances):.4f}\n")
            f.write(f"Consistency percentage: {np.mean(consistencies)*100:.2f}%\n")
            f.write(f"Average extraction time: {np.mean(extract_times):.2f} ms\n")
            f.write(f"Average computation time: {np.mean(compute_times):.2f} ms\n")
            f.write(f"Average total time: {np.mean(total_times):.2f} ms\n\n")
            
            # Add sequence-specific statistics
            f.write("Per-Sequence Statistics:\n")
            sequences = set(r["sequence"] for r in all_results)
            for seq in sorted(sequences):
                seq_results = [r for r in all_results if r["sequence"] == seq]
                seq_cosine = np.mean([r["similarity_metrics"]["cosine_similarity"] for r in seq_results])
                seq_euclidean = np.mean([r["similarity_metrics"]["euclidean_distance"] for r in seq_results])
                seq_consistency = np.mean([1 if r["is_consistent"] else 0 for r in seq_results]) * 100
                
                f.write(f"  {seq}: {len(seq_results)} pairs, "
                       f"Avg cosine similarity: {seq_cosine:.4f}, "
                       f"Avg euclidean distance: {seq_euclidean:.4f}, "
                       f"Consistency: {seq_consistency:.2f}%\n")
            
            # Analyze different sequence types if we have both
            v_sequences = [r for r in all_results if r["sequence"].startswith("v_")]
            i_sequences = [r for r in all_results if r["sequence"].startswith("i_")]
            
            if v_sequences and i_sequences:
                f.write("\nComparison by Sequence Type:\n")
                
                v_cosine = np.mean([r["similarity_metrics"]["cosine_similarity"] for r in v_sequences])
                v_euclidean = np.mean([r["similarity_metrics"]["euclidean_distance"] for r in v_sequences])
                v_consistency = np.mean([1 if r["is_consistent"] else 0 for r in v_sequences]) * 100
                
                i_cosine = np.mean([r["similarity_metrics"]["cosine_similarity"] for r in i_sequences])
                i_euclidean = np.mean([r["similarity_metrics"]["euclidean_distance"] for r in i_sequences])
                i_consistency = np.mean([1 if r["is_consistent"] else 0 for r in i_sequences]) * 100
                
                f.write(f"  Viewpoint (v) sequences ({len(v_sequences)} pairs):\n")
                f.write(f"    Avg cosine similarity: {v_cosine:.4f}\n")
                f.write(f"    Avg euclidean distance: {v_euclidean:.4f}\n")
                f.write(f"    Consistency: {v_consistency:.2f}%\n\n")
                
                f.write(f"  Illumination (i) sequences ({len(i_sequences)} pairs):\n")
                f.write(f"    Avg cosine similarity: {i_cosine:.4f}\n")
                f.write(f"    Avg euclidean distance: {i_euclidean:.4f}\n")
                f.write(f"    Consistency: {i_consistency:.2f}%\n")
    
    # Create a correlation matrix between homography changes and similarity metrics
    h_dets = [abs(r["h_det"] - 1.0) for r in all_results]  # Homography difficulty
    
    plt.figure(figsize=(10, 8))
    plt.scatter(h_dets, cosine_similarities, alpha=0.5)
    plt.title("Homography Difficulty vs. Cosine Similarity")
    plt.xlabel("Homography Difficulty (|det(H) - 1|)")
    plt.ylabel("Cosine Similarity")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, "homography_vs_similarity.png"))
    plt.close()
    
    logging.info(f"All results saved to {results_dir}")
    logging.info("Processing complete!")

if __name__ == "__main__":
    main()