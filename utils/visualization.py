import matplotlib.pyplot as plt
import os

def visualize_results(image, true_mask, pred_mask, model_name, index):
    # Create results directory if it doesn't exist
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Side-by-side comparison (original)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image)
    ax1.imshow(true_mask, alpha=0.3, cmap='Reds')
    ax1.set_title('Original Image with True Mask')
    ax1.axis('off')
    
    ax2.imshow(image)
    ax2.imshow(pred_mask, alpha=0.3, cmap='Blues')
    ax2.set_title('Original Image with Predicted Mask')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/{model_name}_comparison_{index}.png')
    plt.close()

    # New side-by-side comparison (simple comparison)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Convert pred_mask to binary
    # Also useful to look at pred_mask as grayscale since gives
    # indication of model's confidence. cmap='gray'
    binary_mask = (pred_mask > 0.5).astype(int)
    ax2.imshow(binary_mask, cmap='binary')
    ax2.set_title('Predicted Mask (Binary)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/{model_name}_original_vs_mask_{index}.png')
    plt.close()

    # Overlay of predicted cracks (also using binary mask)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(binary_mask, alpha=0.5, cmap='binary')
    plt.title('Predicted Cracks Overlay (Binary)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/{model_name}_overlay_{index}.png')
    plt.close()