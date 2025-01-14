# Capton-project-
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from numpy.fft import fftshift, ifftshift, fft2, ifft2
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error as mse
from skimage.restoration import denoise_tv_chambolle

# Function to create a variable density sampling mask
def variable_density_mask(shape, center_fraction=0.2, sampling_rate=0.25):
    mask = np.zeros(shape, dtype=np.float32)
    center_size = int(center_fraction * min(shape))
    center = shape[0] // 2, shape[1] // 2

    for i in range(center[0] - center_size, center[0] + center_size):
        for j in range(center[1] - center_size, center[1] + center_size):
            mask[i, j] = 1

    num_samples = int(sampling_rate * shape[0] * shape[1]) - np.sum(mask)
    sampled_indices = np.random.choice(np.prod(shape), int(num_samples), replace=False)
    np.put(mask, sampled_indices, 1)

    return mask

# Forward operator using FFT
def forward_operator(image):
    return fftshift(fft2(ifftshift(image)))

# Inverse operator using IFFT
def inverse_operator(k_space):
    return np.abs(ifftshift(ifft2(fftshift(k_space))))

# SMART Reconstruction Algorithm
def smart_reconstruction(k_space, mask, original_image, iterations=500, step_size=0.05, tv_weight=0.05):
    reconstruction = inverse_operator(k_space * mask)

    for _ in range(iterations):
        current_k_space = forward_operator(reconstruction)
        k_space_difference = (k_space - current_k_space) * mask
        reconstruction += step_size * inverse_operator(k_space_difference)
        reconstruction = denoise_tv_chambolle(reconstruction, weight=tv_weight)
        current_ssim = ssim(original_image, reconstruction, data_range=1.0)  # Add data_range here
        if current_ssim >= 0.80:  # Target SSIM for 80% accuracy
            break

    return reconstruction


# Main code
image_size = 256
original_image = resize(shepp_logan_phantom(), (image_size, image_size))

k_space_full = forward_operator(original_image)
mask = variable_density_mask(original_image.shape, sampling_rate=0.25)
k_space_sampled = k_space_full * mask

reconstructed_image = smart_reconstruction(k_space_sampled, mask, original_image)
# Calculate performance metrics
final_ssim = ssim(original_image, reconstructed_image, data_range=1.0)  # Specify data_range
final_psnr = psnr(original_image, reconstructed_image, data_range=1.0)  # Specify data_range
final_mse = mse(original_image, reconstructed_image)

# Display results
plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Undersampled Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image, cmap="gray")
plt.axis("off")
plt.savefig("reconstructed_image.png")  # Save the reconstructed image
plt.show()


# Bar Chart for Performance Metrics
metrics = ["SSIM", "PSNR", "MSE"]
values = [final_ssim, final_psnr, final_mse]

plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=["blue", "green", "yellow"])
plt.title("Performance Metrics")
plt.ylabel("Value")
plt.ylim(0, max(values) + 5)
for i, v in enumerate(values):
    plt.text(i, v + 0.5, f"{v:.3f}", ha="center")
plt.savefig("performance_metrics.png")  # Save the bar chart
plt.show()

# Print performance metrics
print(f"Final SSIM: {final_ssim:.2f}")
print(f"Final PSNR: {final_psnr:.2f} dB")
print(f"Final MSE: {final_mse:.6f}")







