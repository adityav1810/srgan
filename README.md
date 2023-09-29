# SuperVision: Image Super Resolution using SRGAN
## Overview
SuperVision is a project that explores image super resolution using the Super Resolution Generative Adversarial Network (SRGAN). The SRGAN architecture is enhanced with advanced techniques, including perceptual loss, adversarial loss, and content loss, to achieve exceptional improvements in image quality and perceptual fidelity.

The project aims to demonstrate the capability of SRGAN in upscaling images by a scale factor of 4, achieving a remarkable peak signal-to-noise ratio (PSNR) of 42.4. This high PSNR value signifies the effectiveness of the model in enhancing image clarity and detail, making it highly valuable in various domains such as computer vision and medical imaging.

## Key Features
Implementation of Super Resolution Generative Adversarial Network (SRGAN).
Integration of perceptual loss, adversarial loss, and content loss for enhanced training.
Ablation study to optimize hyperparameters and loss functions, particularly focusing on perceptual loss.
Successful upscaling of images by a scale factor of 4, showcasing significant improvements in image quality.
## Technical Details
Programming Language: Python
Libraries/Frameworks: PyTorch, NumPy, Matplotlib
Loss Functions:
Perceptual Loss: Utilized to capture high-level features, enhancing perceptual fidelity.
Adversarial Loss: Introduced to ensure generated images are realistic and match the distribution of real images.
Content Loss: Incorporated to maintain the content similarity between high-resolution and super-resolved images.
## Usage
Clone the repository: git clone https://github.com/adityav1810/srgan.git
Navigate to the project directory: cd srgan
Train the SRGAN model using the provided dataset and configurations using SRGAN.ipynb

## Citation
Built on top of the wonderful resources provided by : https://github.com/aladdinpersson
