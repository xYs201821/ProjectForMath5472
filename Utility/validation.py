"""
With the help of this script, you can calculate the FID score between two sets of images provided by its path.
"""

# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torchvision.models import Inception_V3_Weights
# import numpy as np
# from scipy import linalg
# from pathlib import Path
# from torch.nn.functional import adaptive_avg_pool2d
# from torchvision import transforms
# from tqdm import tqdm
# from PIL import Image
# import matplotlib.pyplot as plt
# def load_pil_images(pil_images, size=299):
#     transform = transforms.Compose([
#         transforms.Resize((size, size), antialias=True),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                              std=[0.229, 0.224, 0.225])
#     ])
    
#     tensors = []
#     for img in tqdm(pil_images, desc="Loading images"):
#         try:
#             if img.mode != 'RGB':
#                 img = img.convert('RGB')
            
#             tensor = transform(img)
#             tensors.append(tensor)
            
#         except Exception as e:
#             print(f"Error processing image: {str(e)}")
#             continue
    
#     if not tensors:
#         raise ValueError("No images were successfully loaded!")
    
#     return torch.stack(tensors)


# def get_inception_feature(images, model, batch_size=4):
#     model.eval()
#     features = []
#     n_batches = (len(images) + batch_size - 1) // batch_size
    
#     with torch.no_grad():
#         for i in tqdm(range(n_batches), desc="Extracting features"):
#             start = i * batch_size
#             end = min(start + batch_size, len(images))
#             batch = images[start:end].cuda()
            
#             # Forward pass
#             feat = model(batch)
            
#             features.append(feat.cpu())
    
#     features = torch.cat(features, dim=0).numpy()
    
#     return features


# def calculate_fid_from_pil(real_pil_images, fake_pil_images):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Load inception model
#     inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
#     inception.fc = nn.Identity()  # Remove final fully connected layer
#     inception.to(device)
#     inception.eval()
    
#     # Convert PIL images to tensors
#     print("Processing real images...")
#     real_images = load_pil_images(real_pil_images)
#     print(f"Loaded {len(real_images)} real images")
    
#     print("Processing fake images...")
#     fake_images = load_pil_images(fake_pil_images)
#     print(f"Loaded {len(fake_images)} fake images")
    
#     # Extract features
#     print("Extracting features from real images...")
#     real_features = get_inception_feature(real_images, inception)
    
#     print("Extracting features from fake images...")
#     fake_features = get_inception_feature(fake_images, inception)
    
#     # Calculate statistics
#     mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
#     mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
#     # Calculate FID
#     diff = mu1 - mu2
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#     fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
#     return float(fid)


# def load_images_from_directory(directory):
#     """Load all images from a directory"""
#     image_paths = Path(directory).glob('*.[jp][pn][g]')
#     images = []
#     for path in image_paths:
#         try:
#             img = Image.open(path)
#             images.append(img)
#         except Exception as e:
#             print(f"Error loading {path}: {e}")
#     return images

# # Convert to PIL images using your function
# def calculate_fid_from_path(realpath='validation/images', fakepath='validation/output/pre'):
#     real_images = load_images_from_directory(realpath)
#     fake_images = load_images_from_directory(fakepath)

#     # Calculate FID
#     fid_score = calculate_fid_from_pil(real_images, fake_images)
#     print(f'FID Score: {fid_score:.2f}')
#     return fid_score

# calculate_fid_from_path(fakepath="validation/output/sds")

# from cleanfid import fid

# print(fid.compute_fid("Pancakes100/base", "Pancakes100/vsd"))