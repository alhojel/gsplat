import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
from PIL import Image
from torch import Tensor, optim

import torch
from PIL import Image
from gsplat.project_gaussians import ProjectGaussians
from gsplat.rasterize import RasterizeGaussians
from gsplat.sh import SphericalHarmonics

### 3D IMPLEMENTATION START

def num_sh_bases(degree: int):
    # Compute the number of spherical harmonics bases for a given degree
    return (degree + 1) ** 2


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor

class SphericalHarmonicsTrainer:
    """Trains Gaussian blobs with spherical harmonics to fit a 3D scene."""

    def __init__(self, gt_images, camera_views, val_images, val_camera_views, num_points=2000, sh_degree=3, output_dir=None, focal = 1):
        self.device = torch.device("cuda:0")
        self.gt_images = [img.to(device=self.device) for img in gt_images]
        self.camera_views = camera_views

        self.output_dir = output_dir

        self.val_images = [img.to(device=self.device) for img in val_images]
        self.val_camera_views = val_camera_views

        self.num_points = num_points
        self.sh_degree = sh_degree
        self.num_sh_coeff = num_sh_bases(sh_degree)

        self.focal = focal
        BLOCK_X, BLOCK_Y = 16, 16
       
        self.H, self.W = gt_images[0].shape[0], gt_images[0].shape[1]
        self.tile_bounds = (
            (self.W + BLOCK_X - 1) // BLOCK_X,
            (self.H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )

        self.validation_interval = 500

        self._init_scene()

    def _init_scene(self):
        """Initialize Gaussian blobs with spherical harmonics"""
        bd = 2
        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        self.sh_coeffs = torch.rand(self.num_points, num_sh_bases(self.sh_degree), 3, device=self.device)

        self.opacities = torch.ones((self.num_points, 1), device=self.device)

        #DOES THIS NEED TO BE ADAPTED TO THE 3D Case? Code copy pasted from 2D example:
        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        #What changes must be done for quats?
        self.quats = self.quats / self.quats.norm(dim=1, keepdim=True)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.sh_coeffs.requires_grad = True
        self.opacities.requires_grad = True
        self.quats.requires_grad = True
        #self.viewmat.requires_grad = False

    def train(self, validate=True, iterations=1000, lr=0.01):
        optimizer = optim.Adam([self.means, self.scales, self.sh_coeffs, self.opacities], lr)
        mse_loss = torch.nn.MSELoss()

        times = [0] * 3
        for iter in range(iterations):
            total_loss = 0
            for view_idx, (gt_image, viewmat) in enumerate(zip(self.gt_images, self.camera_views)):
                start = time.time()

                # Check dimensions of tensors before projection
                print(f"Iteration {iter}, View {view_idx}: means shape {self.means.shape}, scales shape {self.scales.shape}, quats shape {self.quats.shape}")
                print(f"viewmat shape {viewmat.shape}, focal {self.focal}, image dimensions {self.H}, {self.W}, tile bounds {self.tile_bounds}")

                # Project Gaussian blobs to 2D for the current view
                xys, depths, radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(
                    self.means, self.scales, 1, self.quats, viewmat, viewmat, self.focal, self.focal, 
                    self.W / 2, self.H / 2, self.H, self.W, self.tile_bounds
                )

                times[0] += time.time() - start
                start = time.time()

                # Check dimensions after projection and before SH
                print(f"Projection output: xys shape {xys.shape}, depths shape {depths.shape}, radii shape {radii.shape}")

                # Compute spherical harmonics
                sh_rendered = SphericalHarmonics.apply(self.sh_degree, xys, self.sh_coeffs)

                # Check dimensions before rasterization
                print(f"Spherical Harmonics output: sh_rendered shape {sh_rendered.shape}")

                # Rasterize the scene
                out_img = RasterizeGaussians.apply(
                    xys, depths, radii, conics, num_tiles_hit, sh_rendered, self.opacities, 
                    self.H, self.W
                )
                #torch.cuda.synchronize()
                times[1] += time.time() - start
                # Compute loss and backpropagate
                loss = mse_loss(out_img, gt_image)
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                #torch.cuda.synchronize()
                times[2] += time.time() - start
                optimizer.step()

            print(f"Iteration {iter + 1}/{iterations}, Loss: {total_loss.item() / len(self.gt_images)}")
            #Validaiton step    
            if iter % self.validation_interval == 0:
                with torch.no_grad():
                    total_loss = 0
                    mse_loss = torch.nn.MSELoss()

                    for frame_idx, (gt_image, viewmat) in enumerate(zip(self.val_images, self.val_camera_views)):

                        # Project Gaussian blobs to 2D for the current view
                        xys, depths, radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(
                            self.means, self.scales, 1, viewmat, viewmat, self.focal, self.focal, 
                            self.W / 2, self.H / 2, self.H, self.W, self.tile_bounds
                        )

                        # Compute spherical harmonics
                        sh_rendered = SphericalHarmonics.apply(self.sh_degree, xys, self.sh_coeffs)

                        # Rasterize the scene
                        out_img = RasterizeGaussians.apply(
                            xys, depths, radii, conics, num_tiles_hit, sh_rendered, self.opacities, 
                            self.H, self.W
                        )

                        loss = mse_loss(out_img, gt_image)
                        total_loss += loss.item()

                    if self.output_dir:
                        out_img_np = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                        out_frame = Image.fromarray(out_img_np)
                        out_dir = os.path.join(self.output_dir, "renders")
                        os.makedirs(out_dir, exist_ok=True)

                        # Save individual image with iteration and frame index in the filename
                        out_frame.save(f"{out_dir}/validation_iter{iter}_frame{frame_idx}.png")

                    avg_loss = total_loss / len(self.val_images)
                    print(f"Validation Loss at iteration {iter}: {avg_loss}")

        print(
            f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}"
        )
        print(
            f"Per step(s):\nProject: {times[0]/iterations:.5f}, Rasterize: {times[1]/iterations:.5f}, Backward: {times[2]/iterations:.5f}"
        )
    

def main(
    height: int = 200,
    width: int = 200,
    num_points: int = 1,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 1,
    lr: float = 0.01,
    validation_interval: int = 5
) -> None:
    data = np.load(f"source/lego_200x200.npz")
    images_train = torch.tensor(data["images_train"] / 255.0, dtype=torch.float32)
    c2ws_train = torch.tensor(data["c2ws_train"], dtype=torch.float32)
    
    #For validation
    images_val = torch.tensor(data["images_val"] / 255.0, dtype=torch.float32)
    c2ws_val = torch.tensor(data["c2ws_val"], dtype=torch.float32)

    focal = data["focal"].item()
    
    print("A")
    trainer = SphericalHarmonicsTrainer(gt_images=images_train, camera_views=c2ws_train, val_images=images_val, val_camera_views=c2ws_val, num_points=num_points, output_dir="output", focal=focal)
    print("B")
    trainer.train(
        iterations=iterations,
        lr=lr,
    )
    print("C")

if __name__ == "__main__":
    tyro.cli(main)