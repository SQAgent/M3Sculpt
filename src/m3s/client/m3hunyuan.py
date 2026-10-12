import base64
import os
from typing import Any, Dict, Optional

import requests
import trimesh

from m3s.configuration import HunyuanConfig


class M3HunyuanClient:
    def __init__(self, hunyuanconfig: HunyuanConfig):
        self.hunyuanconfig = hunyuanconfig
        self._make_request_dict()  # Create the request dictionary
        self.results: Optional[Dict[str, Any]] = None
        self.output_path = hunyuanconfig.output_path
        self.response: Optional[requests.Response] = None
        self.voxel_matrix = None
        self.largest_model_path =  os.path.join(self.hunyuanconfig.output_path, "largest_model.stl")

    def _make_request_dict(self):
        if self.hunyuanconfig.image_path:
            image_base64 = self._get_base64()
        elif self.hunyuanconfig.image:
            image_base64 = self.hunyuanconfig.image
        self.request_dict = {
            "remove_background": self.hunyuanconfig.remove_background,
            "texture": self.hunyuanconfig.texture,
            "type": self.hunyuanconfig.type,
            "seed": self.hunyuanconfig.seed,
            "octree_depth": self.hunyuanconfig.octree_depth,
            "guidance_scale": self.hunyuanconfig.guidance_scale,
            "face_count": self.hunyuanconfig.face_count,
            "image": image_base64,  # Base64 string
        }
        return self.request_dict

    def _get_base64(self):
        "Convert the image to a base64 string"
        if self.hunyuanconfig.image is None:
            if self.hunyuanconfig.image_path is None:
                print("Image path is not set")
                return None

        try:
            with open(self.hunyuanconfig.image_path, "rb") as img_file:
                encoded_string = base64.b64encode(
                    img_file.read()).decode('utf-8')
            return encoded_string
        except Exception as e:
            print(f"Failed to read the image file: {str(e)}")
            raise ValueError(f"Failed to read the image: {e}")

    def _create_stl_model(self):
        """
        Generate an STL file.
        """
        mesh = trimesh.load(
            os.path.join(self.hunyuanconfig.output_path, "model.glb"), file_type='glb')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        mesh.update_faces(mesh.nondegenerate_faces())

        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        mesh.rezero() if hasattr(mesh, "rezero") else None
        stl_path = os.path.join(self.hunyuanconfig.output_path, "model.stl")
        mesh.export(stl_path, file_type='stl')
        mesh = trimesh.load(stl_path, file_type='stl')
        components = mesh.split(only_watertight=False)
        watertight_comps = [c for c in components if c.is_watertight]   # Extract watertight components
        if not watertight_comps:
            raise RuntimeError(f"No watertight components found in {stl_path}.")

        largest = max(watertight_comps, key=lambda c: c.volume) # Find the component with the largest volume
        largest_model_path = os.path.join(self.hunyuanconfig.output_path, "largest_model.stl")
        largest.export(largest_model_path, file_type='stl')

    def _generate_voxel_matrix(self):
        """
        Generate a voxel array.
        """
        largest_model_mesh  = trimesh.load(
            os.path.join(self.hunyuanconfig.output_path, "largest_model.stl"), file_type='stl')
        resolution = 64 # Voxel resolution
        bounds = largest_model_mesh.bounds
        size = bounds[1] - bounds[0]
        pitch = float(max(size)) / (resolution - 1)
        voxelized = largest_model_mesh.voxelized(pitch=pitch).fill()
        self.voxel_matrix = voxelized.matrix

    def generate(self):
        self.response = requests.post(
            self.hunyuanconfig.url, json=self.request_dict, timeout=900)
        if self.response.status_code == 200:
            # Check whether the directory exists
            os.makedirs(self.hunyuanconfig.output_path, exist_ok=True)
            with open(os.path.join(self.hunyuanconfig.output_path, "model.glb"), "wb") as f:
                f.write(self.response.content)
            self._create_stl_model()
            self._generate_voxel_matrix()
            print("3D model created successfully.")

        else:
            self.response.raise_for_status()
            raise RuntimeError(f"Unexpected reconstruction status: {self.response.status_code}")
