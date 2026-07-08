import PIL.Image
import torch
import wandb
import os
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict

from tqdm import tqdm
from typing import List, Tuple, Any, Dict, Union, Optional
from dataclasses import dataclass
from brepdiff.models.base_model import BaseModel
from brepdiff.models.backbones.sequence_dm import (
    SEQUENCE_DIFFUSION_BACKBONES,
)
from brepdiff.models.tokenizers import Tokens
from brepdiff.models.uv_vae import UvVae, UvVaeOutput
from brepdiff.diffusion import (
    DIFFUSION_PROCESSES,
    Diffusion,
)
from brepdiff.config import Config

from brepdiff.datasets.abc_dataset import ABCDatasetOutput
from brepdiff.utils.vis import concat_h_pil, concat_v_pil, save_vid_from_img_seq
from brepdiff.primitives.uvgrid import UvGrid, stack_batched_uvgrids
from brepdiff.metrics.pc_metrics import compute_pc_metrics
from brepdiff.utils.vis import save_and_render_uvgrid


@dataclass(frozen=True)
class BrepDiffReconstruction:
    # Float tensor sampled with diffusion
    x: torch.Tensor
    uvgrids: UvGrid
    # ------------------------------
    # (optional) trajectories
    # ------------------------------
    uvgrids_traj: List[UvGrid]


class BrepDiff(BaseModel):
    name = "brepdiff"

    def __init__(self, config: Config, acc_logger):
        super().__init__(config, acc_logger)

        self.token_dim_split = [self.config.data_dim]  # coord
        self.token_dim_split.append(1)  # grid mask
        if self.config.n_face_types > 0:
            self.token_dim_split.append(1)  # face type index

        # Load token Vae
        if self.config.token_vae_ckpt_path == "":
            print("Warning: not using a pretrained token vae")
            self.token_vae = UvVae(self.config, acc_logger)
        else:
            self.config_token_vae, state_dict_token_vae = self._load_vae_config(
                self.config.token_vae_ckpt_path
            )
            if self.config.diffusion.z_conditioning:
                self.config_token_vae.diffusion.z_conditioning = True
            self.token_vae = UvVae(self.config_token_vae, acc_logger)

            self.token_vae.load_state_dict(state_dict_token_vae, strict=False)
            self.token_vae.requires_grad_(False)
            # Unfreeze graph encoder so it trains jointly with the diffusion backbone.
            # The graph encoder is not trained during the VAE stage (UvVae has no
            # compute_loss that uses the condition), so it starts from random init
            # and must be optimized here.
            if (
                self.config.diffusion.z_conditioning
                and hasattr(self.token_vae.tokenizer, "graph_encoder")
                and self.token_vae.tokenizer.graph_encoder is not None
            ):
                self.token_vae.tokenizer.graph_encoder.requires_grad_(True)
            assert (
                self.config.x_dim == self.config_token_vae.x_dim
            ), f"current x_dim: {self.config.x_dim}, token_vae_x_dim: {self.config_token_vae.x_dim}"

        self.seq_len = self.config.max_n_prims

        # Compute per_points_dim for backbone
        per_points_dim = self.config.data_dim + 1 + int(self.config.n_face_types > 0)

        # Diffusion backbone
        model_options = dict(self.config.diffusion.model["options"])
        model_options["per_points_dim"] = per_points_dim
        self.backbone = SEQUENCE_DIFFUSION_BACKBONES[
            self.config.diffusion.model["name"]
        ](
            input_dim=self.config.x_dim,
            seq_length=self.seq_len,
            z_dim=self.config.z_dim,
            n_z=self.config.n_z,
            z_conditioning=self.config.diffusion.z_conditioning,
            **model_options,
        )

        # Diffusion process
        self.diffusion_process: Diffusion = DIFFUSION_PROCESSES[
            self.config.diffusion.name
        ](self.backbone, self.config)

    def _load_vae_config(self, vae_ckpt_path: str) -> Tuple[Config, Dict]:
        """
        Loads the TokenVae config file
        """
        tmp = torch.load(vae_ckpt_path)["state_dict"]
        state_dict_vae = {}
        for k, v in tmp.items():
            if k.startswith("model."):
                state_dict_vae[k[len("model.") :]] = v

        config_vae_path = os.path.join(
            os.path.dirname(os.path.dirname(vae_ckpt_path)), "config.yaml"
        )
        config_vae = Config.from_yaml(open(config_vae_path, "r"))

        return config_vae, state_dict_vae

    def forward(
        self,
        x: torch.Tensor,
    ):
        raise NotImplementedError()

    def compute_loss(self, batch: ABCDatasetOutput, epoch: int, split="train"):
        device = next(self.parameters()).device

        if batch.uvgrid is not None:
            batch.uvgrid.to_tensor(device)

        # ------------------
        # TOKENIZE
        # ------------------
        token_vae_output: UvVaeOutput = self.token_vae(batch)

        condition = token_vae_output.tokens.condition
        # ------------------
        # DIFFUSION
        # ------------------
        out, target, t = self.diffusion_process(
            token_vae_output.tokens,
            z=condition,
            mask=token_vae_output.tokens.mask,
            prefix=split,
            empty_embeddings=None,
            return_timesteps=True,
        )
        loss, log_dict = self.uv_loss(
            out=out,
            target=target,
            empty_mask_gt=batch.uvgrid.empty_mask,
            split=split,
            timesteps=t,
            epoch=epoch,
        )

        log_dict[f"tokens/{split}/x0_rescaled_norm"] = [
            token_vae_output.tokens.sample.norm(dim=-1).mean().tolist()
        ]
        self.acc_logger.log_scalar_and_hist_dict(log_dict)
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch: ABCDatasetOutput,
        # Specifies whether to return full denoising trajectories
        return_traj: bool = False,
        cfg_scale: float = 1.0,
    ) -> BrepDiffReconstruction:
        n_batch = len(batch.name)

        token_vae_output: UvVaeOutput = self.token_vae(batch)
        z_sample = token_vae_output.tokens.condition
        attn_mask = token_vae_output.tokens.mask

        if self.config.sample_mode == "fixed":
            # Distribute different numbers of faces evenly across the batch
            num_faces_list = np.linspace(2, self.seq_len, n_batch).round().astype(int)
            attn_mask = torch.zeros(
                (n_batch, self.seq_len),
                dtype=bool,
                device=token_vae_output.tokens.mask.device,
            )
            for i, n_faces in enumerate(num_faces_list):
                attn_mask[i, n_faces:] = True  # Mask out faces after n_faces

        # -----------------
        # SAMPLE DIFFUSION
        # -----------------

        diff_sample, traj = self.diffusion_process.p_sample_loop(
            batch_size=n_batch,
            return_traj=return_traj,
            z=z_sample,
            mask=attn_mask,
            cfg_scale=cfg_scale,
            traj_stride=self.config.vis.trajectory_stride,
            # Use resampling if it is requested!
            use_resampling=self.config.test_use_resampling,
            resampling_jump_length=self.config.test_resampling_jump_length,
            resampling_repetitions=self.config.test_resampling_repetitions,
            resampling_start_t=self.config.test_resampling_start_t,
        )

        # -------------------------
        # DETOKENIZE & RECONSTRUCT
        # -------------------------
        uvgrids = self.token_vae.detokenizer.decode(
            tokens=Tokens(
                sample=diff_sample.get_x_for_detokenizer(),
                labels=diff_sample.l,
                condition=z_sample,
                mask=attn_mask,
            ),
            max_n_prims=self.seq_len,
        )

        uvgrids_traj = []
        if return_traj:
            for i in tqdm(
                range(0, len(traj)),
                desc="Reconstruction trajectory",
            ):
                x = traj[i].get_x_for_detokenizer()
                x = x[: self.config.vis.max_trajectory_points]
                if traj[i].l is not None:
                    labels = traj[i].l[: self.config.vis.max_trajectory_points]
                else:
                    labels = None
                if traj[i].z is not None:
                    z = traj[i].z[: self.config.vis.max_trajectory_points]
                else:
                    z = None
                if traj[i].mask is not None:
                    mask = traj[i].mask[: self.config.vis.max_trajectory_points]
                else:
                    mask = None

                tokens = Tokens(
                    sample=x,
                    labels=labels,
                    condition=z,
                    mask=mask,
                )
                uvgrids_traj_i = self.token_vae.detokenizer.decode(
                    tokens=tokens, max_n_prims=self.seq_len
                )
                uvgrids_traj.append(uvgrids_traj_i)

        return BrepDiffReconstruction(
            x=diff_sample.x,
            uvgrids=uvgrids,
            uvgrids_traj=uvgrids_traj,
        )

    @torch.no_grad()
    def sample_from_topology(
        self,
        face_adj: torch.Tensor,
        n_faces: Optional[Union[torch.Tensor, List[int]]] = None,
        return_traj: bool = True,
        cfg_scale: float = 1.0,
    ) -> BrepDiffReconstruction:
        device = next(self.parameters()).device

        if face_adj.dim() == 2:
            face_adj = face_adj.unsqueeze(0)
        assert face_adj.dim() == 3, "face_adj must be (N,N) or (B,N,N)"

        bsz, n, n2 = face_adj.shape
        assert n == n2, f"face_adj should be square, got {face_adj.shape}"
        assert n <= self.seq_len, (
            f"face_adj size {n} must be <= max_n_prims {self.seq_len}"
        )

        if n_faces is None:
            n_faces_tensor = torch.full((bsz,), n, device=device, dtype=torch.long)
        elif isinstance(n_faces, list):
            n_faces_tensor = torch.tensor(n_faces, device=device, dtype=torch.long)
        else:
            n_faces_tensor = n_faces.to(device=device, dtype=torch.long)

        assert n_faces_tensor.shape[0] == bsz, (
            f"n_faces length {n_faces_tensor.shape[0]} should match batch {bsz}"
        )
        assert torch.all(n_faces_tensor > 0), "n_faces should be > 0"
        assert torch.all(n_faces_tensor <= self.seq_len), (
            f"n_faces should be <= max_n_prims ({self.seq_len})"
        )

        face_adj = face_adj.to(device=device, dtype=torch.float32)
        face_adj_padded = torch.zeros(
            (bsz, self.seq_len, self.seq_len), device=device, dtype=torch.float32
        )
        face_adj_padded[:, :n, :n] = face_adj

        attn_mask = torch.zeros((bsz, self.seq_len), dtype=torch.bool, device=device)
        for i in range(bsz):
            attn_mask[i, n_faces_tensor[i] :] = True

        tokenizer = self.token_vae.tokenizer

        cond_valid_mask = torch.zeros((bsz, self.seq_len), dtype=torch.bool, device=device)
        cond_valid_mask[:, :n] = True
        z_sample = tokenizer.graph_encoder(
            face_adj=face_adj_padded,
            valid_mask= None,
        )

        if self.config.n_z > 1:
            z_sample = z_sample.repeat(1, self.config.n_z, 1)
        return_traj = True
        diff_sample, traj = self.diffusion_process.p_sample_loop(
            batch_size=bsz,
            return_traj=return_traj,
            z=z_sample,
            mask=attn_mask,
            cfg_scale=cfg_scale,
            traj_stride=self.config.vis.trajectory_stride,
            use_resampling=self.config.test_use_resampling,
            resampling_jump_length=self.config.test_resampling_jump_length,
            resampling_repetitions=self.config.test_resampling_repetitions,
            resampling_start_t=self.config.test_resampling_start_t,
        )

        uvgrids = self.token_vae.detokenizer.decode(
            tokens=Tokens(
                sample=diff_sample.get_x_for_detokenizer(),
                labels=diff_sample.l,
                condition=z_sample,
                mask=attn_mask,
            ),
            max_n_prims=self.seq_len,
        )

        uvgrids_traj = []
        if return_traj:
            for i in range(0, len(traj)):
                x = traj[i].get_x_for_detokenizer()
                x = x[: self.config.vis.max_trajectory_points]
                labels = (
                    traj[i].l[: self.config.vis.max_trajectory_points]
                    if traj[i].l is not None
                    else None
                )
                z = (
                    traj[i].z[: self.config.vis.max_trajectory_points]
                    if traj[i].z is not None
                    else None
                )
                mask = (
                    traj[i].mask[: self.config.vis.max_trajectory_points]
                    if traj[i].mask is not None
                    else None
                )

                tokens = Tokens(sample=x, labels=labels, condition=z, mask=mask)
                uvgrids_traj_i = self.token_vae.detokenizer.decode(
                    tokens=tokens,
                    max_n_prims=self.seq_len,
                )
                uvgrids_traj.append(uvgrids_traj_i)

        return BrepDiffReconstruction(
            x=diff_sample.x,
            uvgrids=uvgrids,
            uvgrids_traj=uvgrids_traj,
        )

    def vis(
        self,
        batch: ABCDatasetOutput,
        batch_idx: int,
        step: int,
        split: str,
        vis_traj: bool = True,
        render_blender: bool = True,
        render_gt: bool = True,
        log_wandb: bool = True,
    ):
        """
        Save uvgrid and visualize reconstructions
        """
        training = self.training
        self.eval()

        # Get local batch size and rank info
        n_batch = len(batch.name)
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # Directories
        gt_dir = os.path.join(self.config.log_dir, "vis", split, "gt")
        os.makedirs(gt_dir, mode=0o777, exist_ok=True)

        step_dir = os.path.join(
            self.config.log_dir, "vis", split, f"step-{str(step).zfill(9)}"
        )
        if split == "test":
            # to distinguish with weighted version
            step_dir = os.path.join(step_dir, self.config.dataset)
            if self.config.test_use_resampling:
                step_dir += "-resample"
        total_dir = os.path.join(step_dir, "total")
        os.makedirs(total_dir, mode=0o777, exist_ok=True)

        # Generate samples
        samples, uvgrids_samples, sample_dirs = [], [], []
        for cfg_scale in self.config.diffusion.cfg_scales:
            sample: BrepDiffReconstruction = self.sample(
                batch, return_traj=vis_traj, cfg_scale=cfg_scale
            )
            samples.append(sample)
            uvgrids_sample: UvGrid = sample.uvgrids
            uvgrids_sample.grid_mask = uvgrids_sample.grid_mask > 0
            uvgrids_samples.append(uvgrids_sample)

            sample_dir = os.path.join(step_dir, f"cfg_{cfg_scale:.2f}", "uvgrid")
            os.makedirs(sample_dir, mode=0o777, exist_ok=True)
            sample_dirs.append(sample_dir)

        uvgrids_gt = batch.uvgrid
        names = batch.name

        # select examples to visualize
        vis_config = self.config.vis
        n_examples = vis_config.n_examples

        if split == "test":
            # visualize and save all when testing
            vis_idxs = torch.arange(0, n_batch)
        else:
            if len(vis_config.vis_idxs) != 0:
                if vis_config.vis_idxs == "all":
                    vis_idxs = torch.arange(0, n_batch)
                else:
                    # Adjust vis_idxs for distributed training
                    global_vis_idxs = torch.tensor(vis_config.vis_idxs)
                    # Calculate which indices belong to this GPU
                    local_vis_idxs = []
                    for idx in global_vis_idxs:
                        # Calculate which GPU this index belongs to
                        target_rank = (idx // n_batch) % world_size
                        if target_rank == rank:
                            # Convert to local index
                            local_idx = idx % n_batch
                            if local_idx < n_batch:  # Make sure index is valid
                                local_vis_idxs.append(local_idx)
                    vis_idxs = torch.tensor(local_vis_idxs)
            else:
                if n_examples is None:
                    # visualize all local samples
                    vis_idxs = torch.arange(0, n_batch)
                else:
                    # Distribute n_examples across GPUs
                    n_examples_per_gpu = max(1, n_examples // world_size)
                    n_examples_local = min(n_examples_per_gpu, n_batch)
                    vis_idxs = torch.linspace(0, n_batch - 1, n_examples_local).int()

        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Create a list to store all images and their metadata for this rank
        rank_images = []

        try:
            for vis_idx in tqdm(vis_idxs, desc="rendering and saving"):
                img_list = []

                gt_name = names[vis_idx]
                if render_gt and (uvgrids_gt is not None):
                    img_gt = save_and_render_uvgrid(
                        save_dir=gt_dir,
                        save_name=gt_name,
                        uvgrids=uvgrids_gt,
                        vis_idx=vis_idx,
                        render_blender=render_blender,
                        use_cached_if_exists=True,
                    )
                    img_list.append(img_gt)

                if split == "train":
                    batch_size = self.config.batch_size
                elif split == "val":
                    batch_size = self.config.val_batch_size
                elif split == "test":
                    batch_size = self.config.test_batch_size
                else:
                    raise ValueError(f"{split} not allowed")
                vis_global_idx = vis_idx + batch_size * (batch_idx * world_size + rank)
                vis_global_idx = vis_global_idx.cpu().item()

                try:
                    for cfg_idx, cfg_scale in enumerate(
                        self.config.diffusion.cfg_scales
                    ):
                        img_sample = save_and_render_uvgrid(
                            save_dir=sample_dirs[cfg_idx],
                            save_name=str(vis_global_idx).zfill(5),
                            uvgrids=uvgrids_samples[cfg_idx],
                            vis_idx=vis_idx,
                            render_blender=render_blender,
                        )
                        img_list.append(img_sample)
                except KeyboardInterrupt:
                    print("\nInterrupted. Cleaning up...")
                    raise
                except Exception as e:
                    print(f"Error in vis at batch {batch_idx}, index {vis_idx}: {e}")
                    continue

        except KeyboardInterrupt:
            print("\nVisualization interrupted by user")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()  # Ensure all processes get the interrupt
            raise
        finally:
            self.train(training)  # Ensure model is returned to training state if needed

        # Synchronize all processes before gathering images
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Gather images from all ranks and log them
        if torch.distributed.is_initialized():
            # Create a list to store image metadata from all ranks
            gathered_images = [None] * world_size
            torch.distributed.all_gather_object(gathered_images, rank_images)

            # Log all images to wandb (all ranks will have the data but only rank 0 will log)
            if rank == 0 and log_wandb:
                for rank_data in gathered_images:
                    for img_data in rank_data:
                        img = wandb.Image(img_data["path"], caption=img_data["caption"])
                        self.acc_logger.log_imgs(img_data["tag"], [img])
        else:
            # Non-distributed case - log directly
            if log_wandb:
                for img_data in rank_images:
                    img = wandb.Image(img_data["path"], caption=img_data["caption"])
                    self.acc_logger.log_imgs(img_data["tag"], [img])

        # Similar modification for trajectory visualization
        if vis_traj:
            print("Visualizing trajectories")
            n_traj_points = self.config.vis.max_trajectory_points
            traj_points_per_gpu = max(1, n_traj_points // world_size)

            rank_videos = []

            for vis_idx in range(traj_points_per_gpu):
                vis_traj_dir = os.path.join(
                    self.config.log_dir,
                    "vis",
                    "traj",
                    f"step-{str(step).zfill(9)}",
                    str(vis_idx).zfill(4),
                )
                os.makedirs(vis_traj_dir, mode=0o777, exist_ok=True)
                img_seq = []

                # visualize trajectory only for the first cfg_scale
                for t, uvgrid_traj in enumerate(samples[0].uvgrids_traj):
                    uvgrid_traj.grid_mask = uvgrid_traj.grid_mask > 0
                    traj_t = t * self.config.vis.trajectory_stride
                    try:
                        render_objects = ["coord"]
                        img = save_and_render_uvgrid(
                            save_dir=vis_traj_dir,
                            save_name=f"{rank}_{str(traj_t).zfill(4)}",
                            uvgrids=uvgrid_traj,
                            vis_idx=vis_idx,
                            render_objects=render_objects,  # render only pc
                            render_blender=render_blender,
                        )
                    except Exception as e:
                        print(f"Error in vis at batch {vis_idx}: {e}")
                        continue

            # Synchronize and gather videos from all ranks
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                gathered_videos = [None] * world_size
                torch.distributed.all_gather_object(gathered_videos, rank_videos)

                if rank == 0 and log_wandb:
                    for rank_data in gathered_videos:
                        for vid_data in rank_data:
                            self.acc_logger.log_vid(vid_data["tag"], vid_data["path"])
            else:
                if log_wandb:
                    for vid_data in rank_videos:
                        self.acc_logger.log_vid(vid_data["tag"], vid_data["path"])

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.train(training)

    def test(
        self,
        batch: ABCDatasetOutput,
        global_step: int,
        batch_idx: int,
    ):
        """
        Tests the model on the given batch and saves results.

        Args:
        - batch: The batch of data to test on.
        - global_step: The current global step of the model.
        - batch_idx: The index of the current batch.
        - save: Whether to save the outputs or not.

        Returns:
        - A dictionary with the real, reconstructed, and generated zone graphs and point clouds.
        """
        print(f"Sampling test batch {batch_idx}...")
        device = next(self.parameters()).device
        training = self.training
        self.eval()

        n_batch = len(batch.name)
        labels = batch.uvgrid.prim_type
        sample: BrepDiffReconstruction = self.sample(batch)
        uvgrids_pred = sample.uvgrids
        uvgrids_gt = batch.uvgrid

        # Save generated UV grids to disk
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        save_dir = os.path.join(
            self.config.log_dir,
            "test_results",
            f"step-{str(global_step).zfill(9)}",
            "uvgrid"
        )
        os.makedirs(save_dir, mode=0o777, exist_ok=True)
        
        for i in range(n_batch):
            name = batch.name[i] if isinstance(batch.name, list) else f"batch{batch_idx}_sample{i}"
            save_name = f"rank{rank}_{name}"
            npz_path = os.path.join(save_dir, f"{save_name}.npz")
            uvgrids_pred.export_npz(file_path=npz_path, vis_idx=i)
            print(f"Saved generated UV grid: {npz_path}")

        self.train(training)

        return {
            "uvgrid_gts": uvgrids_gt,
            "uvgrid_preds": uvgrids_pred,
        }

    def compute_metrics(self, outputs, split):
        """
        Computes various metrics for the model outputs.

        Args:
        - outputs: The outputs from the model. (test_step)
        - split: The dataset split (e.g., 'val', 'test').

        Returns:
        - A dictionary containing the computed metrics.
        """
        print("Computing metrics...")
        device = next(self.parameters()).device

        uvgrid_gts = stack_batched_uvgrids([output["uvgrid_gts"] for output in outputs])
        coord_gts = uvgrid_gts.sample_pts(self.config.test_num_pts).to(device)[
            : self.config.n_pc_metric_samples
        ]  # B x N x 3
        uvgrid_preds = stack_batched_uvgrids(
            [output["uvgrid_preds"] for output in outputs]
        )
        uvgrid_preds.grid_mask = uvgrid_preds.grid_mask > 0
        coord_preds = uvgrid_preds.sample_pts(self.config.test_num_pts).to(device)[
            : self.config.n_pc_metric_samples
        ]  # B x N x 3

        metrics = {}
        with torch.no_grad():
            # -------------------------------------
            # Point cloud metrics (generation)
            # -------------------------------------
            print("Computing point cloud metrics...")
            gen_pc_metrics = compute_pc_metrics(
                coord_preds, coord_gts, normalize="longest_axis"
            )

            # Update metrics with point cloud metrics
            metrics.update(
                {
                    **{f"test/gen/{k}": v for k, v in gen_pc_metrics.items()},
                }
            )

        for k, v in metrics.items():
            if isinstance(metrics[k], torch.Tensor):
                metrics[k] = v.cpu().item()
            print(f"  {k}: {metrics[k]}")
            metrics[k] = [metrics[k]]
        print("Computing metrics finished!")

        return metrics

    def uv_loss(
        self,
        out: torch.Tensor,  # [B, n_prims, x_dim]
        target: torch.Tensor,  # [B, n_prims, x_dim]
        empty_mask_gt: torch.Tensor,  # [B, n_prims]
        split: str,
        timesteps: torch.Tensor = None,  # [B]
        epoch: int = None,
    ):
        n_batch = out.shape[0]
        n_prims = self.config.max_n_prims
        n_grid = self.config.n_grid

        # Use pre-calculated SNR and loss weights from diffusion process
        # snr = self.diffusion_process.snr[timesteps]  # [B]
        loss_weights = self.diffusion_process.loss_weight[timesteps]  # [B]
        # Reshape to match loss dimensions
        loss_weights = loss_weights.view(-1, 1).expand(-1, n_prims)  # [B, n_prims]

        # Store unweighted losses for logging
        unweighted_losses = {}

        # Reshape tensors for grid operations
        out = out.view(
            n_batch, n_prims, n_grid, n_grid, -1
        )  # [B, n_prims, n_grid, n_grid, C]
        target = target.view(
            n_batch, n_prims, n_grid, n_grid, -1
        )  # [B, n_prims, n_grid, n_grid, C]
        split_tensors_out = torch.split(
            out, self.token_dim_split, dim=-1
        )
        split_tensors_target = torch.split(
            target, self.token_dim_split, dim=-1
        )
        coord_out = split_tensors_out[0]
        grid_mask_out = split_tensors_out[1]
        coord_target = split_tensors_target[0]
        grid_mask_target = split_tensors_target[1]

        # Coordinate loss
        loss_coord = torch.mean(
            (coord_out - coord_target) ** 2, dim=-1
        )  # [B, n_prims, n_grid, n_grid]
        loss_coord = torch.mean(loss_coord, dim=[-1, -2])  # [B, n_prims]
        unweighted_losses["coord"] = loss_coord[~empty_mask_gt].mean()  # scalar
        masked_loss_coord = (
            loss_coord[~empty_mask_gt] * loss_weights[~empty_mask_gt]
        )  # [num_non_masked]
        loss_coord = masked_loss_coord.mean()  # scalar

        loss_grid_mask = torch.mean((grid_mask_out - grid_mask_target) ** 2, dim=-1)
        loss_grid_mask = torch.mean(loss_grid_mask, dim=[-1, -2])
        unweighted_losses["grid_mask"] = loss_grid_mask[~empty_mask_gt].mean()
        masked_loss_grid_mask = (
            loss_grid_mask[~empty_mask_gt] * loss_weights[~empty_mask_gt]
        )
        loss_grid_mask = masked_loss_grid_mask.mean()

        loss = (
            self.config.alpha_coord * loss_coord
            + self.config.alpha_grid_mask * loss_grid_mask
        )

        log_dict = {}
        with torch.no_grad():
            # Log unweighted losses as default
            log_dict[f"loss/{split}/total"] = [loss.tolist()]
            log_dict[f"loss/{split}/coord"] = [unweighted_losses["coord"].tolist()]
            log_dict[f"loss/{split}/grid_mask"] = [
                unweighted_losses["grid_mask"].tolist()
            ]

        return loss, log_dict

    @torch.no_grad()
    def sample_cond_only(
        self,
        condition_path: str,
        n_samples: int = 10,
        num_faces: int = 10,
        cfg_scale: float = 1.0,
        save_dir: str = None,
    ) -> BrepDiffReconstruction:
        """
        condition STEP 파일만으로 샘플링 (입력 데이터 불필요).
        STEP에서 face_adj를 추출하고 sample_from_topology를 호출한다.
        """
        from occwl.io import load_step
        from brepdiff.datasets.abc_dataset import get_face_adjacency

        device = next(self.parameters()).device
        self.eval()

        # STEP 파일에서 face adjacency 추출
        solids = load_step(condition_path)
        assert len(solids) > 0, f"No solids found in {condition_path}"
        face_adj_np = get_face_adjacency(solids[0])
        n_faces = face_adj_np.shape[0]
        print(f"Condition: {condition_path}, faces={n_faces}")

        face_adj = torch.from_numpy(face_adj_np).float().to(device)
        # (n_faces, n_faces) → (n_samples, n_faces, n_faces)
        face_adj_batch = face_adj.unsqueeze(0).expand(n_samples, -1, -1)
        n_faces_list = [num_faces] * n_samples

        result: BrepDiffReconstruction = self.sample_from_topology(
            face_adj=face_adj_batch,
            n_faces=n_faces_list,
            return_traj=False,
            cfg_scale=cfg_scale,
        )

        return result
