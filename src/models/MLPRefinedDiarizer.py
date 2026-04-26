from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch

from models.baseline import BaselineDiarizer
from debug import vprint


class MLPEmbeddingRefiner(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()

        self.residual = residual

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        refined = self.net(embeddings)

        if self.residual:
            refined = refined + embeddings

        refined = F.normalize(refined, p=2, dim=-1)
        return refined

    # src/models/mlp_advanced.py

    class MLPRefinedDiarizer(BaselineDiarizer):
        """
        Advanced diarizer:

            ECAPA/WavLM embeddings
            -> MLP refinement layer
            -> clustering/post-processing from BaselineDiarizer
        """

        def __init__(
                self,
                *args,
                mlp_hidden_dim: int = 256,
                mlp_dropout: float = 0.1,
                mlp_residual: bool = True,
                mlp_checkpoint: str | None = None,
                **kwargs,
        ):
            super().__init__(*args, **kwargs)

            self.mlp_hidden_dim = mlp_hidden_dim
            self.mlp_dropout = mlp_dropout
            self.mlp_residual = mlp_residual
            self.mlp_checkpoint = mlp_checkpoint

            self.mlp_refiner = None

        def _build_mlp_refiner(self, embedding_dim: int):
            model = MLPEmbeddingRefiner(
                embedding_dim=embedding_dim,
                hidden_dim=self.mlp_hidden_dim,
                dropout=self.mlp_dropout,
                residual=self.mlp_residual,
            ).to(self.device)

            if self.mlp_checkpoint is not None:
                state = torch.load(self.mlp_checkpoint, map_location=self.device)
                model.load_state_dict(state)

            model.eval()
            return model

        def _extract_embeddings(self, windows) -> np.ndarray:
            embeddings = super()._extract_embeddings(windows)

            if self.mlp_refiner is None:
                self.mlp_refiner = self._build_mlp_refiner(
                    embedding_dim=embeddings.shape[1]
                )

            vprint("[MLP] Refining embeddings.")

            x = torch.tensor(
                embeddings,
                dtype=torch.float32,
                device=self.device,
            )

            with torch.no_grad():
                refined = self.mlp_refiner(x)

            return refined.cpu().numpy().astype(np.float32)