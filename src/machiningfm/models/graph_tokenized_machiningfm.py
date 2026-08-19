from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adaptive_graph_learning import AdaptiveGraphLearningStemGNN
from .decoder_only import DecoderOnlyTransformer, estimate_decoder_only_parameters
from .graph_tokenizer import GraphTokenizationLayer
from .heads import ForecastingHead, ImagePatchReconstructionHead, PatchReconstructionHead


class GraphTokenizedStemGNNDecoderOnlyMachiningFM(nn.Module):
    """Graph-tokenized StemGNN-style Decoder-Only foundation model."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.d_model = int(config.get("d_model", 512))
        self.patch_size = int(config.get("patch_size", 32))
        self.image_patch_size = int(config.get("image_patch_size", 16))
        self.max_channels = int(config.get("max_channels", 64))
        self.horizon = int(config.get("horizon", 64))
        self.graph_time_length = int(config.get("graph_time_length", 256))
        material_vocab = int(config.get("material_vocab_size", 2048))
        family_vocab = int(config.get("material_family_vocab_size", 512))
        confidence_vocab = int(config.get("material_confidence_vocab_size", 16))
        self.text_vocab_size = int(config.get("text_vocab_size", 8192))
        self.nc_context_dim = int(config.get("nc_context_dim", 0))
        self.text_embedding = nn.Embedding(self.text_vocab_size, self.d_model)
        self.material_name_embedding = nn.Embedding(material_vocab, self.d_model)
        self.material_family_embedding = nn.Embedding(family_vocab, self.d_model)
        self.material_confidence_embedding = nn.Embedding(confidence_vocab, self.d_model)
        self.global_context_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.dynamic_graph_summary_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.material_context_projection = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.nc_context_projection = (
            nn.Sequential(
                nn.LayerNorm(self.nc_context_dim),
                nn.Linear(self.nc_context_dim, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )
            if self.nc_context_dim > 0
            else None
        )
        graph_config = dict(config.get("adaptive_graph_learning", config.get("stemgnn", {})))
        graph_config.setdefault("d_model", self.d_model)
        self.adaptive_graph = AdaptiveGraphLearningStemGNN(graph_config)
        self.graph_tokenizer = GraphTokenizationLayer(config.get("graph_tokenization", {}), self.d_model)
        self.graph_token_mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.decoder = DecoderOnlyTransformer(config)
        self.next_token_head = nn.Linear(self.d_model, self.d_model)
        self.graph_token_reconstruction_head = nn.Linear(self.d_model, self.d_model)
        self.patch_reconstruction_head = PatchReconstructionHead(self.d_model, self.patch_size)
        self.frequency_reconstruction_head = PatchReconstructionHead(self.d_model, self.patch_size)
        self.image_reconstruction_head = ImagePatchReconstructionHead(self.d_model, self.image_patch_size)
        self.forecasting_head = ForecastingHead(self.d_model, self.max_channels, self.horizon)
        self.embedding_norm = nn.LayerNorm(self.d_model)

    @property
    def graph_token_count(self) -> int:
        return self.graph_tokenizer.graph_token_count

    def encode(self, batch: dict[str, Any]) -> dict[str, Any]:
        variables, variable_mask, variable_metadata = self._build_variable_inputs(batch)
        metadata_prior = self._metadata_prior(variable_metadata, variables.device, variables.dtype)
        graph = self.adaptive_graph(variables, variable_mask, metadata_prior)
        tokenized = self.graph_tokenizer(
            graph["variable_embeddings"],
            graph["learned_adjacency"],
            variable_mask,
            variable_metadata,
        )
        graph_tokens = tokenized["graph_tokens"]
        graph_token_targets = graph_tokens.clone()
        reconstruction_mask = batch.get("graph_token_reconstruction_mask")
        if isinstance(reconstruction_mask, Tensor):
            mask = reconstruction_mask.to(device=graph_tokens.device, dtype=torch.bool)
            if mask.shape != graph_tokens.shape[:2]:
                raise ValueError(
                    f"Expected graph_token_reconstruction_mask {tuple(graph_tokens.shape[:2])}, got {tuple(mask.shape)}"
                )
            graph_tokens = torch.where(mask.unsqueeze(-1), self.graph_token_mask_token.to(graph_tokens.dtype), graph_tokens)
        material_token = self._material_token(batch, graph_tokens.device)
        nc_token, nc_token_mask = self._nc_context_token(batch, graph_tokens.device)
        graph_summary = self.dynamic_graph_summary_token.expand(graph_tokens.shape[0], 1, -1)
        graph_summary = graph_summary + graph["variable_embeddings"].mean(dim=1, keepdim=True)
        global_token = self.global_context_token.expand(graph_tokens.shape[0], 1, -1)
        prefix_tokens = [global_token, material_token]
        prefix_masks = [
            torch.ones(graph_tokens.shape[0], 1, dtype=torch.bool, device=graph_tokens.device),
            torch.ones(graph_tokens.shape[0], 1, dtype=torch.bool, device=graph_tokens.device),
        ]
        nc_index = None
        if nc_token is not None and nc_token_mask is not None:
            nc_index = sum(token.shape[1] for token in prefix_tokens)
            prefix_tokens.append(nc_token)
            prefix_masks.append(nc_token_mask)
        prefix_tokens.append(graph_summary)
        prefix_masks.append(torch.ones(graph_tokens.shape[0], 1, dtype=torch.bool, device=graph_tokens.device))
        prefix_count = sum(token.shape[1] for token in prefix_tokens)
        tokens = torch.cat((*prefix_tokens, graph_tokens), dim=1)
        token_mask = torch.cat((*prefix_masks, tokenized["graph_token_mask"]), dim=1)
        hidden = self.decoder(tokens, token_mask)
        graph_hidden = hidden[:, prefix_count : prefix_count + graph_tokens.shape[1]]
        graph_embedding = _masked_mean(graph_hidden, tokenized["graph_token_mask"])
        if nc_index is not None:
            embedding_source = 0.4 * hidden[:, 0] + 0.4 * graph_embedding + 0.2 * hidden[:, nc_index]
        else:
            embedding_source = 0.5 * hidden[:, 0] + 0.5 * graph_embedding
        embedding = self.embedding_norm(embedding_source)
        return {
            "tokens": hidden,
            "token_mask": token_mask,
            "embedding": embedding,
            "graph_tokens": graph_tokens,
            "graph_token_targets": graph_token_targets,
            "graph_token_hidden": graph_hidden,
            "nc_context_hidden": hidden[:, nc_index] if nc_index is not None else None,
            "graph_token_mask": tokenized["graph_token_mask"],
            "graph_token_names": tokenized["graph_token_names"],
            "graph_token_types": tokenized["graph_token_types"],
            "graph_membership": tokenized["membership"],
            "variable_embeddings": graph["variable_embeddings"],
            "variable_mask": variable_mask,
            "variable_metadata": variable_metadata,
            "learned_adjacency": graph["learned_adjacency"],
            "graph_diagnostics": graph["diagnostics"],
            "spectral_prediction": graph["spectral_prediction"],
            "spectral_target": graph["spectral_target"],
        }

    def forward(self, batch: dict[str, Any], task: str | None = None) -> dict[str, Any]:
        output = self.encode(batch)
        output["next_token_prediction"] = self.next_token_head(output["tokens"])
        output["graph_token_reconstruction"] = self.graph_token_reconstruction_head(output["graph_token_hidden"])
        output["patch_reconstruction"] = self._reconstruct_patches(
            output["graph_token_hidden"],
            batch.get("sensor_series"),
            self.patch_reconstruction_head,
        )
        output["frequency_patch_reconstruction"] = self._reconstruct_patches(
            output["graph_token_hidden"],
            batch.get("frequency"),
            self.frequency_reconstruction_head,
        )
        output["image_patch_reconstruction"] = self._reconstruct_image_patches(output["graph_token_hidden"], batch.get("image"))
        sensor = batch.get("sensor_series")
        channels = int(sensor.shape[1]) if isinstance(sensor, Tensor) else self.max_channels
        output["forecast"] = self.forecasting_head(output["embedding"], channels)
        if task:
            output["prediction"] = output["forecast"] if task == "future_forecasting" else output["embedding"]
        return output

    def _build_variable_inputs(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, list[list[dict[str, Any]]]]:
        parts: list[Tensor] = []
        masks: list[Tensor] = []
        metadata_parts: list[list[list[dict[str, Any]]]] = []
        batch_size, device = _batch_context(batch, self)
        metadata = batch.get("metadata") or [{} for _ in range(batch_size)]
        sensor = batch.get("sensor_series")
        if isinstance(sensor, Tensor):
            parts.append(_resize_series(sensor, self.graph_time_length))
            sensor_mask = batch.get("sensor_mask")
            masks.append(sensor_mask.bool() if isinstance(sensor_mask, Tensor) else torch.ones(sensor.shape[:2], dtype=torch.bool, device=device))
            metadata_parts.append(_sensor_metadata(metadata, sensor.shape[1]))
        frequency = batch.get("frequency")
        if isinstance(frequency, Tensor):
            parts.append(_resize_series(frequency, self.graph_time_length))
            frequency_mask = batch.get("frequency_mask")
            masks.append(frequency_mask.bool() if isinstance(frequency_mask, Tensor) else torch.ones(frequency.shape[:2], dtype=torch.bool, device=device))
            metadata_parts.append(_frequency_metadata(metadata, frequency.shape[1]))
        condition = batch.get("condition")
        if isinstance(condition, Tensor):
            parts.append(_condition_series(condition, self.graph_time_length))
            condition_mask = batch.get("condition_mask")
            masks.append(condition_mask.bool() if isinstance(condition_mask, Tensor) else torch.ones(condition.shape[:2], dtype=torch.bool, device=device))
            metadata_parts.append(_condition_metadata(metadata, condition.shape[1]))
        image = batch.get("image")
        if isinstance(image, Tensor):
            parts.append(_image_series(image, self.graph_time_length))
            image_mask = batch.get("image_mask")
            masks.append(image_mask.bool().view(batch_size, 1) if isinstance(image_mask, Tensor) else torch.ones(batch_size, 1, dtype=torch.bool, device=device))
            metadata_parts.append(_single_metadata(metadata, "image", "image", "image"))
        text_ids = batch.get("text_ids")
        if isinstance(text_ids, Tensor):
            text_values = self.text_embedding(text_ids.long().clamp(0, self.text_vocab_size - 1)).mean(dim=-1)
            parts.append(_resize_series(text_values.unsqueeze(1), self.graph_time_length))
            text_mask = batch.get("text_mask")
            masks.append(text_mask.bool().any(dim=-1, keepdim=True) if isinstance(text_mask, Tensor) else torch.ones(batch_size, 1, dtype=torch.bool, device=device))
            metadata_parts.append(_single_metadata(metadata, "text_or_latent_context", "text_context", "text"))
        if not parts:
            parts.append(torch.zeros(batch_size, 1, self.graph_time_length, device=device))
            masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
            metadata_parts.append(_single_metadata(metadata, "global_metadata", "metadata_only", "metadata"))
        variables = torch.cat(parts, dim=1).unsqueeze(-1)
        variable_mask = torch.cat(masks, dim=1)
        variable_metadata = []
        for batch_index in range(batch_size):
            sample_metadata: list[dict[str, Any]] = []
            for part in metadata_parts:
                sample_metadata.extend(part[batch_index])
            variable_metadata.append(sample_metadata)
        return variables, variable_mask, variable_metadata

    def _metadata_prior(self, metadata: list[list[dict[str, Any]]], device: torch.device, dtype: torch.dtype) -> Tensor:
        batch = len(metadata)
        variable_count = max((len(sample) for sample in metadata), default=1)
        prior = torch.eye(variable_count, device=device, dtype=dtype).expand(batch, -1, -1).clone()
        for batch_index, sample in enumerate(metadata):
            for left, left_meta in enumerate(sample):
                for right, right_meta in enumerate(sample):
                    if left >= variable_count or right >= variable_count:
                        continue
                    score = 0.0
                    if left_meta.get("view") == right_meta.get("view"):
                        score += 0.35
                    if left_meta.get("sampling_rate_group") == right_meta.get("sampling_rate_group"):
                        score += 0.20
                    if left_meta.get("material") == right_meta.get("material"):
                        score += 0.10
                    if _axis(left_meta.get("name")) and _axis(left_meta.get("name")) == _axis(right_meta.get("name")):
                        score += 0.25
                    prior[batch_index, left, right] = max(float(prior[batch_index, left, right]), min(1.0, score))
        return prior

    def _material_token(self, batch: dict[str, Any], device: torch.device) -> Tensor:
        metadata = batch.get("metadata") or [{}]
        material_ids = []
        family_ids = []
        confidence_ids = []
        for item in metadata:
            material_ids.append(_stable_id(item.get("material", "unknown"), self.material_name_embedding.num_embeddings))
            family_ids.append(_stable_id(item.get("material_family", "unknown"), self.material_family_embedding.num_embeddings))
            confidence_ids.append(_stable_id(item.get("material_confidence", "low"), self.material_confidence_embedding.num_embeddings))
        material = torch.tensor(material_ids, dtype=torch.long, device=device)
        family = torch.tensor(family_ids, dtype=torch.long, device=device)
        confidence = torch.tensor(confidence_ids, dtype=torch.long, device=device)
        token = (
            self.material_name_embedding(material)
            + self.material_family_embedding(family)
            + self.material_confidence_embedding(confidence)
        )
        return self.material_context_projection(token).unsqueeze(1)

    def _nc_context_token(self, batch: dict[str, Any], device: torch.device) -> tuple[Tensor | None, Tensor | None]:
        if self.nc_context_projection is None:
            return None, None
        context = batch.get("nc_context")
        if not isinstance(context, Tensor):
            return None, None
        target_dtype = next(self.nc_context_projection.parameters()).dtype
        context = _fit_last_dim(context, self.nc_context_dim).to(device=device, dtype=target_dtype)
        mask = batch.get("nc_context_mask")
        if isinstance(mask, Tensor):
            token_mask = mask.to(device=device, dtype=torch.bool).view(context.shape[0], 1)
        else:
            token_mask = torch.ones(context.shape[0], 1, dtype=torch.bool, device=device)
        token = self.nc_context_projection(context).unsqueeze(1)
        return token, token_mask

    def _reconstruct_patches(self, hidden: Tensor, source: Any, head: nn.Module) -> Tensor:
        if not isinstance(source, Tensor):
            return hidden.new_zeros(hidden.shape[0], 0, self.patch_size)
        patches_per_channel = max(1, math.ceil(source.shape[-1] / self.patch_size))
        total = source.shape[1] * patches_per_channel
        repeated = _repeat_tokens(hidden, total)
        return head(repeated)

    def _reconstruct_image_patches(self, hidden: Tensor, image: Any) -> Tensor:
        if not isinstance(image, Tensor):
            return hidden.new_zeros(hidden.shape[0], 0, 3 * self.image_patch_size * self.image_patch_size)
        patches_per_axis = max(1, math.ceil(image.shape[-1] / self.image_patch_size))
        total = patches_per_axis * patches_per_axis
        return self.image_reconstruction_head(_repeat_tokens(hidden, total))


def estimate_graph_tokenized_decoder_only_parameters(config: dict[str, Any]) -> dict[str, int]:
    d_model = int(config.get("d_model", 512))
    nc_context_dim = int(config.get("nc_context_dim", 0))
    graph_tokens = len((config.get("graph_tokenization", {}).get("subgraphs") or {})) or 6
    decoder = estimate_decoder_only_parameters(config)
    embeddings = (
        int(config.get("text_vocab_size", 8192)) * d_model
        + int(config.get("material_vocab_size", 2048)) * d_model
        + int(config.get("material_family_vocab_size", 512)) * d_model
        + int(config.get("material_confidence_vocab_size", 16)) * d_model
        + graph_tokens * d_model
    )
    nc_context_projection = 0
    if nc_context_dim > 0:
        nc_context_projection = 2 * nc_context_dim + nc_context_dim * d_model + d_model + d_model * d_model + d_model
    graph_layer = (
        4 * d_model
        + 6 * d_model
        + 2 * d_model * int(config.get("adaptive_graph_learning", {}).get("graph_projection_dim", d_model))
        + d_model * d_model
        + 2 * d_model * d_model * int(config.get("adaptive_graph_learning", {}).get("graph_propagation", {}).get("num_layers", 2))
    )
    heads = 5 * d_model * d_model + d_model * int(config.get("patch_size", 32)) * 2
    total = int(decoder + embeddings + nc_context_projection + graph_layer + heads)
    return {
        "decoder_only": int(decoder),
        "embeddings": int(embeddings),
        "nc_context_projection": int(nc_context_projection),
        "adaptive_graph_and_tokenization": int(graph_layer),
        "heads": int(heads),
        "total": total,
    }


def _batch_context(batch: dict[str, Any], module: nn.Module) -> tuple[int, torch.device]:
    for value in batch.values():
        if isinstance(value, Tensor):
            return value.shape[0], value.device
    return 1, next(module.parameters()).device


def _resize_series(series: Tensor, length: int) -> Tensor:
    if series.shape[-1] == length:
        return series.float()
    return F.interpolate(series.float(), size=length, mode="linear", align_corners=False)


def _condition_series(condition: Tensor, length: int) -> Tensor:
    return condition.float().unsqueeze(-1).expand(-1, -1, length)


def _image_series(image: Tensor, length: int) -> Tensor:
    flat = image.float().flatten(1)
    resized = F.interpolate(flat.unsqueeze(1), size=length, mode="linear", align_corners=False)
    return resized


def _sensor_metadata(metadata: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    result = []
    for item in metadata:
        names = list(item.get("channel_names") or item.get("raw_channel_names") or [])
        descriptors = list(item.get("channel_descriptors") or [])
        sample = []
        for index in range(count):
            name = str(names[index]) if index < len(names) else f"sensor_{index}"
            descriptor = descriptors[index] if index < len(descriptors) and isinstance(descriptors[index], dict) else {}
            sample.append(_variable_metadata(item, name, _sensor_view(name, item), descriptor.get("quantity", "sensor")))
        result.append(sample)
    return result


def _frequency_metadata(metadata: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    result = []
    for item in metadata:
        names = list(item.get("frequency_names") or [])
        sample = []
        for index in range(count):
            name = str(names[index]) if index < len(names) else f"frequency_{index}"
            sample.append(_variable_metadata(item, name, "high_rate_fft", "frequency"))
        result.append(sample)
    return result


def _condition_metadata(metadata: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    return [
        [_variable_metadata(item, f"condition_{index}", "process_condition", "condition") for index in range(count)]
        for item in metadata
    ]


def _single_metadata(
    metadata: list[dict[str, Any]],
    view: str,
    name: str,
    quantity: str,
) -> list[list[dict[str, Any]]]:
    return [[_variable_metadata(item, name, view, quantity)] for item in metadata]


def _variable_metadata(item: dict[str, Any], name: str, view: str, quantity: str) -> dict[str, Any]:
    sampling_rate = item.get("sampling_rate") or item.get("estimated_sampling_rate")
    return {
        "name": name,
        "view": view,
        "quantity": quantity,
        "sampling_rate": sampling_rate,
        "sampling_rate_group": _sampling_rate_group(sampling_rate),
        "material": item.get("material", "unknown"),
        "material_family": item.get("material_family", "unknown"),
        "material_confidence": item.get("material_confidence", "low"),
        "dataset_group": item.get("dataset_id") or item.get("dataset_group"),
        "source_dataset_group": item.get("dataset_id") or item.get("dataset_group"),
    }


def _sensor_view(name: str, metadata: dict[str, Any]) -> str:
    lower = name.lower()
    rate = metadata.get("sampling_rate") or metadata.get("estimated_sampling_rate")
    try:
        sampling_rate = float(rate) if rate not in (None, "") else None
    except (TypeError, ValueError):
        sampling_rate = None
    if sampling_rate and sampling_rate >= 1000.0:
        return "high_rate_raw_timeseries"
    if any(token in lower for token in ("vibration", "acceler", "acoustic", "waveform", "hf")):
        return "high_rate_raw_timeseries"
    if any(token in lower for token in ("rpm", "feed", "axis", "position", "load", "servo", "gcode", "tool")):
        return "cnc_low_rate_timeseries"
    return "high_rate_raw_timeseries" if sampling_rate and sampling_rate >= 500.0 else "cnc_low_rate_timeseries"


def _sampling_rate_group(value: Any) -> str:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if rate >= 1000.0:
        return "high_rate"
    if rate > 0:
        return "low_rate"
    return "unknown"


def _axis(value: Any) -> str | None:
    if value is None:
        return None
    lower = str(value).lower()
    for axis in ("x", "y", "z", "u", "v", "w"):
        if f"_{axis}" in lower or f".{axis}" in lower or f"/{axis}" in lower:
            return axis
    return None


def _stable_id(value: Any, modulo: int) -> int:
    text = str(value or "unknown")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def _repeat_tokens(tokens: Tensor, count: int) -> Tensor:
    if count <= 0:
        return tokens[:, :0]
    repeats = math.ceil(count / max(1, tokens.shape[1]))
    return tokens.repeat(1, repeats, 1)[:, :count]


def _fit_last_dim(value: Tensor, size: int) -> Tensor:
    if value.shape[-1] == size:
        return value
    if value.shape[-1] > size:
        return value[..., :size]
    pad = value.new_zeros(*value.shape[:-1], size - value.shape[-1])
    return torch.cat((value, pad), dim=-1)


def _masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
