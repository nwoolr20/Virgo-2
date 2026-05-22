from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .memory import MemoryRecord, NeuralMemory


def save_memory(memory: NeuralMemory, path: str | Path) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)

    bundle = memory.to_bundle()
    field = bundle["field"]
    np.savez_compressed(
        target / "field.npz",
        input_dim=field["input_dim"],
        basis_count=field["basis_count"],
        regularization=field["regularization"],
        seed=field["seed"],
        frequencies=field["frequencies"],
        phases=field["phases"],
        weights=field["weights"] if field["weights"] is not None else np.array([]),
        coordinates=bundle["coordinates"],
        encoder_dimensions=bundle["encoder"]["dimensions"],
        encoder_seed=bundle["encoder"]["seed"],
    )

    with (target / "records.tsv").open("w", encoding="utf-8") as f:
        f.write("text\tsalience\tmetadata_json\n")
        for record in bundle["records"]:
            text = record["text"].replace("\t", " ").replace("\n", " ")
            meta = json.dumps(record.get("metadata", {}), ensure_ascii=False)
            f.write(f"{text}\t{record['salience']}\t{meta}\n")


def load_memory(path: str | Path) -> NeuralMemory:
    source = Path(path)
    npz = np.load(source / "field.npz", allow_pickle=False)
    from .coordinates import CoordinateEncoder
    from .field import NeuralField

    field = NeuralField(
        input_dim=int(npz["input_dim"]),
        basis_count=int(npz["basis_count"]),
        regularization=float(npz["regularization"]),
        seed=int(npz["seed"]),
    )
    field.frequencies = npz["frequencies"].astype(np.float64)
    field.phases = npz["phases"].astype(np.float64)
    weights = npz["weights"]
    field.weights = None if weights.size == 0 else weights.astype(np.float64)

    memory = NeuralMemory(
        encoder=CoordinateEncoder(dimensions=int(npz["encoder_dimensions"]), seed=str(npz["encoder_seed"])),
        field=field,
    )
    memory.coordinates = npz["coordinates"].astype(np.float64)

    with (source / "records.tsv").open("r", encoding="utf-8") as f:
        next(f)
        for line in f:
            text, salience, metadata_json = line.rstrip("\n").split("\t", 2)
            memory.records.append(MemoryRecord(text=text, salience=float(salience), metadata=json.loads(metadata_json)))
    return memory
