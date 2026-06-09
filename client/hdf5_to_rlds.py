# tsc
import argparse
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _snake_to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part)


def _load_steps(path: Path, min_action_norm: float) -> list:
    with h5py.File(path, "r") as f:
        actions = f["action"][:].astype(np.float32)
        images = f["observations"]["images"][:].astype(np.uint8)
        states = f["observations"]["state"][:].astype(np.float32)
        instruction = f.attrs.get("language_instruction", "")
        reward = float(f.attrs.get("reward", 0.0))

    if isinstance(instruction, bytes):
        instruction = instruction.decode("utf-8")
    else:
        instruction = str(instruction)

    num_steps = int(actions.shape[0])
    if min_action_norm > 0:
        norms = np.linalg.norm(actions[:, :6], axis=1)
        keep = (norms >= min_action_norm)
        keep[0] = True
        keep[-1] = True
        keep_indices = np.nonzero(keep)[0]
        if keep_indices.size < 2:
            return []
        states = states[keep_indices]
        images = images[keep_indices]
        actions = np.zeros_like(states)
        actions[:-1, :6] = states[1:, :6] - states[:-1, :6]
        actions[:-1, 6] = states[1:, 6]
        actions[-1, :6] = 0.0
        actions[-1, 6] = states[-1, 6]
        num_steps = int(actions.shape[0])
    steps = []
    for i in range(num_steps):
        steps.append(
            {
                "observation": {
                    "image": images[i],
                    "state": states[i],
                },
                "action": actions[i],
                "reward": 0.0,
                "discount": 1.0,
                "is_first": i == 0,
                "is_last": i == num_steps - 1,
                "is_terminal": i == num_steps - 1,
                "language_instruction": instruction,
            }
        )

    if steps:
        steps[-1]["reward"] = reward
        steps[-1]["discount"] = 0.0
    return steps


def _make_builder(dataset_name: str, version: str, paths: list):
    class_name = _snake_to_camel(dataset_name)
    step_features = tfds.features.FeaturesDict(
        {
            "observation": tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Tensor(shape=(224, 224, 3), dtype=tf.uint8),
                    "state": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                }
            ),
            "action": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            "reward": tfds.features.Scalar(dtype=tf.float32),
            "discount": tfds.features.Scalar(dtype=tf.float32),
            "is_first": tfds.features.Scalar(dtype=tf.bool),
            "is_last": tfds.features.Scalar(dtype=tf.bool),
            "is_terminal": tfds.features.Scalar(dtype=tf.bool),
            "language_instruction": tfds.features.Text(),
        }
    )
    features = tfds.features.FeaturesDict({"steps": tfds.features.Dataset(step_features)})

    def _info(self):
        return tfds.core.DatasetInfo(builder=self, features=features)

    def _split_generators(self, dl_manager):
        del dl_manager
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"paths": paths},
            )
        ]

    def _generate_examples(self, paths):
        for path in paths:
            steps = _load_steps(path, min_action_norm)
            if not steps:
                continue
            yield path.stem, {"steps": steps}

    return type(
        class_name,
        (tfds.core.GeneratorBasedBuilder,),
        {
            "VERSION": tfds.core.Version(version),
            "_info": _info,
            "_split_generators": _split_generators,
            "_generate_examples": _generate_examples,
        },
    )


# tsc
def _collect_datasets(input_root: Path, dataset_name: str, exclude_fail: bool) -> list:
    datasets = []
    if dataset_name:
        dataset_id = dataset_name.strip().replace(" ", "_")
        dataset_dir = input_root / dataset_name
        # tsc
        if not dataset_dir.is_dir():
            dataset_dir = input_root / dataset_name.replace("_", " ")
        if not dataset_dir.is_dir():
            dataset_dir = input_root
        paths = sorted(dataset_dir.glob("*.hdf5"))
        if exclude_fail:
            paths = [p for p in paths if "FAIL" not in p.name]
        if not paths:
            raise SystemExit("No HDF5 files found.")
        datasets.append((dataset_id, paths))
        return datasets

    subdirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if subdirs:
        for subdir in subdirs:
            dataset_id = subdir.name.strip().replace(" ", "_")
            paths = sorted(subdir.glob("*.hdf5"))
            if exclude_fail:
                paths = [p for p in paths if "FAIL" not in p.name]
            if paths:
                datasets.append((dataset_id, paths))
    else:
        dataset_id = input_root.name.strip().replace(" ", "_")
        paths = sorted(input_root.glob("*.hdf5"))
        if exclude_fail:
            paths = [p for p in paths if "FAIL" not in p.name]
        if paths:
            datasets.append((dataset_id, paths))

    if not datasets:
        raise SystemExit("No HDF5 files found.")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    # tsc
    parser.add_argument(
        "--input_dir",
        default="openvla/piper/data",
    )
    # tsc
    parser.add_argument(
        "--output_dir",
        default="openvla/piper/data_rlds",
    )
    # tsc
    parser.add_argument(
        "--dataset_name",
        default="",
    )
    parser.add_argument(
        "--version",
        default="1.0.0",
    )
    # tsc
    parser.add_argument(
        "--exclude_fail",
        action="store_true",
    )
    parser.add_argument(
        "--min_action_norm",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    # tsc
    input_root = Path(args.input_dir)
    datasets = _collect_datasets(input_root, args.dataset_name, args.exclude_fail)
    for dataset_name, paths in datasets:
        builder_cls = _make_builder(dataset_name, args.version, paths)
        builder = builder_cls(data_dir=args.output_dir)
        global min_action_norm
        min_action_norm = args.min_action_norm
        builder.download_and_prepare()
        print(builder.data_dir)


if __name__ == "__main__":
    main()
