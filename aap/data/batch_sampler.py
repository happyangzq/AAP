import random

from torch.utils.data import Sampler


class ClassBalancedBatchSampler(Sampler):
    """Yield homogeneous class batches for the asymmetric objective."""

    def __init__(self, dataset, batch_size, world_size=1, rank=0, seed=42):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.indices_by_class = {0: [], 1: [], 2: []}

        for idx in range(len(dataset)):
            cls = dataset.cls_labels[idx]
            self.indices_by_class[cls].append(idx)

        if self.rank == 0:
            for class_label, indices in self.indices_by_class.items():
                print(f"Class {class_label}: {len(indices)} images")

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        all_batches = []
        for indices_for_class in self.indices_by_class.values():
            indices = list(indices_for_class)
            generator.shuffle(indices)
            for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                all_batches.append(indices[i : i + self.batch_size])

        generator.shuffle(all_batches)
        self.epoch += 1

        if self.world_size > 1:
            usable = len(all_batches) - len(all_batches) % self.world_size
            all_batches = all_batches[:usable][self.rank::self.world_size]

        return iter(all_batches)

    def __len__(self):
        batch_count = sum(
            len(indices) // self.batch_size
            for indices in self.indices_by_class.values()
        )
        if self.world_size > 1:
            return batch_count // self.world_size
        return batch_count
