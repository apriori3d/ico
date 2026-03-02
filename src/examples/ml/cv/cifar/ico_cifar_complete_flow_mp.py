"""CIFAR-10 Multiprocessing Pipeline - ICO Framework with Parallel Data Loading

This comprehensive example demonstrates advanced multiprocessing capabilities of ICO Framework for
CIFAR-10 image classification, showcasing ICO as a superior PyTorch DataLoader replacement.

🚀 **Key Multiprocessing Features:**
- **IcoAsyncStream**: Parallel processing across multiple worker processes
- **MPAgent**: Distributed data loading with independent augmentation pipelines
- **WorkerFlowFactory**: Process-isolated pipelines with individual data access
- **Shared Memory Tensors**: Zero-copy inter-process communication
- **Runtime Monitoring**: Real-time progress tracking across all workers

🏗️ **Architecture Highlights:**
- Each MPAgent worker runs in separate OS process (no Python GIL limitation)
- Independent dataset item fetching across workers (true concurrent I/O)
- Process-isolated augmentation pipelines with unique random states
- Efficient shared memory tensor transfer (eliminates serialization costs)
- Automatic work distribution and result collection via IcoAsyncStream


"""

import random
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import final

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.datasets import CIFAR10  # pyright: ignore[reportMissingTypeStubs]
from torchvision.models import resnet18  # pyright: ignore[reportMissingTypeStubs]
from torchvision.transforms import ToTensor  # pyright: ignore[reportMissingTypeStubs]
from torchvision.transforms import (  # pyright: ignore[reportMissingTypeStubs]
    functional as F,
)

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.batcher import IcoBatcher
from apriori.ico.core.context_operator import IcoContextOperator
from apriori.ico.core.context_pipeline import IcoContextPipeline
from apriori.ico.core.epoch import IcoEpoch
from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.runtime.progress import IcoProgress
from apriori.ico.core.runtime.runtime import IcoRuntime
from apriori.ico.core.source import IcoSource
from apriori.ico.runtime.agent.mp.mp_agent import MPAgent
from apriori.ico.tools.progress.rich_progress_tool import RichProgressTool

# ─────────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES & DATASET CLASSES
# ─────────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class CifarItem:
    """Individual CIFAR-10 data item containing image and label tensors."""

    image: Tensor  # (3, 32, 32)
    label: Tensor  # (,)


class CifarDataset:
    images: Tensor  # (N, 3, 32, 32)
    labels: Tensor  # (N,)

    def __init__(self, root: str = "data/"):
        dataset = CIFAR10(root=root, download=True, transform=ToTensor())

        # Preload all data into memory as tensors
        self.images = torch.stack(
            [dataset[i][0] for i in range(len(dataset))]
        )  # (N, 3, 32, 32)
        self.labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])  # (N,)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> CifarItem:
        return CifarItem(image=self.images[index], label=self.labels[index])

    def __iter__(self) -> Iterator[CifarItem]:
        for i in range(len(self)):
            yield self[i]

    def share_memory_(self):
        """Enable shared memory for multiprocessing - reduces memory overhead across workers."""
        self.images.share_memory_()
        self.labels.share_memory_()


# ─────────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION CLASSES
# ─────────────────────────────────────────────────────────────────────────────────


class CifarTransform(ABC):
    """Base class for CIFAR-10 image transformations with probability control."""

    p: float
    factor: float

    def __init__(self, p: float = 1.0, factor: float = 1.0):
        self.p = p
        self.factor = factor

    @abstractmethod
    def _image_transform(self, image: Tensor, factor: float) -> Tensor:
        raise NotImplementedError

    def __call__(self, item: CifarItem) -> CifarItem:
        if torch.rand(1) >= self.p:
            return item

        return CifarItem(
            image=self._image_transform(item.image, self.factor), label=item.label
        )


class HorizontalFlip(CifarTransform):
    """Horizontal flip augmentation for CIFAR-10 images."""

    def _image_transform(self, image: Tensor, factor: float) -> Tensor:
        return F.hflip(image)


class VerticalFlip(CifarTransform):
    """Vertical flip augmentation for CIFAR-10 images."""

    def _image_transform(self, image: Tensor, factor: float) -> Tensor:
        return F.vflip(image)


class AdjustBrightness(CifarTransform):
    """Brightness adjustment augmentation for CIFAR-10 images."""

    def _image_transform(self, image: Tensor, factor: float) -> Tensor:
        return F.adjust_brightness(image, factor)


class AdjustContrast(CifarTransform):
    """Contrast adjustment augmentation for CIFAR-10 images."""

    def _image_transform(self, image: Tensor, factor: float) -> Tensor:
        return F.adjust_contrast(image, factor)


# ─────────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING & MULTIPROCESSING PIPELINE FACTORIES
# ─────────────────────────────────────────────────────────────────────────────────


@final
class CifarBatch:
    """Batch container for CIFAR-10 data with shared memory support."""

    __slots__ = ("images", "labels")

    images: Tensor  # (B, 3, 32, 32)
    labels: Tensor  # (B,)

    def __init__(self, items: Iterator[CifarItem]):
        items_list = list(items)
        self.images = torch.stack([item.image for item in items_list])  # (B, 3, 32, 32)
        self.labels = torch.tensor([item.label for item in items_list])  # (B,)

    def share_memory_(self):
        self.images.share_memory_()
        self.labels.share_memory_()


@operator()
def collate(items: Iterator[CifarItem]) -> CifarBatch:
    """
    Collate individual items into a batch optimized for multiprocessing.

    🧠 **Memory Optimization:**
    - Creates shared memory tensors accessible across processes
    - Eliminates need for data serialization/deserialization
    - Enables zero-copy tensor passing between worker processes

    🏗️ **Implementation Details:**
    This function replaces the traditional PyTorch DataLoader's collate_fn
    with multiprocessing-aware batching that maintains tensor sharing.
    """
    batch = CifarBatch(items)
    # CRITICAL: Share memory to enable zero-copy inter-process communication
    # Without this, tensors would be copied/pickled between processes (expensive!)
    batch.share_memory_()
    return batch


class WorkerFlowFactory:
    """
    Factory for creating multiprocessing workers with independent data pipelines.

    🔄 **Multiprocessing Design:**
    Each MPAgent spawns separate process running this factory's pipeline:
    - Independent data loading (no GIL contention)
    - Isolated augmentation strategies per worker
    - Progress tracking across process boundaries

    ⚡ **Performance Features:**
    - Configurable processing delay (simulates I/O bound operations)
    - Shared memory tensors for efficient inter-process communication
    - Batch-level processing reduces process boundaries crossing

    🛠️ **Technical Requirements:**
    - Must be at module level for multiprocessing pickling
    - Each worker gets independent random state for augmentations
    - Dataset shared across workers (read-only, memory efficient)
    """

    name: str | None = None
    batch_size: int

    def __init__(
        self,
        dataset: CifarDataset,
        batch_size: int,
        name: str | None = None,
        delay: float | None = 0.1,
    ):
        self.name = name
        self.batch_size = batch_size
        self.dataset = dataset
        self.delay = delay

    def __call__(self) -> IcoOperator[Iterator[int], CifarBatch]:
        worker_progress = IcoProgress[int](total=self.batch_size, name="Worker")

        @operator()
        def fetch_item(idx: int) -> CifarItem:
            """
            Fetch individual CIFAR item by index (simulating disk I/O).

            🚀 **Multiprocessing Advantage:**
            Multiple workers can fetch different indices concurrently without
            Python GIL contention - this is where ICO outperforms traditional
            PyTorch DataLoader which is single-threaded for dataset access.
            """
            if self.delay:
                time.sleep(
                    self.delay
                )  # Simulate realistic I/O latency (disk reads, network requests)

            return self.dataset[idx]

        # ─────────────────────────────────────────────────────────────────────────────────
        # 🎨 PARALLEL AUGMENTATION PIPELINE
        # ─────────────────────────────────────────────────────────────────────────────────
        # Each worker process applies same transformations with independent random state
        # Result: Diverse augmented data across workers (different random seeds per worker)
        # Performance: No serialization overhead - transforms applied in worker memory space
        item_aug_flow = IcoPipeline(
            IcoOperator(HorizontalFlip()),  # Independent random flip per worker
            IcoOperator(VerticalFlip()),  # Parallel processing with other workers
            IcoOperator(AdjustBrightness(factor=0.2)),  # Isolated worker RNG state
            IcoOperator(AdjustContrast(factor=0.2)),  # Efficient memory usage
        )

        # ─────────────────────────────────────────────────────────────────────────────────
        # 🔄 MULTIPROCESSING WORKER PIPELINE COMPOSITION
        # ─────────────────────────────────────────────────────────────────────────────────
        # This pipeline runs entirely within each worker process:
        # 1. worker_progress: Tracks batch progress across process boundaries
        # 2. fetch_item: Parallel data loading (no GIL, concurrent disk I/O)
        # 3. item_aug_flow: Independent augmentation per worker process
        # 4. .stream(): Convert to async stream for batch processing
        # 5. collate: Create shared memory tensors for zero-copy transfer
        worker_flow = (worker_progress | fetch_item | item_aug_flow).stream() | collate
        worker_flow.name = "Worker flow"

        return worker_flow


# ─────────────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK ARCHITECTURE & SETUP
# ─────────────────────────────────────────────────────────────────────────────────


def create_cifar10_resnet18(num_classes: int = 10):
    """Create ResNet-18 model customized for CIFAR-10 dataset."""
    model = resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]
    model.fc = nn.Linear(512, num_classes)
    return model


# ─────────────────────────────────────────────────────────────────────────────────
# TRAINING CONTEXT & PIPELINE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class CifarTrainContext:
    """Training context containing model, optimizer, and training state."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss()
    iter_num: int = 0
    total_loss: float = 0.0


@operator()
def create_train_context(_: None) -> CifarTrainContext:
    """Initialize training context with model and optimizer."""
    model = create_cifar10_resnet18()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return CifarTrainContext(model=model, optimizer=optimizer)


@operator()
def finalize_train_context(context: CifarTrainContext) -> None:
    """Finalize training and report completion."""
    print("Training complete. Final model ready for test or deployment.")


@operator()
def start_train(context: CifarTrainContext) -> CifarTrainContext:
    """Set model to training mode."""
    context.model.train()
    return context


def train_step(batch: CifarBatch, context: CifarTrainContext) -> CifarTrainContext:
    """Execute single training step with forward/backward pass."""
    context.optimizer.zero_grad()

    outputs = context.model(batch.images)
    loss = context.loss_fn(outputs, batch.labels)
    loss.backward()
    context.optimizer.step()

    context.total_loss = loss.item()
    context.iter_num += 1
    return context


def logging_step(context: CifarTrainContext) -> CifarTrainContext:
    """Log training progress every 10 iterations."""
    if context.iter_num % 10 == 0:
        print(f"Iteration {context.iter_num}, Loss: {context.total_loss:.4f}")
    return context


def save_checkpoint_step(context: CifarTrainContext) -> CifarTrainContext:
    """Save model checkpoint every 100 iterations."""
    if context.iter_num % 100 == 0:
        print(f"Checkpointing model at iteration {context.iter_num}")
    return context


# ─────────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATION & VALIDATION ROUTINES
# ─────────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class CifarEvalContext:
    """Evaluation context for tracking validation accuracy and metrics."""

    train_context: CifarTrainContext
    accuracy: float = 0.0
    total_samples: int = 0


def calculate_accuracy(item: CifarItem, context: CifarEvalContext) -> CifarEvalContext:
    """Calculate accuracy for single validation item."""
    with torch.no_grad():
        outputs = context.train_context.model(item.image.unsqueeze(0))  # (1, 10)

        predicted = torch.argmax(outputs, dim=1)
        correct = predicted.item() == item.label
        context.total_samples += 1
        context.accuracy += int(correct)

    return context


@operator()
def start_eval(context: CifarTrainContext) -> CifarEvalContext:
    """Initialize evaluation context and set model to eval mode."""
    context.model.eval()
    return CifarEvalContext(train_context=context)


@operator()
def log_accuracy(context: CifarEvalContext) -> CifarEvalContext:
    """Calculate and log final validation accuracy."""
    accuracy = (
        context.accuracy / context.total_samples if context.total_samples > 0 else 0.0
    )
    print(
        f"Validation Accuracy: {accuracy * 100:.2f}% ({context.accuracy}/{context.total_samples})"
    )
    return context


@operator()
def end_eval(context: CifarEvalContext) -> CifarTrainContext:
    """End evaluation phase and return to training context."""
    return context.train_context


# ─────────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION PIPELINE & MULTIPROCESSING COORDINATION
# ─────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────────────────────
    # 🔧 TRAINING CONFIGURATION & HYPERPARAMETERS
    # ─────────────────────────────────────────────────────────────────────────────────

    # make it 2% of a real dataset size for speed up in demo.
    num_epoch = 5
    train_split_ratio = 0.02
    batch_size = 8
    num_workers = 4
    val_split_ratio = 0.01
    worker_delay = 0.1  # Simulate slow data loading in workers (e.g., from disk)

    # ─────────────────────────────────────────────────────────────────────────────────
    # 🗄️ DATASET PREPARATION & SHARED MEMORY SETUP
    # ─────────────────────────────────────────────────────────────────────────────────

    dataset = CifarDataset()
    dataset.share_memory_()  # Share dataset tensors in memory for multiprocessing

    # ─────────────────────────────────────────────────────────────────────────────────
    # 🔬 WORKER FLOW DEMONSTRATION & PIPELINE STRUCTURE
    # ─────────────────────────────────────────────────────────────────────────────────

    worker_flow = WorkerFlowFactory(
        dataset=dataset,
        batch_size=batch_size,
        delay=worker_delay,
    )()
    print("🔧 Worker Flow Structure (runs in each separate process):")
    worker_flow.describe()

    # ──────── Create train data flow ────────

    # Use indices as input items to workers for efficient data loading and sharing
    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))
    random.shuffle(all_indices)

    # Split indices into train and validation sets
    train_split = int(dataset_size * train_split_ratio)

    # ─────────────────────────────────────────────────────────────────────────────────
    # 🚀 PARALLEL DATA LOADING PIPELINE CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────────────

    # Create data source
    train_source = IcoSource[int](
        lambda: all_indices[:train_split],
        name="CIFAR10 train indices",
    )

    # Report progress of the train data flow
    track_train_progress = IcoProgress[int](
        total=train_split, name="Train Epoch Progress"
    )

    # Group indices into batches for workers to process.
    batcher = IcoBatcher[int](batch_size=batch_size)

    # Each worker will fetch the actual data items from the dataset using the indices, apply augmentations, and collate into batches.
    # Dataset is shared in memory, so workers can access it efficiently without copying.
    # CifarBatch also will be shared in memory for efficient transfer between processes.
    # Node: batch_size needs only for progress tracking.

    # ─────────────────────────────────────────────────────────────────────────────────
    # 🏭 MULTIPROCESSING WORKER FACTORY & AGENT POOL CREATION
    # ─────────────────────────────────────────────────────────────────────────────────
    def create_mp_agent() -> IcoOperator[Iterator[int], CifarBatch]:
        """
        Factory function for creating MPAgent workers with independent data pipelines.

        🔄 **Multiprocessing Benefits:**
        - Each MPAgent runs in separate OS process (no Python GIL limitation)
        - Independent data loading: workers can fetch different dataset items concurrently
        - Isolated augmentation: each worker has own random state for diverse transforms
        - Shared memory tensors: zero-copy data transfer between worker and main process

        🆚 **vs PyTorch DataLoader:**
        - PyTorch: Single-threaded dataset access + multiprocessing for transforms only
        - ICO: Parallel dataset access + parallel transforms + efficient memory sharing
        """
        worker_flow_factory = WorkerFlowFactory(
            dataset=dataset,
            batch_size=batch_size,
            delay=worker_delay,  # Simulates latency for demo purposes (
        )
        return MPAgent(worker_flow_factory)

    # ─────────────────────────────────────────────────────────────────────────────────
    # 🏊 PARALLEL WORKER POOL - CORE MULTIPROCESSING ARCHITECTURE
    # ─────────────────────────────────────────────────────────────────────────────────
    # IcoAsyncStream coordinates multiple MPAgent processes:
    # - Distributes work (batch indices) across available workers
    # - Collects results from parallel workers as they complete
    # - Maintains load balancing between processes automatically
    workers_pool = IcoAsyncStream(create_mp_agent, pool_size=num_workers)

    # ─────────────────────────────────────────────────────────────────────────────────
    # 🚀 MAIN TRAINING PIPELINE WITH MULTIPROCESSING DATA LOADING
    # ─────────────────────────────────────────────────────────────────────────────────
    # This pipeline demonstrates ICO's advantage as PyTorch DataLoader replacement:
    # 1. train_source: Generate batch indices (lightweight, single-threaded)
    # 2. epoch_progress: Track overall epoch progress
    # 3. batcher: Group indices into batch-size chunks
    # 4. workers_pool: **PARALLEL DATA LOADING** - Multiple processes simultaneously:
    #    - Fetch different dataset items by indices (no GIL contention)
    #    - Apply independent data augmentations per worker
    #    - Create shared memory batches for zero-copy transfer
    train_flow = train_source | track_train_progress.stream() | batcher | workers_pool
    train_flow.name = "Train Flow"

    print("Train flow structure with MPAgent pool (workers run in parallel):")
    train_flow.describe()

    # ──────── Create model train flow ────────
    train_pipeline = IcoContextPipeline(train_step, logging_step, save_checkpoint_step)
    train_pipeline.name = "Training Pipeline"

    train_epoch = IcoEpoch(
        source=train_flow,
        context_operator=train_pipeline,
    )
    train_epoch.name = "Training Epoch"

    train_flow = start_train | train_epoch
    train_flow.name = "Train Flow"

    print("\nComplete train flow structure with MPAgent pool and training pipeline:")
    train_flow.describe()

    # ──────── Validation flow -────────

    # Validation data flow is mutch simpler, just fetch items by indices and run through the model for evaluation.
    val_split = int(dataset_size * val_split_ratio)
    val_source = IcoSource[int](
        lambda: all_indices[-val_split:], name="CIFAR10 evaluation indices"
    )

    # Report progress of the validation data flow
    track_val_progress = IcoProgress[int](
        total=val_split, name="Validation Epoch Progress"
    )

    @operator()
    def fetch_item(index: int) -> CifarItem:
        return dataset[index]

    val_data_flow = val_source | (track_val_progress | fetch_item).stream()
    val_data_flow.name = "Validation Flow"

    val_epoch = IcoEpoch(
        source=val_data_flow,
        context_operator=IcoContextOperator(calculate_accuracy),
        name="Validation Epoch",
    )

    val_flow = start_eval | val_epoch | log_accuracy | end_eval
    val_flow.name = "Validation Flow"
    val_flow.describe()

    # ──────── Complete train flow structure ────────

    # Add total progress to track iteration over epochs
    total_progress = IcoProgress[CifarTrainContext](
        total=num_epoch, name="Total Progress"
    )

    epoch_flow = total_progress | train_flow | val_flow
    epoch_flow.name = "Train and Validation Epoch Flow"

    train_process = IcoProcess(epoch_flow, num_iterations=num_epoch)
    train_process.name = "CIFAR-10 Training Process"

    # Create autonomous flow to run in runtime without input and outputs
    cifar_flow = create_train_context | train_process | finalize_train_context
    cifar_flow.name = "Complete CIFAR-10 Training and Validation Flow"
    cifar_flow.describe()

    # ─────────────────────────────────────────────────────────────────────────────────
    # 🎬 RUNTIME EXECUTION WITH MULTIPROCESSING COORDINATION
    # ─────────────────────────────────────────────────────────────────────────────────
    # ICO Runtime manages the complete multiprocessing infrastructure:
    # - Spawns and coordinates MPAgent worker processes
    # - Synchronizes progress tracking across process boundaries
    # - Handles process cleanup and resource management
    # - Provides unified monitoring via RichProgressTool
    #
    # 📊 **Performance Monitoring:**
    # RichProgressTool shows real-time progress from all parallel workers,
    # demonstrating concurrent data loading and processing capabilities.

    runtime = IcoRuntime(cifar_flow, tools=[RichProgressTool()])
    runtime.activate().run().deactivate()
