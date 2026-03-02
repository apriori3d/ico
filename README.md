# ICO Framework
<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Type Safety](https://img.shields.io/badge/Type_Safety-Guaranteed-green?style=for-the-badge)
![Rich Integration](https://img.shields.io/badge/Rich_Console-Integrated-purple?style=for-the-badge)

*Transform complex ML code into elegant, type-safe and self-describing flows*

</div>

ICO formalizes a common ML pattern: take **Input**, apply to **Context**, produce **Output**.
It provides an elegant, type-safe and fully-transparent framework for ML engineers and researchers.

## 🚀 Quick Demo

```python
from apriori.ico.core.process import IcoProcess

# Fibonacci as an iterative process
def fib_step(state: tuple[int, int]) -> tuple[int, int]:
    return (state[1], state[0] + state[1])

fib_process = IcoProcess(fib_step, num_iterations=8)
result = fib_process((0, 1))
print(result)  # (21, 34) - 8th Fibonacci number

```

## 🔥 Key Features

### 🔍 **Introspection and Rich Visualization**

ICO provides beautiful console visualization with a fully-defined signature for any operator:
```python
fib_process.describe()
```

<img src="docs/images/fib_describe.jpg" width="800" alt="Fibonacci process visualization">

### 🎯 **Declarative Elegance**
```python
# Complex workflows become simple
ml_pipeline = (
    load_train_data
    | augment_pipeline
    | train_epoch
    | save_checkpoint
)
```


### 🛡️ **Type Safety Everywhere**
Static type checking with mypy, Pylance, and other type checkers.
<img src="docs/images/type_checking.jpg" width="800" alt="Type checking">


### ⚡ **Distributed by Design**
```python
# Multiprocessing with zero configuration
workers = IcoAsyncStream(
    lambda: MPAgent(heavy_computation),
    pool_size=cpu_count()
)

# Automatic work distribution and result collection
distributed_flow = data_source | workers | train
```

### 📊 **Built-in Progress Tracking**
```python
# Rich progress bars and metrics
progress = IcoProgress(name="Overall progress", total=epochs)
pipeline = source | progress | processing | train

# Real-time console updates with ETA, speed, and more
```
<img src="docs/images/progress.jpg" width="600" alt="Fibonacci process visualization">


## 📚 Use Cases

**Perfect for:**
- 🧠 **ML Training Pipelines** — Data loading, augmentation and distributed training
- 📊 **ETL Workflows** — Extract, transform and load with comprehensive monitoring
- 🔄 **Stream Processing** — Real-time data processing with intelligent backpressure
- 🧪 **Research Experiments** — Reproducible and monitorable scientific computing
- 📈 **Data Analytics** — Complex data transformations with rich visualization

## 🎯 Why ICO?

| Feature | ICO | Others |
|---------|-----|--------|
| Type Safety | ✅ Static type checking (mypy/Pylance) | ❌ Runtime errors |
| Visualization | ✅ Rich console integration | ❌ External tools needed |
| Distribution | ✅ Built-in multiprocessing | ❌ Manual setup |
| Composability | ✅ Pipe syntax `\|` | ❌ Complex APIs |
| Monitoring | ✅ Real-time state tracking | ❌ Limited introspection |

## 🚀 Getting Started

### 📖 Examples

#### 🎯 **ICO Basics**
- [Basic introduction to ICO approach](src/examples/ico_basics.ipynb): Main building blocks and core concepts
- [ICO Runtime introduction](src/examples/ico_runtime_basics.ipynb): Progress monitoring, printing and runtime architecture

#### 🔄 **Multiprocessing**
- [Multi-processing example](src/examples/mp_basic.py): Basic example of distributed computational flows
- [Parallel Multi-processing Pool example](src/examples/mp_basic_pool.py): Distributed compute flows with parallel worker pools

#### 🧠 **Machine Learning**
- [Linear Regression](src/examples/ml/linear_regression.ipynb): ICO-based ML pipeline development
- [CIFAR-10 Classification with validation](src/examples/ml/cv/cifar/complete_flow.ipynb): Complete CV pipeline replacing PyTorch DataLoader
- [CIFAR-10 Classification with worker pools](src/examples/ml/cv/cifar/complete_flow_mp.py): Complete CV pipeline with parallel data processing workers




## 📈 Future Development

### 🔮 **Planned Features**

- **📊 ICO Profiler**: Comprehensive performance analysis toolkit
  - Memory usage tracking across distributed workers
  - Bottleneck detection in complex pipelines
  - Time profiling with operator-level granularity
  - Export reports for optimization insights

- **😎 Stateful Runtime**: Advanced state management for production scenarios
  - Pause/resume operations for long-running training jobs
  - Checkpoint recovery after system failures
  - Migration between different compute environments
  - Perfect for cloud-native distributed training

- **🌐 ICO Live Board**: Real-time monitoring dashboard
  - Web-based interface for pipeline visualization
  - Live progress tracking across multiple experiments
  - Resource utilization monitoring (CPU, GPU, memory)
  - Integration with Jupyter notebooks and MLflow

### 🚀 **Community-Driven Roadmap**
We're building based on real ML engineer needs. Have ideas? [Join the discussions](https://github.com/apriori3d/ico/discussions) and help shape ICO's future!

Start experimenting and create your own innovative ML pipelines! 🎯

## 🤝 Contributing

We welcome contributions! Contribution guidelines are coming soon.

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

---

<div align="center">

**Happy Coding with ICO! ☺️**

[Documentation](docs/) • [Examples](examples/) • [Discussions](https://github.com/apriori3d/ico/discussions)

</div>
