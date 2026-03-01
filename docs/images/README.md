# Documentation Images

This directory contains images used in the project documentation.

## Required Images

- `pipeline_visualization.png` - Screenshot of the `transform_pipeline.describe()` output showing the pipeline structure with unicode symbols and proper formatting

To generate the pipeline visualization image:

1. Run the simple pipeline example:
```python
from apriori.ico.core.operator import operator

@operator()
def square(x: int) -> int:
    return x * x

@operator()
def add_ten(x: int) -> int:
    return x + 10

@operator()
def to_string(x: int) -> str:
    return f"Result: {x}"

transform_pipeline = square | add_ten | to_string
transform_pipeline.describe()
```

2. Take a screenshot of the console output
3. Save it as `pipeline_visualization.png` in this directory