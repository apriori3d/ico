from examples.ml.skrub.base import SKChain, SKPipeline
from examples.ml.skrub.ops import SKStringEncoder
from ico.describe import PlanRendererDefaultOptions

PlanRendererDefaultOptions.renderers_paths.insert(
    0, "examples.ml.skrub.describe.plan.renderers"
)
PlanRendererDefaultOptions.flatten_node_type.update(
    [SKChain, SKPipeline, SKStringEncoder]
)
