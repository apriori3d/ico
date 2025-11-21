from __future__ import annotations

from typing import Any

from apriori.ico.core.meta.ico_form import infer_ico_form
from apriori.ico.core.types import IcoOperator


def is_sink(node: IcoOperator[Any, Any]) -> bool:
    """
    Identify whether a node is a Sink (final consumer).

    Criteria:
      • ICOForm matches I → ()
    """
    form = infer_ico_form(node)
    return form and form.i != "()" and form.o == "()"


def is_source(node: IcoOperator[Any, Any]) -> bool:
    """
    Identify whether a node is a Source (root emitter).

    Criteria:
      • ICOForm matches () → O
    """
    form = infer_ico_form(node)
    return form and form.i == "()" and form.o != "()"
