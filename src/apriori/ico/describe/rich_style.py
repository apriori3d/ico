from enum import Enum


class DescribeStyle(Enum):
    """
    Rich console color scheme for ICO describe rendering.

    Defines consistent styling for different code elements:
    functions, classes, types, keywords, metadata, etc.
    """

    fn = "#A67F59"
    type = "#569CD6"
    class_ = "#0052CC"
    keyword = "#E12EE1"
    dimmed = "gray70"
    meta = "#4FC1FF"
    text = "#4EB169"
    string = "#DD1616"
    signature = "cyan"
    semantic_meta = "#8446F7 italic"
    group = "#AAAAAA"
    tree = "#AAAAAA"
