from ...frontend.entry import ScatteringEntry

class ScatteringEntryGraph(ScatteringEntry):
    def __init__(self, *args, **kwargs):
        super().__init__(name="graph", class_name="scatteringgraph", *args, **kwargs)

__all__ = ['ScatteringEntryGraph']
