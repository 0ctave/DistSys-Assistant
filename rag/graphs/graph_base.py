from abc import abstractmethod, ABC

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

class GraphBase(ABC):
    def __init__(self, assistant, graph_state, input_state, output_state):
        self._assistant = assistant

        self._agents = dict()
        self._load_agents()

        builder = StateGraph(graph_state, input=input_state, output=output_state)
        self._builder = builder
        self._load_nodes(builder)
        self._load_edges(builder)

        self._graph = builder.compile()

    @abstractmethod
    def _load_agents(self) -> None:
        pass

    @abstractmethod
    def _load_nodes(self, builder: StateGraph) -> None:
        pass

    @abstractmethod
    def _load_edges(self, builder: StateGraph) -> None:
        pass

    def get_graph(self) -> CompiledStateGraph:
        return self._graph
