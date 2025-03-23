from rag.graphs.command_graph import CommandGraph
from rag.graphs.context_graph import ContextGraph
from rag.graphs.decision_graph import DecisionGraph
from rag.utils.command_executor import CommandExecutor
from rag.utils.vector_store import LocalVectorStore
import getpass


class Assistant:
    def __init__(self):
        password = getpass.getpass("Enter your sudo password: ")

        self._command_executor = CommandExecutor(password)

        self._vector_store = LocalVectorStore()
        self._vector_store.load_index("FolderDocs")

        self._command_graph = CommandGraph(self)
        self._context_graph = ContextGraph(self)
        self._decision_graph = DecisionGraph(self)

    def get_graph(self):
        return self._decision_graph.get_graph()

    def get_path(self):
        return self._command_graph.get_path()

    def get_command_graph(self):
        return self._command_graph.get_graph()

    def get_context_graph(self):
        return self._context_graph.get_graph()

    def get_vector_store(self):
        return self._vector_store

    def get_command_executor(self):
        return self._command_executor

    def run(self, args):
        return self.get_graph().stream(args)

    def close(self):
        self._command_executor.close()
        self._vector_store.close()