import box
import yaml

import warnings

from rag.utils.vector_store import LocalVectorStore

warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    with open('config.yml', 'r', encoding='utf8') as config_file:
        cfg = box.Box(yaml.safe_load(config_file))

    store = LocalVectorStore()

    store.ingest_fs(cfg.DATA_PATH)
    store.close()



