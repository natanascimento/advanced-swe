from os.path import abspath, dirname, join
from os import environ


class AppSettings:
    APP_NAME = "Semantic search"
    ROOT_PATH = dirname(dirname(dirname(dirname(abspath(__file__)))))
    SRC_PATH = dirname(dirname(dirname(abspath(__file__))))
    DATALAKE_PATH = join(ROOT_PATH, "datalake")


class MilvusSettings:
    host: str = environ.get("milvus_host")
    port: str = environ.get("milvus_port")
    db: str = environ.get("milvus_db")


class Settings:
    app: AppSettings = AppSettings()
    milvus: MilvusSettings = MilvusSettings()


settings = Settings()
