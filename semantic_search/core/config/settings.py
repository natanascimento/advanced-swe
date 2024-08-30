from os.path import abspath, dirname, join


class AppSettings:
    APP_NAME = "Semantic search"
    ROOT_PATH = dirname(dirname(dirname(dirname(abspath(__file__)))))
    SRC_PATH = dirname(dirname(dirname(abspath(__file__))))
    DATALAKE_PATH = join(ROOT_PATH, "datalake")


class Settings:
    app: AppSettings = AppSettings()


settings = Settings()
