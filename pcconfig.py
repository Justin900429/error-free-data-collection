import pynecone as pc


class WebConfig(pc.Config):
    pass


config = WebConfig(
    app_name="web",
    deploy_url="http://192.168.1.114:3000",
    api_url="http://192.168.1.114:8000",
    db_url="sqlite:///pynecone.db",
)
