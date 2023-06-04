import os
import pynecone as pc
import dotenv

dotenv.load_dotenv()


class WebConfig(pc.Config):
    pass


config = WebConfig(
    app_name="web",
    deploy_url=f"http://{os.getenv('IP_ADDRESS')}:3000",
    api_url=f"http://{os.getenv('IP_ADDRESS')}:8000",
    db_url="sqlite:///pynecone.db",
)
