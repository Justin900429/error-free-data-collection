from io import BytesIO
import os
import warnings

warnings.filterwarnings("ignore")

from fastapi import Response, BackgroundTasks
import pynecone as pc

from web.utils import BaseState
from web.clip import index
from web.moco import moco
from web.imgbind import bind
from pcconfig import config

import dotenv

dotenv.load_dotenv()

filename = f"{config.app_name}/{config.app_name}.py"


app = pc.App(state=BaseState)
app.add_page(index)
app.add_page(moco)
app.add_page(bind)


@app.api.get("/download/{filename}")
async def file_download(filename: str, background_tasks: BackgroundTasks):
    buffer = BytesIO()

    local_filename = "zipfiles/" + filename + ".zip"

    # Open a zipfile and write into the buffer
    with open(local_filename, "rb") as zip_file:
        buffer.write(zip_file.read())
    buffer.seek(0)

    background_tasks.add_task(buffer.close)
    background_tasks.add_task(os.remove, path=local_filename)
    headers = {"Content-Disposition": f'attachment; filename="{filename}.zip"'}
    return Response(buffer.getvalue(), headers=headers, media_type="application/zip")


app.compile()
