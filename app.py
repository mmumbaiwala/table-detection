from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import io

# Model Imports
from transformers import DetrFeatureExtractor
feature_extractor = DetrFeatureExtractor()

from transformers import TableTransformerForObjectDetection
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")


app = FastAPI()

def process_image(file) -> Image.Image:
    image = Image.open(file)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    # Do any processing you want with the image here
    # For example, resizing or applying filters
    return image

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    processed_image = process_image(file.file)

    output = io.BytesIO()
    processed_image.save(output, format="JPEG")
    output.seek(0)

    return StreamingResponse(io.BytesIO(output.read()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # uvicorn.run(app, host="0.0.0.0", port=8000)
