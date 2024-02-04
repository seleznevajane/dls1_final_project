from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from processing import process_image


os.mkdir("static")
os.mkdir("uploads")

app = FastAPI()
# Serve HTML file for file upload
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Provide HTML form for GET request
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request, "start_page.html")


# Upload endpoint
@app.post("/example_park")
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Process the image with MMDetection
    result = process_image.detect_animals(file_path, show=False, no_save_vis=True, no_save_pred=True)

    # Add detection results to the image and create a caption
    output_path = os.path.join('static', file.filename)
    text_data, coords_data = process_image.process_detection_result(
                        image_path=file_path, 
                        detection_result=result['predictions'][0], 
                        output_image_path=output_path)
    result_url = request.url_for("static", path=file.filename)
    os.remove(file_path)
    return templates.TemplateResponse(request, "example_park.html",
                                      {"image_url": result_url, "text_data": text_data, "coords_data": coords_data})


# Provide HTML form for GET request
@app.get("/example_park", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request, "example_park.html")
