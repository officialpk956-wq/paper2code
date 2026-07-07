import codecs
with codecs.open("temp_papers.py", "r", "utf-16") as f:
    lines = f.readlines()

confirm_upload_func = []
recording = False
for line in lines:
    if "@router.post(\"/papers/confirm-upload\")" in line:
        recording = True
    if recording:
        confirm_upload_func.append(line)
        if "\"message\": \"Upload confirmed. Poll poll_url every 3s for generated code.\"," in line:
            confirm_upload_func.append("    }\n")
            break

with open("backend/routers/papers_pipeline.py", "r", encoding="utf-8") as f:
    content = f.read()

target = "class ConfirmUploadRequest(BaseModel):\n    key: str\n    paper_name: str\n    visibility: str = \"public\"\n    terms_accepted: bool = False\n    file_size_bytes: int = 0\n"
content = content.replace(target, target + "\n" + "".join(confirm_upload_func))

with open("backend/routers/papers_pipeline.py", "w", encoding="utf-8") as f:
    f.write(content)
