from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from typing import List
import shutil
import os
import config
from task_router import TaskRouter

app = FastAPI()
router = TaskRouter()

@app.post("/screen_cvs")
async def screen_cvs(jd: str = Form(...), files: List[UploadFile] = File(...)):
    safe_ids = []
    for file in files:
        # 1. Clean the ID first
        cid_safe = router.emb.clean_candidate_id(file.filename)
        
        # 2. Read the file content
        content = await file.read()
        
        # 3. Embed using the content and safe ID
        router.emb.embed_cv(content, cid_safe)
        safe_ids.append(cid_safe)

    # Now all DBs are ready and named exactly as items in safe_ids
    batch_result = await router.batch_screen(jd, safe_ids)
    return batch_result

@app.post("/firing_analysis")
async def firing_analysis(candidate_id: str = Form(...), query: str = Form(...)):
    # Standardize ID before processing
    safe_id = router.emb.clean_candidate_id(candidate_id)
    result = await router.execute("firing_analysis", {
        "candidate_id": safe_id,
        "query": query
    })
    return result

@app.post("/chat")
async def chat_with_ai(candidate_id: str = Form(...), query: str = Form(...), history: str = Form(default="")):
    safe_id = router.emb.clean_candidate_id(candidate_id)
    response = await router.execute("general_chat", {
        "candidate_id": safe_id,
        "query": query,
        "history": history
    })
    return {"response": response}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)