from langchain.tools import tool
from google import genai
import os, json, requests, tempfile

def _download(url: str, name: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    suffix = "." + name.split(".")[-1]
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(r.content)
    return path

@tool
def multimodal_document_analyzer_tool(attachments: list, focus: str = "general"):
    """
    Uploads each document to Gemini 1.5 Pro File API and extracts
    structured JSON entities.
    """

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    documents = []

    for att in attachments:
        url = att["attachment_url"]
        name = att["name"]

        local_path = _download(url, name)

        file_obj = client.files.upload(file_path=local_path)

        structured = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[
                {"parts": [{
                    "text": f"""
Extract structured JSON from this file.
Focus: {focus}
Return fields:
model_numbers, order_numbers, part_numbers,
product_names, quantities, dates, key_entities
                    """
                }]},
                {"parts": [{
                    "file_data": {"file_uri": file_obj.uri}
                }]}
            ]
        )

        documents.append({
            "filename": name,
            "extracted_info": json.loads(structured.text)
        })

        os.remove(local_path)

    return {
        "success": True,
        "documents": documents,
        "count": len(documents)
    }
