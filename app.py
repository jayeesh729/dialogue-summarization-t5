from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration , T5Tokenizer 
import re
import torch
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize fast api app

app = FastAPI(title="Text summarizer App", description="Text Summarization using T5", version="1.0")

# model 
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_models")
tokenizer= T5Tokenizer.from_pretrained("./saved_summary_models")

# device 
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


templates = Jinja2Templates(directory="templates")

# input schema

class DialogueInput(BaseModel):
    dialogue : str

# Preprocessing Text
def clean_data(text):
    text = re.sub(r"\r\n" ," ", text)
    text = re.sub(r"\rs+", " ", text)
    text = re.sub(r"<.*?>"," ", text)
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue):
    dialogue = clean_data(dialogue)
    #tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    #generate the summary
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    ) 
    #token ids convert to summary
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

# API endpoints

# get /
@app.get("/", response_class=HTMLResponse)
async def home(request : Request):
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={"request": request}
)

# post /Summarize/
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}
