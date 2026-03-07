from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import sys
sys.path.append('../')


DEBUG = False
MODEL_NAME = "ProsusAI/finbert"

#initialise finbert model (force apple silicon processor for Macbook)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
model.to("mps")

#Function to sentiment score a string, scores range from -1 (most negative) to 1 (most positive), returns float
def score_text(text: str) -> float:

    #set up parameters to take text as input return tensors and truncate input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    #run the model with the parameters and calculate the propabilities from the logits
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    #print score mapping
    if DEBUG:
        print(f"Positive Score: {probs[0, 0]}")
        print(f"Negative Score: {probs[0, 1]}")
        print(f"Neutral Score: {probs[0, 2]}")

    # calculate overall score (positive - negative)
    score = (
        probs[0, 0]   # positive score
        - probs[0, 1] # negative negative score
    )

    return float(score)

#Function to sentiment score a batch of strings
def score_batch(texts: list[str]) -> list[float]:
    
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    inputs = {k: v.to("mps") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    # calculate overall score (positive - negative) for each text
    scores = (probs[:, 0] - probs[:, 1]).tolist()
    return scores

#Function to score a pandas Series/column in batches
def score_dataframe(texts, batch_size: int = 32) -> list[float]:
    texts_list = texts.tolist()
    all_scores = []
    for i in range(0, len(texts_list), batch_size):
        batch = texts_list[i:i + batch_size]
        all_scores.extend(score_batch(batch))
    return all_scores

#debug loop for model testing
if DEBUG:
    while True:
        print("Enter text to score")
        userin = input()
        print(f"{score_text(userin)}\n")