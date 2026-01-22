from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

DEBUG = False
MODEL_NAME = "ProsusAI/finbert"

#initialise finbert model (force CPU for Macbook)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
model.to("cpu")

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

#debug loop for model testing
if DEBUG:
    while True:
        print("Enter text to score")
        userin = input()
        print(f"{score_text(userin)}\n")