from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def analyze_sentiment(text):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp(text)

# Majoring in Intelligence Systems - AI development
# Implementing scalable NLP solutions