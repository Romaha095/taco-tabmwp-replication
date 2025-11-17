from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_tapex(model_name="microsoft/tapex-large"):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tok, model
