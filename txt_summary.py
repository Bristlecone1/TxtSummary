import streamlit as st 
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

model_name = 'google/pegasus-xsum'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

txt = 0
st.write('Text to analyze')
txt = st.text_area('')

if txt != 0 and bool(txt.strip()) is True:
  batch = tokenizer.prepare_seq2seq_batch(txt, truncation=True, padding='longest',return_tensors='pt')
  translated = model.generate(**batch)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

  st.write("Output:")
  st.write(tgt_text[0])
