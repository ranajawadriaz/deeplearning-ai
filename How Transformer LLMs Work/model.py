
import warnings

#remove warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline

# for tokenization
tokenizer=AutoTokenizer.from_pretrained('gpt2')

#for model
model=AutoModelForCausalLM.from_pretrained(
    'gpt2',
    device_map='cpu',
    torch_dtype='auto', #data type auto
    trust_remote_code=True, # trust the repository code
)

#pipelining the model and tokenizer in one pipeline
generator=pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=50,
    do_sample=False,#no randomness, deterministic

)

prompt="write deep psycho love message to my girl-friend"
output=generator(prompt)

print(output[0]['generated_text']) #only return the response not the prompt

"""**Model Analyze**"""

model #peeking inside engine

model.transformer.wte #embed text vectors; model.model.embed_tokens for other models

model.transformer.h[0] #first transformer block

#model.model for all layers
#model.model.layers[0] for first transformer block
model.lm_head #final layer

"""**Complete the text**"""

message="The capital of France is the "

input_ids=tokenizer(message,return_tensors='pt').input_ids
# all message token ids

model_output=model.transformer(input_ids)
model_output[0].shape
# feeded into the model

lm_head_output=model.lm_head(model_output[0])
lm_head_output.shape
# final layer output

ParisId=lm_head_output[0,-1].argmax(-1)
ParisId
#the highest number token or most likely after the last token in batch 0

actualPredictedWord=tokenizer.decode(ParisId)
actualPredictedWord
#predicted text is wrong due to old model