from bdb import checkfuncname
import pickle
import torch
from torchinfo import summary
from transformers import BartTokenizer
from models.text_model import BartTextModel
from models.multimodal_bart.modules import MultiModalBartEncoder
from models.multimodal_bart.config import MultiModalBartConfig

from models.multimodal_bart.tokenizer import MultimodalBartTokenizer
from ibm_dataset import IBMDebater
from torch.utils.data import DataLoader
from utils.batch_generators import batch_generator_mult_bart

#cfg = MultiModalBartConfig()

#model = BartTextModel(classify=True, multimodal=True)
#model.to('cuda')
#model = MultiModalBartEncoder(config=cfg)

#summary(model)
#print(len(model.layers))
#summary(model.text_gen)
#tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

tokenizer = MultimodalBartTokenizer()

data = IBMDebater(path='full', split='train', tokenizer=tokenizer, chunk_length=10, sample_cut_type='last', load_text=True, load_motion=True, load_audio_emb=True, load_audio=False)

loader = DataLoader(data, batch_size=8, collate_fn=batch_generator_mult_bart)
for elem in loader:
    print(len(elem))
    print(elem[0])
    print(elem[1])
    print(elem[2].shape)
    print(elem[3])
    break

"""
text = We should adopt multiculturalism because it's good from the perspective in that it's going to help other people and we should also adopt it because it's good for our own society as well economically and ideologically.
So let's start by talking about why this is just the moral, good position to have in the world.
Because when you have a more multicultural society you're going to be more understanding of different cultures.
So if you're a society that's open in accepting to muslims, to people of all different religions or ethnicities, genders and all kinds of things, when you have more of a cultural diversity, you're going to be better able to understand those cultures.
And that's crucial because there's we live in a society in the internet age where things are very interconnected.
The economy is incredibly interconnected, a lot of political events and like for example terrorist attacks that happened, things like that, a lot of conflicts take place on a global level, when you talk about the environment, we're seeing a proliferation in global issues that we have to grapple with as a globe, there's things like global poverty , there's all kinds of global issues.
And when you have more of a multicultural understanding, when societies are more representative of the world at large, you're better able to just be a leader in society but you're also better able to generate moral and representative policies.
Which is obviously very good.
But I also think that there is economic good that comes out of this as well.
Because for example, if you like have like a large chinese population in america, they are people who are like who know chinese and things like that, maybe people who are like descendants of chinese immigrants to america, there could be more of a familiarity with like business opportunities, you could have more of an understanding diplomatically, maybe, if you have like ties to certain conflict regions, and you can help therefore solve in those kinds of ends, there's so there's all kinds of ways in which like you can help solve problems substantively.
But I also think and this is very important is that multiculturalism leads to more of a diversity of experiences and more of an enriching life experience.
Because I think that one of the biggest things is that people are oftentimes constantly learning, looking for new things to understand, meeting different kinds of people.
And I think that one of the big reasons why, for example, a lot of people like to travel, is because there's this like this basic human urge to learn about different cultures and it doesn't have to come at the devaluation of your own, but learning about different cultures, being more aware and understanding.
And I think that a lot of this comes with education and citizenship and all kinds of things that are going to be more likely to happen if you live in a multicultural society.
So if that's not good enough though, there's also economic benefits, political benefits and environmental benefits.
All these kinds of things.
So for all those reasons we should adopt multiculturalism.
motion = "We should adopt multiculturalism"



with open('out/10-last/train/DJ_3162_multiculturalism_pro.pkl', 'rb') as f:
    audio = pickle.load(f)

audio1 = audio.squeeze().tolist()
out = tokenizer.encode_condition(audio_features_num=len(audio1), text_features=text)
out2 = tokenizer.alternative_encode(audio_features_num=len(audio1), text_features=text)

#print(out2['input_ids'][None, :].shape)

#output = model(out2['input_ids'][None, :].to('cuda'), audio, out2['attention_mask'][None, :].to('cuda'))

#print(output[1].logits.size(), output[0].size())

gen = model.generate(
    input_ids=out2['input_ids'][None, :].to('cuda'),
    audio_features=audio,
    attention_mask=out2['attention_mask'][None, :].to('cuda'))
print(gen)

print(tokenizer.batch_decode(gen))
"""

"""
inputs = tokenizer(text, truncation=True, return_tensors='pt')
print(inputs['input_ids'].shape)
labels = tokenizer(motion, truncation=True, return_tensors='pt')['input_ids']

inputs = {k:torch.cat([inputs[k] for _ in range(4)]) for k in inputs.keys()}
labels = torch.cat([labels for _ in range(4)])
print(inputs['input_ids'].shape, labels.shape)

#print(model.config.embed_dim)

out = model(**inputs, labels=labels)
print(model.get_encoder())
print(out.logits.shape, out.loss)
print(out.logits.view(-1, model.config.vocab_size).shape, labels.view(-1).shape)
"""