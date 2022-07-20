import torchaudio
import torch
from torch import nn
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import BartForConditionalGeneration
from models.stance_prediction_module import StancePredictionModule

class BartMultForConditionalGeneration(BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = ['attn.in_proj_bias', 'attn.out_proj.weight', 'attn.out_proj.bias', 'attn.in_proj_weight', 'lm_head.weight', 'final_logits_bias']
    def __init__(self, config):
        super().__init__(config)
        self.attn = nn.MultiheadAttention(embed_dim=config.d_model, num_heads=8, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, audio_embeddings=None, decoder_input_ids=None, decoder_attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, encoder_outputs=None, past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                print("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_output = outputs[0]

        mult_outputs, _ = self.attn(query=text_output, key=audio_embeddings, value=audio_embeddings)
        lm_logits = self.lm_head(mult_outputs)+ self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            return (masked_lm_loss, outputs[0])
        else:
            return (lm_logits, outputs[0])

class TextGenerationModel(StancePredictionModule):
    def __init__(
        self, 
    ):
        super(TextGenerationModel, self).__init__()
        self.bart = BartMultForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.__bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2 = self.__bundle.get_model()
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(2*768, 1)
        
    def forward(self, input_ids, attention_mask, audio, labels_lm=None, labels_cls=None):
        out_audio, _ = self.wav2vec2(audio)
        loss_lm, out_bart = self.bart(input_ids, attention_mask, out_audio, labels=labels_lm)
        
        out_audio = torch.mean(out_audio, axis=1)
        out_bart = out_bart[:, -1, :]

        out_cls = torch.concat([out_bart, out_audio], axis=1)
        out_cls = self.relu(out_cls)
        out_cls = self.classifier(out_cls)
        out_cls = out_cls.squeeze(1)

        loss_fn = nn.BCEWithLogitsLoss()
        loss_cls = loss_fn(out_cls, labels_cls)

        return loss_cls + loss_lm

    def generate(self, **kwargs):
        return self.bart.generate(**kwargs)
