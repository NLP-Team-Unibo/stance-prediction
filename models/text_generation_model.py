from turtle import forward
import torchaudio
import torch
from torch import nn
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import BartForConditionalGeneration, BartPretrainedModel, BartModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from models.stance_prediction_module import StancePredictionModule
from models.mult_modules import transformer

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, n_transformers):
        super().__init__()
        assert n_transformers >= 0
        self.n_transformers = n_transformers
        if n_transformers == 0:
            self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        else:
            self.attn = transformer.TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, layers=n_transformers)
    def forward(self, query, key, value):
        if self.n_transformers > 0:
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            x = self.attn(query, key, value)
            x = x.permute(1, 0, 2)
        else:
            x, _ = self.attn(query, key, value)
            
        return x
                

class BartMultForConditionalGeneration(BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = ['attn.in_proj_bias', 'attn.out_proj.weight', 
                                       'attn.out_proj.bias', 'attn.in_proj_weight', 
                                       'lm_head.weight', 'final_logits_bias', 
                                       'attn.attn.version', 'attn.attn.layers.0.fc1.bias', 
                                       'attn.attn.layers.0.layer_norms.1.bias', 'attn.attn.layers.1.fc2.weight', 
                                       'attn.attn.layer_norm.weight', 'attn.attn.layer_norm.bias', 
                                       'attn.attn.embed_positions._float_tensor', 'attn.attn.layers.1.fc1.weight', 
                                       'attn.attn.layers.1.layer_norms.0.bias', 'attn.attn.layers.0.layer_norms.0.weight', 
                                       'attn.attn.layers.0.fc2.weight', 'attn.attn.layers.1.fc2.bias', 
                                       'attn.attn.layers.0.fc1.weight', 'attn.attn.layers.1.layer_norms.0.weight', 
                                       'attn.attn.layers.1.layer_norms.1.bias', 'attn.attn.layers.1.layer_norms.1.weight', 
                                       'attn.attn.layers.0.layer_norms.1.weight', 'attn.attn.layers.0.fc2.bias', 
                                       'attn.attn.layers.1.fc1.bias', 'attn.attn.layers.0.layer_norms.0.bias']
    def __init__(self, config, n_transformers=0):
        super().__init__(config)
        self.attn = CrossAttention(embed_dim=self.config.d_model, num_heads=8, n_transformers=n_transformers)
    
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name):
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "audio_embeddings"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)

        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        #print('decode input ', decoder_input_ids.shape)
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        batch_size = kwargs['audio_embeddings'].shape[0]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "audio_embeddings": kwargs['audio_embeddings'].repeat_interleave(decoder_input_ids.shape[0] // batch_size, dim=0), #kwargs['audio_embeddings'],
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

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
        """
        if self.num_beams > 0:
            split_texts = torch.split(text_output, self.num_beams, axis=0)
            for text_output in split_texts:
                out, _ = self.attn(query=text_output, key=audio_embeddings, value=audio_embeddings)
                mult_ouputs.append(out)
            mult_outputs = torch.stack(mult_outputs, axis=0)
        """
        text_output = outputs[0]
        
        #mult_outputs = text_output
        mult_outputs = self.attn(query=text_output, key=audio_embeddings, value=audio_embeddings)
        lm_logits = self.lm_head(mult_outputs)+ self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            return (masked_lm_loss, outputs[0]) if masked_lm_loss is not None else (lm_logits, outputs[0])

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

class TextGenerationModel(StancePredictionModule):
    def __init__(
            self, 
            dropout_value = 0.3,
            bart_encoder_n_trainable_layers = 6,
            bart_decoder_n_trainable_layers = 6,
            wav2vec2_n_transformers = 12,
            wav2vec2_n_trainable_layers = 12,
            cross_attn_n_layers = 0,
        ):
        super(TextGenerationModel, self).__init__()
        self.bart = BartMultForConditionalGeneration.from_pretrained('facebook/bart-base', cross_attn_n_layers)
        self.__bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2 = self.__bundle.get_model()
        self.wav2vec2.encoder.transformer.layers = self.wav2vec2.encoder.transformer.layers[:wav2vec2_n_transformers]

        for param in self.bart.get_encoder().parameters():
            param.requires_grad = False
        if bart_encoder_n_trainable_layers > 0:
            for layer in self.bart.get_encoder().layers[-bart_encoder_n_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        for param in self.bart.get_decoder().parameters():
            param.requires_grad = False
        if bart_decoder_n_trainable_layers > 0:
            for layer in self.bart.get_decoder().layers[-bart_decoder_n_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        assert wav2vec2_n_transformers >= wav2vec2_n_trainable_layers
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        if wav2vec2_n_trainable_layers > 0:
            for layer in self.wav2vec2.encoder.transformer.layers[-wav2vec2_n_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.dropout = nn.Dropout(p=dropout_value)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(2*768, 1)
        
    def forward(self, input_ids, attention_mask, audio, labels_lm=None, labels_cls=None, return_dict=True):
        out_audio, _ = self.wav2vec2(audio)
        loss_lm, out_bart = self.bart(input_ids, attention_mask, out_audio, labels=labels_lm, return_dict=return_dict)
        
        out_audio = torch.mean(out_audio, axis=1)
        out_bart = out_bart[:, -1, :]

        out_cls = torch.concat([out_bart, out_audio], axis=1)
        out_cls = self.dropout(out_cls)

        out_cls = self.relu(out_cls)
        out_cls = self.classifier(out_cls)
        out_cls = out_cls.squeeze(1)

        loss_fn = nn.BCEWithLogitsLoss()
        loss_cls = loss_fn(out_cls, labels_cls)

        return loss_lm, loss_cls, out_cls

    def generate(self, **kwargs):
        kwargs['audio_embeddings'], _ = self.wav2vec2(kwargs['audio_embeddings'])
        return self.bart.generate(**kwargs)