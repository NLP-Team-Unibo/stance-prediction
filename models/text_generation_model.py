import torchaudio
import torch
from torch import nn
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import BartForConditionalGeneration, BartForSequenceClassification
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput, Seq2SeqSequenceClassifierOutput
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

class BartCustomForSequenceClassification(BartForSequenceClassification):
    def __init__(self, config, model, **kwargs):
        super().__init__(config, **kwargs)
        self.model = model
        self.classification_head = None

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        audio_embeddings = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            audio_embeddings=audio_embeddings,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        """eos_mask = input_ids.eq(self.config.eos_token_id)
        print

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]"""
        sentence_representation = hidden_states[:, -1, :]
        return sentence_representation

class BartCustomForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config, model):
        super().__init__(config)
        self.model = model
    
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
            audio_embeddings=audio_embeddings,
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
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

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

from transformers import BartModel
from transformers.models.bart.modeling_bart import BartDecoder, BartEncoder

class BartCustomEncoder(BartEncoder):
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
    def __init__(self, config, embed_tokens, n_transformers=0):
        super().__init__(config, embed_tokens)
        self.attn = CrossAttention(embed_dim=config.d_model, num_heads=8, n_transformers=n_transformers)
    

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        audio_embeddings=None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        output = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        if audio_embeddings is None:
            audio_embeddings = output[0]

        if not return_dict:
            out_attn = self.attn(output[0], audio_embeddings, audio_embeddings)
            output = (out_attn, ) + output[1:]
            return output
        output.last_hidden_state = self.attn(output.last_hidden_state, audio_embeddings, audio_embeddings)
        return output

class BartCustomModel(BartModel):
    def __init__(self, config, encoder):
        super().__init__(config)
        self.encoder = encoder
        self.shared = self.encoder.get_input_embeddings()
        self.decoder = BartDecoder(config, self.shared)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        audio_embeddings = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds= None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = True
    ):
    
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_embeddings=audio_embeddings,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class TextGenerationModel(StancePredictionModule):
    def __init__(
            self, 
            dropout_value = 0.3,
            bart_encoder_n_trainable_layers = 6,
            bart_decoder_cls_n_trainable_layers = 6,
            bart_decoder_gen_n_trainable_layers = 6,
            wav2vec2_n_transformers = 12,
            wav2vec2_n_trainable_layers = 12,
            cross_attn_n_layers = 0,
            use_audio=True,
            generate_motion=True,
        ):
        super(TextGenerationModel, self).__init__()
        self.use_audio = use_audio
        self.generate_motion = generate_motion
        bart_encoder = BartCustomEncoder.from_pretrained('facebook/bart-base', embed_tokens=None, n_transformers=cross_attn_n_layers)
        bart_model_cls = BartCustomModel.from_pretrained('facebook/bart-base', bart_encoder)
        self.bart_cls = BartCustomForSequenceClassification.from_pretrained('facebook/bart-base', bart_model_cls)
        if self.generate_motion:
            bart_model_gen = BartCustomModel.from_pretrained('facebook/bart-base', bart_encoder)
            self.bart_gen = BartCustomForConditionalGeneration.from_pretrained('facebook/bart-base', bart_model_gen)

        if use_audio:
            self.__bundle = torchaudio.pipelines.WAV2VEC2_BASE
            self.wav2vec2 = self.__bundle.get_model()
            self.wav2vec2.encoder.transformer.layers = self.wav2vec2.encoder.transformer.layers[:wav2vec2_n_transformers]
            assert wav2vec2_n_transformers >= wav2vec2_n_trainable_layers
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            if wav2vec2_n_trainable_layers > 0:
                for layer in self.wav2vec2.encoder.transformer.layers[-wav2vec2_n_trainable_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        for param in bart_encoder.parameters():
            param.requires_grad = False
        if bart_encoder_n_trainable_layers > 0:
            for layer in bart_encoder.layers[-bart_encoder_n_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        for param in bart_model_cls.get_decoder().parameters():
            param.requires_grad = False
        if bart_decoder_cls_n_trainable_layers > 0:
            for layer in bart_model_cls.get_decoder().layers[-bart_decoder_cls_n_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        if self.generate_motion:
            for param in bart_model_gen.get_decoder().parameters():
                param.requires_grad = False
            if bart_decoder_gen_n_trainable_layers > 0:
                for layer in bart_model_gen.get_decoder().layers[-bart_decoder_gen_n_trainable_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        

        self.dropout = nn.Dropout(p=dropout_value)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(768, 1)
        
    def forward(self, input_ids, attention_mask, audio, labels_lm=None, labels_cls=None, return_dict=True):
        out_audio = None
        if self.use_audio:
            out_audio, _ = self.wav2vec2(audio)

        encoder_outputs = None
        loss_lm = None
        if self.generate_motion:
            out = self.bart_gen(input_ids, attention_mask, out_audio, labels=labels_lm, return_dict=return_dict)
            loss_lm, encoder_outputs = out[0], (out[2], )
            #_, encoder_outputs = out_bart
        
        out_cls = self.bart_cls(input_ids, attention_mask, out_audio, encoder_outputs=encoder_outputs, return_dict=return_dict)

        out_cls = self.dropout(out_cls)
        out_cls = self.relu(out_cls)
        out_cls = self.classifier(out_cls)
        out_cls = out_cls.squeeze(1)

        loss_fn = nn.BCEWithLogitsLoss()
        loss_cls = loss_fn(out_cls, labels_cls)

        return loss_lm, loss_cls, out_cls

    def generate(self, **kwargs):
        if self.use_audio:
            kwargs['audio_embeddings'], _ = self.wav2vec2(kwargs['audio_embeddings'])
        return self.bart_gen.generate(**kwargs)