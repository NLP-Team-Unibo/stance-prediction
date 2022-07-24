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
        """
        Creates a crossmodal attention module: if n_transformers is 0, then the module is composed of
        a simple multi-head attention layer, otherwise it's composed of a stack of Multimodal Transformer
        Layers, inpired by the MulT architecture.

        Parameters
        ----------
            - embed_dim: int
                Dimension of the input embeddings
            - num_heads: int
                Numer of heads for the multi-head attention layers
            - n_transformers: int
                Number of transformer layers in the MulT encoder, if zero then the encoder is replaced by a
                simple multi-head attention layer.
        """
        super().__init__()
        assert n_transformers >= 0
        self.n_transformers = n_transformers
        if n_transformers == 0:
            self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        else:
            self.attn = transformer.TransformerEncoder(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                layers=n_transformers, 
                attn_dropout=0.1, 
                relu_dropout=0.1, 
                res_dropout=0.1,
                embed_dropout=0.25,
                attn_mask=False)

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
    """
    Adapted from https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/bart/modeling_bart.py
    Changes:
        - Added an input parameter to __init__() in order be able to assign a custom model to self.model
        - Removed the classification head and the loss computation
        - Added an audio_embeddings input in forward()
    """
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
        sentence_representation = hidden_states[:, -1, :]
        return sentence_representation

class BartCustomForConditionalGeneration(BartForConditionalGeneration):
    """
    Adapted from https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/bart/modeling_bart.py
    Changes:
        - Added an input parameter to __init__() in order be able to assign a custom model to self.model
        - Added an audio_embeddings input in forward()
    """
    def __init__(self, config, model):
        super().__init__(config)
        self.model = model
    
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        audio_embeddings=None, 
        decoder_input_ids=None, 
        decoder_attention_mask=None, 
        head_mask=None, 
        decoder_head_mask=None, 
        cross_attn_head_mask=None, 
        encoder_outputs=None, 
        past_key_values=None, 
        inputs_embeds=None, 
        decoder_inputs_embeds=None, 
        labels=None, 
        use_cache=None, 
        output_attentions=None, 
        output_hidden_states=None, 
        return_dict=None):

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

class BartCustomDecoder(BartDecoder):
    """
    Adapted from https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/bart/modeling_bart.py
    Changes:
        - Added CrossAttention module
        - Added an audio_embeddings input in forward() and adapted the method to use the CrossAttention 
    """
    def __init__(self, config, embed_tokens, n_transformers=0, embed_audio_in_encoder=True):
        super().__init__(config, embed_tokens)
        self.embed_audio_in_encoder = embed_audio_in_encoder
        if not embed_audio_in_encoder:
            self.attn = CrossAttention(embed_dim=config.d_model, num_heads=8, n_transformers=n_transformers)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        audio_embeddings=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
    
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        if not self.embed_audio_in_encoder:
            if audio_embeddings is None:
                    audio_embeddings = output[0]

            if not return_dict:
                out_attn = self.attn(output[0], audio_embeddings, audio_embeddings)
                output = (out_attn, ) + output[1:]
                return output
            output.last_hidden_state = self.attn(output.last_hidden_state, audio_embeddings, audio_embeddings)
        return output        

class BartCustomEncoder(BartEncoder):
    """
    Adapted from https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/bart/modeling_bart.py
    Changes:
        - Added CrossAttention module
        - Added an audio_embeddings input in forward() and adapted the method to use the CrossAttention 
    """
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
    def __init__(self, config, embed_tokens, n_transformers=0, embed_audio_in_encoder=True):
        super().__init__(config, embed_tokens)
        self.embed_audio_in_encoder = embed_audio_in_encoder
        if embed_audio_in_encoder:
            self.attn = CrossAttention(embed_dim=config.d_model, num_heads=8, n_transformers=n_transformers)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        audio_embeddings=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        if self.embed_audio_in_encoder:
            if audio_embeddings is None:
                audio_embeddings = output[0]

            if not return_dict:
                out_attn = self.attn(output[0], audio_embeddings, audio_embeddings)
                output = (out_attn, ) + output[1:]
                return output
            output.last_hidden_state = self.attn(output.last_hidden_state, audio_embeddings, audio_embeddings)
        return output

class BartCustomModel(BartModel):
    """
    Adapted from https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/bart/modeling_bart.py
    Changes:
        - Added the possibility of overriding self.encoder e self.decoder with our custom models
        - Added an audio_embeddings input in forward() and adapted the method to use it
    """
    def __init__(self, config, encoder, n_transformers=0, embed_audio_in_encoder=True):
        super().__init__(config)
        self.encoder = encoder
        self.shared = self.encoder.get_input_embeddings()
        self.decoder = BartCustomDecoder(config, self.shared, n_transformers=n_transformers, embed_audio_in_encoder=embed_audio_in_encoder)

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
            audio_embeddings=audio_embeddings,
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
            embed_audio_in_encoder=True
        ):
        """
            Creates a Multi-task model combining a BartCustomencoder with two different BartCustomDecoder, one trained for text
            generation and the other trained for text classification. If generate_motion=False, the Multi-task model becomes single task and
            restricts itself to text classification. If use_audio=False, then the model doesn't use any multimodal information.
    
            Parameters
            ----------
            dropout_value: float
                Dropout value to apply before passing the embeddings to the classification head. Default to 0.3.
            bart_encoder_n_trainable_layers: int
                Number of trainable BART encoder Transformer layers. Default to 6.
            bart_decoder_cls_n_trainable_layers: int
                Number of trainable BART dencoder Transformer layers for the classification model. Default to 6.
            bart_decoder_gen_n_trainable_layers: int
                Number of trainable BART dencoder Transformer layers for the generation model. Default to 6.
            wav2vec2_n_transformers: int
                Number of Transformer layers to use for wav2vec2.0. Default to 12.
            wav2vec2_n_trainable_layers: int
                Number of Transformer layers to train for wav2vec2.0. Default to 12.
            cross_attn_n_layers: int
                Number of Multimodal Transformer layers to use in the CrossAttention module. Default to 0.
            use_audio: bool
                Whether to use audio information or not. Default to True.
            generate_motion: bool
                Whether to use the text generation decoder or not. Default to True.
            embed_audio_in_encoder: bool
                Whether the CrossAttention module should be added after the custom BART encoder or after the custom BART decoder. Default to True.
        """
        super(TextGenerationModel, self).__init__()
        self.use_audio = use_audio
        self.generate_motion = generate_motion
        bart_encoder = BartCustomEncoder.from_pretrained('facebook/bart-base', embed_tokens=None, n_transformers=cross_attn_n_layers, embed_audio_in_encoder=embed_audio_in_encoder)
        bart_model_cls = BartCustomModel.from_pretrained('facebook/bart-base', encoder=bart_encoder, n_transformers=cross_attn_n_layers, embed_audio_in_encoder=embed_audio_in_encoder)
        self.bart_cls = BartCustomForSequenceClassification.from_pretrained('facebook/bart-base', bart_model_cls)
        if self.generate_motion:
            bart_model_gen = BartCustomModel.from_pretrained('facebook/bart-base', encoder=bart_encoder, n_transformers=cross_attn_n_layers, embed_audio_in_encoder=embed_audio_in_encoder)
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

        self.__freeze_encoder_layers(
            encoder=bart_encoder, 
            n_trainable_layers=bart_encoder_n_trainable_layers, 
            embed_audio_in_encoder=embed_audio_in_encoder)
        self.__freeze_decoder_layers(
            decoder=bart_model_cls.get_decoder(),
            n_trainable_layers=bart_decoder_cls_n_trainable_layers,
            embed_audio_in_encoder=embed_audio_in_encoder
        )

        if self.generate_motion:
            self.__freeze_decoder_layers(
            decoder=bart_model_gen.get_decoder(),
            n_trainable_layers=bart_decoder_gen_n_trainable_layers,
            embed_audio_in_encoder=embed_audio_in_encoder
        )
        self.dropout = nn.Dropout(p=dropout_value)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(768, 1)

    def __freeze_encoder_layers(self, encoder, n_trainable_layers, embed_audio_in_encoder):
        for param in encoder.parameters():
            param.requires_grad = False
        if n_trainable_layers > 0:
            for layer in encoder.layers[-n_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        if embed_audio_in_encoder:
            for param in encoder.attn.parameters():
                param.requires_grad = True
        return encoder
    
    def __freeze_decoder_layers(self, decoder, n_trainable_layers, embed_audio_in_encoder):
        for param in decoder.parameters():
            param.requires_grad = False
        if n_trainable_layers > 0:
            for layer in decoder.layers[-n_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        for param in decoder.embed_positions.parameters():
            param.requires_grad = True
        for param in decoder.layernorm_embedding.parameters():
            param.requires_grad = True
        if not embed_audio_in_encoder:
            for param in decoder.attn.parameters():
                param.requires_grad = True

        
    def forward(self, input_ids, attention_mask, audio, labels_lm=None, labels_cls=None, return_dict=True):
        out_audio = None
        if self.use_audio:
            out_audio, _ = self.wav2vec2(audio)

        encoder_outputs = None
        loss_lm = None
        if self.generate_motion:
            out = self.bart_gen(input_ids, attention_mask, out_audio, labels=labels_lm, return_dict=return_dict)
            loss_lm, encoder_outputs = out[0], (out[2], )
        
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