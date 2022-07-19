from os import truncate
import torch
from transformers import BartTokenizer
from utils.data import truncate_encoded_text

class MultimodalBartTokenizer:
    """
    tokenizer for text and audio features.
    """

    def __init__(
            self,
            pretrained_model_name='facebook/bart-base',
            begin_audio="<audio>",
            end_audio="</audio>",
            begin_text="<text>",
            end_text="</text>",
            audio_feat='<audio_feat>',
            cls_token="<cls>"
    ):
        self._base_tokenizer = BartTokenizer.from_pretrained(
            pretrained_model_name,
        )

        self.additional_special_tokens = [
            begin_audio,
            end_audio,
            begin_text,
            end_text,
            audio_feat,
            cls_token,
        ]

        self._base_tokenizer.add_special_tokens(
            {'additional_special_tokens': self.additional_special_tokens}
        )

        self.begin_audio = begin_audio
        self.end_audio = end_audio
        self.begin_text = begin_text
        self.end_text = end_text
        self.audio_feat = audio_feat
        self.cls_token = cls_token

        self.begin_audio_id = self.convert_tokens_to_ids(begin_audio)
        self.end_audio_id = self.convert_tokens_to_ids(end_audio)
        self.begin_text_id = self.convert_tokens_to_ids(begin_text)
        self.end_text_id = self.convert_tokens_to_ids(end_text)
        self.audio_feat_id = self.convert_tokens_to_ids(audio_feat)
        self.cls_token_id = self.convert_tokens_to_ids(cls_token)

        self.vocab_size = self._base_tokenizer.vocab_size
        self.bos_token = self._base_tokenizer.bos_token
        self.bos_token_id = self._base_tokenizer.bos_token_id
        self.eos_token = self._base_tokenizer.eos_token
        self.eos_token_id = self._base_tokenizer.eos_token_id
        self.pad_token = self._base_tokenizer.pad_token
        self.pad_token_id = self._base_tokenizer.pad_token_id
        self.unk_token = self._base_tokenizer.unk_token
        self.unk_token_id = self._base_tokenizer.unk_token_id

    def encode(self, *args, **kwargs):
        return self._base_tokenizer(*args, **kwargs)

    def encode_condition(self, audio_features_num=None, text_features=None):
        """
        tokenize text and audio features.
        the output format (after decoded back):
        [<audio> <audio_feat> ... <audio_feat> </audio>] [<text> <token_id> ... <token_id> </text>]
        :param task_type: str or list[str]
        :param img_num: int or list[int], the number of image features
        :param event: str or list[str], event descriptions
        :param mlm: str or list[str], sentence for masked language modeling
        :return: dict {str: Tensor}, {
                "input_ids": ...,
                "attention_mask": ...,
                "event_mask": ...,          only exist if event is given. 1 for the position with event tokens
                "mlm_mask": ...,            only exist if mlm is given. 1 for the position with mlm tokens
                "img_mask":...,             only exist if img_num is given. 1 for the position with img tokens
            }
        """
        text = ''

        # build audio features
        # <audio> <audio_feat> ... <audio_feat> </audio>
        if audio_features_num is not None:
                text += self.begin_audio + self.audio_feat * audio_features_num + self.end_audio

        # build text tokens
        # <text> <token_id> ... <token_id> </text>
        if text_features is not None:
            text += self.begin_text + text_features + self.end_text
        
        encoded = self.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            return_tensors='pt',
            padding=True
        )

        # build img mask
        if audio_features_num is not None:
            encoded['audio_mask'] = encoded['input_ids'] == self.audio_feat_id

        return encoded
    
    def encode_mult(self, audio_features_num, text_features, cut_mode='first'):

        audio_str = ''
        if audio_features_num is not None:
                audio_str += self.begin_audio + self.audio_feat * audio_features_num + self.end_audio
        audio_encoded = self.encode(audio_str, add_special_tokens=False)

        text_encoded = self.encode(text_features, add_special_tokens=False)
        text_encoded = truncate_encoded_text(text_encoded, mode=cut_mode, is_bart_encoding=True)
        encoded = text_encoded
        encoded['input_ids'] = audio_encoded['input_ids'] + [self.begin_text_id] + text_encoded['input_ids'] + [self.end_text_id]

        encoded['attention_mask'] = [1 for _ in encoded['input_ids']]
        for k in encoded.keys():
            encoded[k] = torch.tensor(encoded[k])
        
        #encoded['audio_mask'] = encoded['input_ids'] == self.audio_feat_id
        return encoded

    def encode_label(self, label, audio_features_num=None):

        #TODO: rewrite this to go in accordance with the above methods
        text = self.bos_token + label + self.eos_token

        if audio_features_num is not None:
            text += self.begin_audio + self.audio_feat * audio_features_num + self.end_audio

        encoded_label = self.encode(
            text,
            add_special_tokens=False,
            return_tensors='pt',
            truncation=True
        )

        input_ids = encoded_label['input_ids']
        attention_mask = encoded_label['attention_mask']

        output_shape = input_ids[:, 1:].shape
        labels = torch.empty(output_shape, dtype=torch.long)
        decoder_input_ids = torch.empty(output_shape, dtype=torch.long)
        decoder_attention_mask = torch.empty(output_shape, dtype=torch.long)

        # remove <s> from labels, remove </s> from decoder_input_ids
        # remove the element in attention_mask at the same position as </s> in decoder_input_ids
        for i in range(labels.size(0)):
            labels[i] = input_ids[i][input_ids[i] != self.bos_token_id]
            decoder_input_ids[i] = input_ids[i][input_ids[i] != self.eos_token_id]
            decoder_attention_mask[i] = attention_mask[i][input_ids[i] != self.eos_token_id]

        labels[(labels == self.pad_token_id) |
                (labels == self.begin_audio_id) |
                (labels == self.end_audio_id) |
                (labels == self.audio_feat_id)] = -100

        output = {
            'labels': labels,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask
        }

        # build audio mask
        if audio_features_num is not None:
            output['label_audio_mask'] = labels == self.audio_feat_id
            output['decoder_input_audio_mask'] = decoder_input_ids == self.audio_feat_id

        return output

    def decode(self, token_ids, skip_special_tokens=False):
        return self._base_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def batch_decode(self, token_ids, skip_special_tokens=False):
        return self._base_tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )

    def convert_tokens_to_ids(self, tokens):
        return self._base_tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        return self._base_tokenizer.convert_ids_to_tokens(ids)

    def get_base_tokenizer(self):
        return self._base_tokenizer

    def __len__(self):
        return len(self._base_tokenizer)