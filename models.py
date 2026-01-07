"""Model architectures for CLIPDR."""

from typing import Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
from clip import clip
from clip.model import CLIP


class PlainPromptLearner(nn.Module):
    """Learnable prompt module for CLIP."""
    
    clip_max_num_tokens = 77
    rank_tokens_position_candidates = {"tail", "middle", "front"}
    
    def __init__(
        self,
        clip_model: CLIP,
        num_ranks: int,
        num_tokens_per_rank: Union[int, List],
        num_context_tokens: int,
        rank_tokens_position: str = "tail",
        init_context: Optional[str] = None,
        rank_specific_context: bool = False,
        init_rank_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.num_ranks = num_ranks
        self.num_context_tokens = num_context_tokens
        self.rank_tokens_positon = rank_tokens_position
        
        dtype = clip_model.token_embedding.weight.dtype
        context_embeds, _num_context_tokens = self.create_context_embeds(
            clip_model, num_ranks, num_context_tokens, init_context, rank_specific_context, dtype
        )
        num_context_tokens = _num_context_tokens
        self.context_embeds = nn.Parameter(context_embeds)
        
        if isinstance(num_tokens_per_rank, int):
            num_tokens_per_rank = [num_tokens_per_rank] * num_ranks
        rank_embeds, _num_tokens_per_rank = self.create_rank_embeds(
            clip_model, num_ranks, num_tokens_per_rank, init_rank_path, dtype, num_context_tokens
        )
        num_tokens_per_rank = _num_tokens_per_rank
        self.rank_embeds = nn.Parameter(rank_embeds)
        assert len(rank_embeds) == num_ranks
        
        psudo_sentence_tokens = self.create_psudo_sentence_tokens(
            num_tokens_per_rank, num_context_tokens, num_ranks
        )
        self.register_buffer("psudo_sentence_tokens", psudo_sentence_tokens, persistent=False)
        
        self.num_context_tokens = num_context_tokens
        self.num_tokens_per_rank = num_tokens_per_rank
        if rank_tokens_position not in self.rank_tokens_position_candidates:
            raise ValueError(f"Invalid rank_tokens_position: {rank_tokens_position}")
        self.rank_tokens_positon = rank_tokens_position
        self.num_ranks = num_ranks
        self.embeddings_dim = clip_model.token_embedding.embedding_dim
        
        self.create_sentence_embeds_template(clip_model, num_ranks, psudo_sentence_tokens)
    
    def forward(self):
        context_embeds = self.context_embeds
        
        if context_embeds.dim() == 2:
            context_embeds = context_embeds[None].expand(self.num_ranks, *context_embeds.shape)
        
        sentence_embeds = self.sentence_embeds.clone()
        if self.rank_tokens_positon == "tail":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                sentence_embeds[i, 1:1 + pure_sentence_length] = torch.cat(
                    [context_embeds[i], self.rank_embeds[i, :_num_tokens_per_rank]], dim=0
                )
        elif self.rank_tokens_positon == "front":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                sentence_embeds[i, 1:1 + pure_sentence_length] = torch.cat(
                    [self.rank_embeds[i, :_num_tokens_per_rank], context_embeds[i]], dim=0
                )
        elif self.rank_tokens_positon == "middle":
            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + _num_tokens_per_rank
                _context_embeds = context_embeds[i]
                half_range = self.num_context_tokens // 2
                sentence_embeds[i, 1:1 + pure_sentence_length] = torch.cat(
                    [
                        _context_embeds[:half_range],
                        self.rank_embeds[i, :_num_tokens_per_rank],
                        _context_embeds[half_range:],
                    ],
                    dim=0,
                )
        return sentence_embeds
    
    def create_sentence_embeds_template(self, clip_model, num_ranks, psudo_sentence_tokens):
        with torch.no_grad():
            device = clip_model.token_embedding.weight.device
            dtype = clip_model.token_embedding.weight.dtype
            
            null_embed = clip_model.token_embedding(torch.tensor([0], device=device))[0].to(dtype)
            sot_embed = clip_model.token_embedding(torch.tensor([49406], device=device))[0].to(dtype)
            eot_embed = clip_model.token_embedding(torch.tensor([49407], device=device))[0].to(dtype)
            full_stop_embed = clip_model.token_embedding(torch.tensor([269], device=device))[0].to(dtype)
        
        sentence_embeds = null_embed[None, None].repeat(num_ranks, self.clip_max_num_tokens, 1)
        argmax_index = psudo_sentence_tokens.argmax(dim=-1)
        rank_index = torch.arange(num_ranks)
        
        sentence_embeds[:, 0, :] = sot_embed
        sentence_embeds[rank_index, argmax_index] = eot_embed
        sentence_embeds[rank_index, argmax_index - 1] = full_stop_embed
        
        self.register_buffer("sentence_embeds", sentence_embeds, persistent=False)
    
    def create_psudo_sentence_tokens(self, num_tokens_per_rank, num_context_tokens, num_ranks):
        psudo_sentence_tokens = torch.zeros(num_ranks, self.clip_max_num_tokens, dtype=torch.long)
        
        if isinstance(num_tokens_per_rank, List):
            assert num_ranks == len(num_tokens_per_rank)
            for i, _num_tokens_per_rank in enumerate(num_tokens_per_rank):
                sentence_length = 1 + num_context_tokens + _num_tokens_per_rank + 1 + 1
                psudo_sentence_tokens[i, :sentence_length] = torch.arange(0, sentence_length, dtype=torch.long)
        else:
            sentence_length = 1 + num_context_tokens + num_tokens_per_rank + 1 + 1
            psudo_sentence_tokens[:, :sentence_length] = torch.arange(0, sentence_length, dtype=torch.long)
        return psudo_sentence_tokens
    
    def create_rank_embeds(
        self, clip_model, num_ranks, num_tokens_per_rank, init_rank_path, dtype, num_context_tokens
    ):
        if init_rank_path is not None:
            rank_names = self.read_rank_file(init_rank_path)
            
            if len(rank_names) != num_ranks:
                raise ValueError("rank_names length mismatch")
            
            _rank_tokens = [clip._tokenizer.encode(rank_name) for rank_name in rank_names]
            _num_tokens_per_rank = [len(rank_token) for rank_token in _rank_tokens]
            num_tokens_per_rank = _num_tokens_per_rank
            max_num_tokens_per_rank = np.max(num_tokens_per_rank)
            
            rank_tokens = torch.zeros(len(_rank_tokens), max_num_tokens_per_rank, dtype=torch.long)
            for i, rank_token in enumerate(_rank_tokens):
                valid_length = self.clip_max_num_tokens - num_context_tokens - 3
                if len(rank_token) > valid_length:
                    rank_token = rank_token[:valid_length]
                    raise ValueError("rank tokens too long")
                rank_tokens[i, :len(rank_token)] = torch.LongTensor(rank_token)
            
            rank_embeds = clip_model.token_embedding(rank_tokens).type(dtype)
            rank_embeds = rank_embeds[:, :max_num_tokens_per_rank]
        else:
            embeddings_dim = clip_model.token_embedding.embedding_dim
            if isinstance(num_tokens_per_rank, List):
                max_num_tokens_per_rank = np.max(num_tokens_per_rank)
            else:
                max_num_tokens_per_rank = num_tokens_per_rank
            if self.clip_max_num_tokens < num_context_tokens + max_num_tokens_per_rank + 3:
                raise ValueError("rank tokens too long")
            rank_embeds = torch.empty((num_ranks, max_num_tokens_per_rank, embeddings_dim), dtype=dtype)
            nn.init.normal_(rank_embeds, std=0.02)
        
        return rank_embeds, num_tokens_per_rank
    
    def read_rank_file(self, init_rank_path):
        rank_names = []
        with open(init_rank_path, "r") as f:
            for line in f.readlines():
                line = line.strip().replace("_", " ")
                rank_names.append(line)
        return rank_names
    
    def create_context_embeds(
        self,
        clip_model,
        num_ranks: int,
        num_context_tokens: int,
        init_context: Optional[str],
        rank_specific_context: bool,
        dtype,
    ):
        if init_context is not None:
            init_context = init_context.replace("_", " ")
            prompt_tokens = clip.tokenize(init_context)[0]
            _num_context_tokens = torch.argmax(prompt_tokens).item() - 1
            num_context_tokens = _num_context_tokens
            
            with torch.no_grad():
                context_embeds = clip_model.token_embedding(prompt_tokens).type(dtype)
            context_embeds = context_embeds[1:1 + num_context_tokens]
            
            if rank_specific_context:
                context_embeds = context_embeds[None].repeat(num_ranks, 1, 1)
        else:
            embeds_dim = clip_model.token_embedding.embedding_dim
            if rank_specific_context:
                context_embeds = torch.empty((num_ranks, num_context_tokens, embeds_dim), dtype=dtype)
            else:
                context_embeds = torch.empty((num_context_tokens, embeds_dim), dtype=dtype)
            nn.init.normal_(context_embeds, std=0.02)
        
        return context_embeds, num_context_tokens


class TextEncoder(nn.Module):
    """Text encoder from CLIP."""
    
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
    
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        
        x = x[
            torch.arange(x.shape[0]),
            tokenized_prompts.argmax(dim=-1)
        ] @ self.text_projection
        
        return x


class CLIPDR(nn.Module):
    """CLIP-based Diabetic Retinopathy model."""
    
    def __init__(self, clip_model, prompt_learner):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale
        self.prompt_learner = prompt_learner
        self.psudo_sentence_tokens = prompt_learner.psudo_sentence_tokens
        self.embed_dims = clip_model.text_projection.shape[1]
        self.num_ranks = self.prompt_learner.num_ranks
        self.text_encoder = TextEncoder(clip_model)
    
    def forward(self, images):
        sentence_embeds = self.prompt_learner()
        psudo_sentence_tokens = self.psudo_sentence_tokens
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = self.image_encoder(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        # Return raw features too for FDS
        return logits, image_features, text_features


def build_clipdr_model(device='cuda'):
    """Build and initialize CLIPDR model."""
    import config
    
    clip_model, _ = clip.load(config.CLIP_MODEL_NAME, device=device)
    clip_model = clip_model.float()
    
    prompt_learner = PlainPromptLearner(
        clip_model=clip_model,
        num_ranks=config.NUM_RANKS,
        num_tokens_per_rank=config.NUM_TOKENS_PER_RANK,
        num_context_tokens=config.NUM_CONTEXT_TOKENS,
        rank_tokens_position=config.RANK_TOKENS_POSITION,
        init_context=config.INIT_CONTEXT,
        rank_specific_context=config.RANK_SPECIFIC_CONTEXT
    ).to(device)
    
    model = CLIPDR(
        clip_model=clip_model,
        prompt_learner=prompt_learner
    ).to(device)
    
    return model