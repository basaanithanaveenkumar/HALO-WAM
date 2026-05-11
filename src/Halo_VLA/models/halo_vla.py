from config import HaloVLMConfig
from models.vit import VisTransformer
from models.transformer import DecoderTransformer
from models.lm_head import LMHead
from models.image_proj import ImageProjector
from models.state_encoder import StateEncoder
# --- Flow matching action decoder replaces MLP ActionDecoder ---
from models.flow_action_decoder import FlowActionDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.positional_embeddings import SinusoidalPositionalEmbedding


class HaloVLM(nn.Module):
    def __init__(self, config: HaloVLMConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = HaloVLMConfig(**kwargs)
        self.config = config
        network_config = config

        self.vis_enc = VisTransformer(
            img_size=network_config.img_size,
            p_size=network_config.patch_size,
            in_chans=network_config.in_chans,
            emb_dim=network_config.emb_dim,
            num_layers=network_config.vit_num_layers,
            num_heads=network_config.vit_num_heads,
            mlp_dim=network_config.vit_mlp_dim,
            drop_fact=network_config.vit_drop,
        )
        self.decoder_transformer = DecoderTransformer(
            num_layers=network_config.dec_num_layers,
            emb_dim=network_config.emb_dim,
            num_heads=network_config.dec_num_heads,
            mlp_dim=network_config.dec_mlp_dim,
            drop_fact=network_config.dec_drop,
            use_moe=network_config.use_moe,
            moe_hid_scale=network_config.moe_hid_scale,
            moe_num_routed_experts=network_config.moe_num_routed_experts,
            moe_top_k=network_config.moe_top_k,
            moe_num_shared_experts=network_config.moe_num_shared_experts,
        )
        self.token_emb = nn.Embedding(network_config.vocab_size, network_config.emb_dim)
        self.pos_embed = nn.Embedding(network_config.max_position_embeddings, network_config.emb_dim)
        self.layer_norm = nn.LayerNorm(network_config.emb_dim)
        self.lm_head = LMHead(hidden_size=network_config.emb_dim, vocab_size=network_config.vocab_size)
        self.image_projector = ImageProjector(
            vision_dim=network_config.proj_vision_dim or network_config.emb_dim,
            llm_dim=network_config.proj_llm_dim or network_config.emb_dim,
        )
        self.state_encoder = StateEncoder(config=network_config)

        # --- Flow matching decoder: replaces the MLP ActionDecoder ---
        # action_dim_flat = chunk_size * action_dim  (the flow network operates on
        # the full flattened action chunk for each <halo_action> token position)
        action_dim_flat = network_config.action_chunk_size * network_config.action_dim
        self.flow_decoder = FlowActionDecoder(
            action_dim_flat=action_dim_flat,
            obs_dim=network_config.emb_dim,
            hidden_dim=network_config.flow_hidden_dim,
            time_embed_dim=network_config.flow_time_embed_dim,
        )
        # Store action shape info for reshape during sampling
        self.action_chunk_size = network_config.action_chunk_size
        self.action_dim = network_config.action_dim
        self.flow_num_ode_steps = network_config.flow_num_ode_steps
    

    def forward(self, images, input_ids, attention_mask, states):
        """
        Forward pass with interleaved multi-image, state-injection, and
        action-decoding support.

        The input_ids sequence contains three kinds of special tokens:
          • <image>  (image_token_id)  — replaced by ViT patch embeddings
          • <state>  (state_token_id)  — replaced by StateEncoder output
          • <halo_action> (action_token_id) — used *after* the transformer to
            extract hidden states for ActionDecoder

        Args:
            images:         [B, N_img, 3, H, W]  — N_img images per sample.
            input_ids:      [B, seq_len]          — token ids with <image>,
                            <state>, and <halo_action> placeholders.
            attention_mask: [B, seq_len]          — 1 for real tokens, 0 pad.
            states:         [B, N_state, state_dim] — proprioceptive states,
                            one per <state> token.

        Returns:
            logits:         [B, total_len, vocab_size] — next-token logits.
            action_hiddens: [B, n_action_tokens, emb_dim]
                            — transformer hidden states at <halo_action> positions,
                            used as conditioning for the flow matching decoder.
                            None if no <halo_action> tokens are present.
        """

        B = input_ids.size(0)
        device = input_ids.device
        network_config = self.config
        image_token_id = network_config.image_token_id    # e.g. 151665
        state_token_id = network_config.state_token_id    # e.g. 151667
        action_token_id = network_config.action_token_id  # e.g. 151666

        # ------------------------------------------------------------------ #
        # 1. IMAGES — count <image> tokens, encode each image through ViT,
        #    project into LLM space, and concatenate all patch features.
        #    img_proj → [B, N_img * num_patches, emb_dim]
        # ------------------------------------------------------------------ #
        num_images = (input_ids == image_token_id).sum(dim=1)  # [B]
        N_img = num_images.max().item()                        # scalar

        all_img_proj = []
        for i in range(N_img):
            img_i = images[:, i]                               # [B, 3, H, W]
            feat_i = self.vis_enc(img_i)                       # [B, num_patches, emb_dim]
            proj_i = self.image_projector(feat_i)              # [B, num_patches, emb_dim]
            all_img_proj.append(proj_i)

        img_proj = torch.cat(all_img_proj, dim=1)              # [B, N_img*num_patches, emb_dim]

        # ------------------------------------------------------------------ #
        # 2. STATES — count <state> tokens, encode each state vector through
        #    the StateEncoder MLP to get a single embedding per state.
        #    state_embeds → [B, N_state, emb_dim]
        # ------------------------------------------------------------------ #
        num_states = (input_ids == state_token_id).sum(dim=1)  # [B]
        N_state = num_states.max().item()                      # scalar

        all_state_embeds = []
        for i in range(N_state):
            state_i = states[:, i]                             # [B, state_dim]
            emb_i = self.state_encoder(state_i)                # [B, emb_dim]
            all_state_embeds.append(emb_i.unsqueeze(1))        # [B, 1, emb_dim]

        if all_state_embeds:
            state_embeds = torch.cat(all_state_embeds, dim=1)  # [B, N_state, emb_dim]
        else:
            state_embeds = torch.empty(B, 0, network_config.emb_dim, device=device)

        # ------------------------------------------------------------------ #
        # 3. TEXT EMBEDDINGS — embed all tokens, then zero out positions that
        #    correspond to <image> or <state> placeholders (their actual
        #    information comes from the encoders above).
        # ------------------------------------------------------------------ #
        special_mask = (input_ids != image_token_id) & (input_ids != state_token_id)
        text_embeds = self.token_emb(input_ids)                # [B, seq_len, emb_dim]
        text_embeds = text_embeds * special_mask.unsqueeze(-1)  # zero out placeholders

        # ------------------------------------------------------------------ #
        # 4. INJECT STATE EMBEDDINGS — replace the zeroed-out <state> token
        #    positions in text_embeds with the encoded state embeddings.
        #    This keeps states at their original position in the sequence
        #    (rather than prepending like images).
        # ------------------------------------------------------------------ #
        for b in range(B):
            state_positions = (input_ids[b] == state_token_id).nonzero(as_tuple=True)[0]
            for idx, pos in enumerate(state_positions):
                if idx < N_state:
                    text_embeds[b, pos] = state_embeds[b, idx]

        # ------------------------------------------------------------------ #
        # 5. CONCATENATE — prepend image patch features before the text+state
        #    sequence so the decoder can attend to visual context first.
        #    combined_embeds → [B, N_img*num_patches + seq_len, emb_dim]
        # ------------------------------------------------------------------ #
        combined_embeds = torch.cat([img_proj, text_embeds], dim=1)

        # ------------------------------------------------------------------ #
        # 6. POSITIONAL EMBEDDINGS — add learned positions over the full
        #    combined sequence.
        # ------------------------------------------------------------------ #
        total_len = combined_embeds.size(1)
        pos_emb = self.pos_embed(torch.arange(total_len, device=device)).unsqueeze(0)
        combined_embeds = combined_embeds + pos_emb

        # ------------------------------------------------------------------ #
        # 7. DECODER TRANSFORMER — causal self-attention + MoE FFN.
        # ------------------------------------------------------------------ #
        print(f"[HaloVLM] Sequence length to decoder: {combined_embeds.size(1)}")
        transformer_out = self.decoder_transformer(combined_embeds)
        transformer_out = self.layer_norm(transformer_out)

        # ------------------------------------------------------------------ #
        # 8. LANGUAGE HEAD — project every position to vocab logits.
        #    logits → [B, total_len, vocab_size]
        # ------------------------------------------------------------------ #
        logits = self.lm_head(transformer_out)

        # ------------------------------------------------------------------ #
        # 9. ACTION HIDDEN STATES — find <halo_action> token positions in the
        #    *original* input_ids (offset by the prepended image patches).
        #    Extract the transformer hidden states at those positions and
        #    return them as conditioning for the flow matching decoder.
        #
        #    action_hiddens → [B, n_action_tokens, emb_dim]
        #    Returns None if no <halo_action> tokens are present.
        # ------------------------------------------------------------------ #
        num_prepended = img_proj.size(1)  # N_img * num_patches
        action_mask = (input_ids == action_token_id)                # [B, seq_len]
        n_action_tokens = action_mask.sum(dim=1).max().item()  # scalar

        # ------------------------------------------------------------------ #
        # 9b. EXTRACT ACTION HIDDEN STATES — collect transformer hidden states
        #     at <halo_action> positions.  Instead of decoding actions directly
        #     with an MLP, we return these hidden states as *conditioning*
        #     for the flow matching decoder.  Flow matching loss is computed
        #     separately via  compute_flow_loss()  and actions are sampled
        #     at inference via  sample_actions().
        # ------------------------------------------------------------------ #
        action_hiddens = None
        if n_action_tokens > 0:
            action_hidden_list = []
            for b in range(B):
                action_positions = action_mask[b].nonzero(as_tuple=True)[0]  # in input_ids
                # Offset by prepended image patches to index into transformer_out
                action_positions = action_positions + num_prepended
                hiddens = transformer_out[b, action_positions]              # [n_act, emb_dim]
                # Pad if this sample has fewer <halo_action> tokens than the batch max
                if hiddens.size(0) < n_action_tokens:
                    pad = torch.zeros(
                        n_action_tokens - hiddens.size(0), network_config.emb_dim,
                        device=device,
                    )
                    hiddens = torch.cat([hiddens, pad], dim=0)
                action_hidden_list.append(hiddens)

            # action_hiddens: [B, n_action_tokens, emb_dim]
            # These serve as conditioning vectors for the flow decoder.
            action_hiddens = torch.stack(action_hidden_list, dim=0)
        else:
            import logging
            logging.warning("action_hiddens is None: No <halo_action> tokens found in input_ids.")

        return logits, action_hiddens

    # ------------------------------------------------------------------ #
    # Flow Matching — training loss
    #
    # Inspired by the flow matching formulation presented in:
    #   https://www.youtube.com/live/sW75XtmutfE?si=_ZKNQtiKPz4BxxZG
    #
    # The core idea follows the Conditional Flow Matching (CFM) framework,
    # but we added *conditioning on the transformer's hidden states* so
    # that the velocity field is aware of the full visual-language context
    # when predicting robot actions.
    # ------------------------------------------------------------------ #
    def compute_flow_loss(self, action_hiddens, target_actions, action_mask_seq):
        """
        Conditional Flow Matching (CFM) loss for action prediction.

        Given transformer hidden states at <halo_action> positions (conditioning)
        and ground-truth target actions, this method:
          1. Flattens each action token's target chunk into x_1.
          2. Samples t ~ U(0,1) and Gaussian noise x_0 ~ N(0, I).
          3. Constructs the linear interpolant  x_t = (1-t)*x_0 + t*x_1.
          4. Predicts the velocity field  v_theta(x_t, t, cond).
          5. Returns MSE loss between v_theta and the true velocity (x_1 - x_0).

        Args:
            action_hiddens : [B, n_act, emb_dim]  — conditioning from transformer
            target_actions : [B, T, action_dim]    — ground-truth action sequence
            action_mask_seq: [B, T]                — 1 for real timesteps, 0 for pad
        Returns:
            loss : scalar tensor — flow matching MSE loss
        """
        if action_hiddens is None:
            return torch.tensor(0.0, device=target_actions.device)

        B, n_act, D = action_hiddens.shape
        device = action_hiddens.device
        chunk = self.action_chunk_size
        act_dim = self.action_dim
        flat_dim = chunk * act_dim  # total dims per action token

        # --- Build x_1: target actions reshaped to [B*n_act, flat_dim] ---
        T_avail = target_actions.size(1)
        T_need = n_act * chunk
        # Truncate or pad target actions to match n_act * chunk_size
        if T_avail >= T_need:
            targets = target_actions[:, :T_need, :]   # [B, T_need, act_dim]
        else:
            pad = torch.zeros(B, T_need - T_avail, act_dim, device=device)
            targets = torch.cat([target_actions, pad], dim=1)
        # Reshape: [B, n_act, chunk*act_dim]
        x_1 = targets.view(B, n_act, flat_dim).view(B * n_act, flat_dim)

        # --- Condition vector: flatten to [B*n_act, emb_dim] ---
        # Each <halo_action> token's transformer hidden state becomes an
        # independent conditioning vector for the flow network.  By reshaping
        # [B, n_act, D] → [B*n_act, D] we treat every action-token position
        # as its own sample, so the flow decoder sees one (x_t, t, cond)
        # triple at a time.
        cond = action_hiddens.view(B * n_act, D)

        # --- Sample t ~ Uniform(0, 1) per element ---
        # Each of the B*n_act samples gets its own random "time" along the
        # flow path.  t=0 corresponds to pure noise, t=1 to clean data.
        # Drawing independent t values per sample provides diverse gradient
        # signal across the full [0,1] interval in a single batch.
        t = torch.rand(B * n_act, device=device)

        # --- Sample noise x_0 ~ N(0, I) ---
        # x_0 is the source distribution (isotropic Gaussian).  At inference
        # time we will also start from this distribution and integrate the
        # learned velocity field forward to t=1 to recover clean actions.
        x_0 = torch.randn_like(x_1)

        # --- Optimal-transport linear interpolant: x_t = (1-t)*x_0 + t*x_1 ---
        # This is the *conditional* OT path between x_0 (noise) and x_1 (data).
        # At t=0 we are at pure noise x_0; at t=1 we arrive at the ground-truth
        # action x_1.  The straight-line interpolation is the optimal transport
        # displacement map for Gaussian → point-mass, and its derivative w.r.t. t
        # is the constant velocity  v* = x_1 - x_0.
        t_expand = t.unsqueeze(-1)                     # [B*n_act, 1]
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1  # [B*n_act, flat_dim]

        # --- Predict velocity field v_theta(x_t, t, cond) ---
        # The flow decoder receives the noisy action x_t, the scalar time t
        # (embedded via sinusoidal frequencies inside the decoder), and the
        # transformer conditioning vector.  It outputs a velocity vector of
        # the same dimensionality as x_t, representing its estimate of the
        # direction and magnitude needed to move from noise toward data.
        v_pred = self.flow_decoder(x_t, t, cond)       # [B*n_act, flat_dim]

        # --- Ground-truth velocity: dx/dt = x_1 - x_0 (constant for OT path) ---
        # Because the interpolant is linear in t, its time-derivative is just
        # the displacement vector (x_1 - x_0), constant for all t.  This is
        # what we supervise the network to predict.
        v_target = x_1 - x_0

        # --- MSE loss ---
        # Mean squared error between the predicted and true velocity fields.
        # Minimising this objective trains the flow decoder to learn the
        # conditional vector field that transports N(0,I) to the action
        # distribution conditioned on the transformer's representation.
        loss = F.mse_loss(v_pred, v_target)
        return loss

    # ------------------------------------------------------------------ #
    # Flow Matching — Euler ODE sampling (inference)
    #
    # During training we taught the flow decoder to predict the velocity
    # field that moves probability mass from Gaussian noise toward the
    # real action distribution.  At inference we *reverse-engineer* the
    # actual actions by starting from noise and following that learned
    # velocity field forward in time, like tracing a river from its
    # source to its mouth.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_actions(self, action_hiddens, num_steps=None):
        """
        Generate actions by integrating the learned velocity field from t=0 to t=1
        using the Euler method.

        Starting from Gaussian noise x_0 ~ N(0, I), we iteratively update:
            x_{t+dt} = x_t + v_theta(x_t, t, cond) * dt
        until we reach t=1, which gives the predicted clean action chunk.

        Args:
            action_hiddens : [B, n_act, emb_dim] — conditioning from transformer
            num_steps      : int — number of Euler integration steps (default
                             from config.flow_num_ode_steps)
        Returns:
            action_preds : [B, n_act, chunk_size, action_dim]
        """
        # If the transformer didn't produce any <halo_action> hidden states
        # (e.g. the prompt has no action tokens), there's nothing to decode.
        if action_hiddens is None:
            return None

        # Fall back to the default number of ODE integration steps from
        # config if the caller didn't specify one.  More steps = more
        # accurate but slower; 20 is a good default for Euler.
        if num_steps is None:
            num_steps = self.flow_num_ode_steps

        B, n_act, D = action_hiddens.shape
        device = action_hiddens.device
        # flat_dim is the total number of scalars in one action chunk,
        # e.g. 16 timesteps * 7 DOF = 112.  The flow network works in
        # this flattened space and we reshape back at the end.
        flat_dim = self.action_chunk_size * self.action_dim

        # Each <halo_action> token position is treated as an independent
        # sample.  We flatten [B, n_act, emb_dim] → [B*n_act, emb_dim]
        # so every action-token gets its own conditioning vector — the
        # same way we handled it during training.
        cond = action_hiddens.view(B * n_act, D)

        # This is where generation begins: pure random noise, the same
        # distribution we used as x_0 during training.  Think of it as
        # a blank canvas that the velocity field will sculpt into a
        # meaningful action trajectory over the next num_steps iterations.
        x = torch.randn(B * n_act, flat_dim, device=device)

        # We integrate the ODE  dx/dt = v_theta(x, t, cond)  from t=0
        # to t=1 using the simplest possible solver: forward Euler.
        # dt is the time increment per step — with 20 steps, each step
        # advances time by 0.05.
        dt = 1.0 / num_steps
        for step_i in range(num_steps):
            # Current time along the flow.  At step 0 we're at t=0
            # (pure noise); by the last step we're near t=1 (clean data).
            t_val = step_i * dt
            # Broadcast the scalar time to every sample in the batch
            t = torch.full((B * n_act,), t_val, device=device)

            # Ask the flow decoder: "given this partially-denoised action
            # x at time t, and this transformer context, which direction
            # and how far should I move?"  The sinusoidal time embedding
            # inside the decoder tells it *where* it is on the noise→data
            # continuum, so it knows how aggressively to push.
            v = self.flow_decoder(x, t, cond)

            # Take one Euler step: move x a little bit in the predicted
            # velocity direction.  After all num_steps of these small
            # nudges, x will have traveled from noise to (approximately)
            # the clean action that the model thinks the robot should take.
            x = x + v * dt

        # The integration is done — x is now our best estimate of the
        # clean action chunk in flattened form [B*n_act, flat_dim].
        # Reshape it back to the structured [B, n_act, chunk_size, action_dim]
        # so downstream code can interpret each timestep × DOF independently.
        action_preds = x.view(B, n_act, self.action_chunk_size, self.action_dim)
        return action_preds