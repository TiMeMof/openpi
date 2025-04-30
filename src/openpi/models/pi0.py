import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # 创建并初始化语言模型
        # TODO: 用NNX重写Gemma，目前使用桥接
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")

        # 创建并初始化图像模型
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        # 组合LLM和图像模型为PaLI-Gemma多模态模型
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        
        # 状态投影层：将机器人状态投影到模型维度
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        
        # 动作输入投影层：将动作投影到模型维度
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        
        # 动作-时间MLP输入层：将连接的动作和时间特征投影到模型维度
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        
        # 动作-时间MLP输出层
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        
        # 动作输出投影层：将模型输出投影回动作维度
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """嵌入前缀部分（图像和文本），图像通过SigLip模型编码，文本通过Gemma LLM编码，皆为双向注意力，用ar_mask = false表示"""
        input_mask = []
        ar_mask = []
        tokens = []
        # 嵌入图像
        for name in obs.images:
            # 通过图像模型获取图像token
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            # 添加图像token
            tokens.append(image_tokens)
            # 重复图像掩码以匹配token维度
            # 将图像掩码扩展到与图像tokens相同的序列长度，使用einops.repeat进行形状变换，
            # 这些掩码会指示哪些图像是有效的，而哪些是填充的
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s", # 调整形状：批次维度保持不变，添加序列维度
                    s=image_tokens.shape[1], # 序列长度等于图像token数
                )
            )
            # 设置图像tokens之间的注意力为双向(False表示双向注意力)，原因在于图像内容通常是非时序性的数据
            # 为什么非时序性的数据可以使用双向注意力？因为图像数据通常是静态的，不具有时间序列的特性，
            # 因此可以在处理时考虑全局上下文信息，而不需要限制为单向的时间序列处理。
            ar_mask += [False] * image_tokens.shape[1]

        # 使用LLM模型对文本输入 tokenized_inputs 进行嵌入 
        if obs.tokenized_prompt is not None:# 通过语言模型嵌入分词后的提示
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)# 添加文本token
            input_mask.append(obs.tokenized_prompt_mask)# 添加提示掩码
            # 同样设置为双向注意力，相当于语言token可以关注图像token，图像token反过来亦可关注语言token，最终实现多模态融合
            ar_mask += [False] * tokenized_inputs.shape[1]
#     连接所有token和掩码，其中包含了
# ->  多模态信息的融合表示tokens——图像token和语言token
# ->  以及指示哪些token是有效信息的 input_mask
# ->  和如何在这些token之间进行注意力计算规则的 ar_mask ——相当于控制信息流动的方向 
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask


    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """嵌入后缀部分（状态和动作），处理机器人状态信息q_t、噪声化的动作信息noise(状态和噪声动作经过线性投影和MLP处理)，创建后缀 token"""
        input_mask = [] # 
        ar_mask = []    # 自回归掩码
        tokens = []     # tokens 列表
        # 添加单个状态 token
        state_token = self.state_proj(obs.state)[:, None, :] # 投影状态并且添加一个维度
        tokens.append(state_token) # 添加状态 token
        # 添加状态掩码（全为1），表示这个状态token是有效的
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # 设置为单向注意力(True)，表明图像和语言输入不能关注状态信息
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # 时间步嵌入，使用正弦-余弦位置编码生成时间步嵌入，敏感度范围为[0, 1]
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        ''' 动作和时间信息融合，比如通过action_time_tokens连接：「带噪声的动作」和「时间token」'''
        # mix timestep + action information using an MLP
        # 混合时间步 + 动作信息，使用MLP
        action_tokens = self.action_in_proj(noisy_actions)# 投影带噪声的动作
        
        # 重复时间嵌入以匹配动作序列长度
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        
        # 连接动作和时间token
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        
        ''' MLP 处理 '''
        # 两层MLP和swish激活函数对「动作和时间的组合表示」进行非线性变换，以进一步融合：(噪声)动作和时间信息 
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)    # 输入层
        action_time_tokens = nnx.swish(action_time_tokens)                  # swish激活函数
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)   # 输出层
        
        # 添加动作-时间 token
        tokens.append(action_time_tokens)
        # 添加动作-时间掩码（全为1），表示这个动作-时间 token 是有效的
        # 这里的掩码是为了指示哪些 token 是有效的，通常情况下，动作-时间 token 是有效的
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        # 图像/语言/状态输入不关注动作token，设置为单向注意力(True)
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        
        # 连接所有token和掩码
        tokens = jnp.concatenate(tokens, axis=1)        # 在序列维度上连接所有token
        input_mask = jnp.concatenate(input_mask, axis=1)# 在序列维度上连接所有掩码
        ar_mask = jnp.array(ar_mask)                    # 将自回归掩码转换为布尔数组
        return tokens, input_mask, ar_mask  # 返回tokens、输入掩码和自回归掩码  

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """计算扩散模型的损失函数"""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # 分割随机数生成器为三部分，用于不同的随机操作
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # 获取动作的批次形状
        batch_shape = actions.shape[:-2]
        # 生成与动作形状相同的噪声
        noise = jax.random.normal(noise_rng, actions.shape)
        # jax.random.beta(key, a, b, shape) 
        # beta分布：x^(a-1) * (1-x)^(b-1)，x ~ Beta(a, b)，在[0, 1]区间内
        # Beta(1.5, 1)偏向较低的值
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        # 拓展时间维度以匹配动作形状
        time_expanded = time[..., None, None]
        # x_t是噪声化的动作，随着时间从 0 到 1 ，原始动作action逐渐添加真实噪声u_t，变为纯噪声 noise
        # 而 u_t 代表所加的真实噪声，便是咱们所要预测噪声 v_t 的ground truth
        # 创建带噪声的动作 x_t = t*noise + (1 - t)*actions
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        # 计算目标噪声 u_t = noise - actions, 这是模型需要预测的目标
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        # 通过链接前缀和后缀的掩码，创建完整的输入序列
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        # 创建注意力掩码，从而控制不同token之间的可见性
        attn_mask = make_attn_mask(input_mask, ar_mask)
        # 计算位置编码
        positions = jnp.cumsum(input_mask, axis=1) - 1
        # 通过PaLI-Gemma模型处理tokens，得到前缀和后缀的输出
        # 这里的prefix_out是None，因为我们只关心后缀的输出
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )

        # 预测噪声v_t，将模型输出的后缀部分投影到动作维度
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # 返回预测噪声和真实噪声之间的均方误差
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,                       # 随机数生成器
        observation: _model.Observation,            # 观察数据，包括文本和图像
        *,
        num_steps: int | at.Int[at.Array, ""] = 10, # 扩散步骤数
    ) -> _model.Actions:                            # 返回生成的动作
        """基于扩散模型逆向采样(即去噪)，生成机器人动作序列 """
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        # 注意：这里使用扩散模型文献中更常见的约定，t=1是噪声，t=0是目标分布
        # 这与pi0论文相反
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        # 生成初始噪声，形状为[批次大小, 动作序列长度, 动作维度]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        # 通过前缀部分的前向传播填充 Key-Value 缓存
        # 获取前缀部分的的tokens、掩码和自回归掩码
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        # 创建前缀部分的注意力掩码，形状为[批次大小, 前缀长度, 前缀长度]，指示前缀tokens之间的注意力
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        # 计算位置编码
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # 通过PaLI-Gemma模型处理前缀tokens，得到前缀部分的输出和KV缓存
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            """定义单步去噪函数"""
            x_t, time = carry # carry数组包含当前状态和时间
            # 将时间广播到批次大小，并且嵌入后缀（批次和动作）
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            """构建复杂的注意力掩码系统，处理前缀-后缀之间的注意力关系
            ——这个复杂的掩码系统允许后缀token（包括状态和动作）有选择地关注前缀token（图像和文本），
            实现了条件生成"""
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            # 创建后缀对后缀的注意力掩码，形状为[批次, 后缀长度, 后缀长度]
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            # 创建后缀对前缀的注意力掩码，形状为[批次, 后缀长度, 前缀长度]
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            # 组合掩码，形状为[批次, 后缀长度, 前缀长度 + 后缀长度]
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            # 计算位置编码，为后缀token计算其在完整序列中的位置
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # 使用KV缓存进行高效的前向传递
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            # 确保前缀输出为None，因为使用了KV缓存
            assert prefix_out is None
            # 这里action_horizon是50，表示动作序列长度，
            # suffix_out[:, -self.action_horizon :]表示取后缀输出的最后50个token
            # action_out_proj是一个线性层，将后缀输出投影到动作维度
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # v_t是预测的噪声，dt这里是负的，表示时间从1到0
            # 所以这里 x_t+dt*v_t 是去噪后的动作
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            # 迭代条件：时间大于等于-dt/2，注：这里的dt是负数
            return time >= -dt / 2

        # 使用while循环进行迭代采样，从t=1（噪声）开始
        # carry首次为噪声noise和1.0（时间）
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
