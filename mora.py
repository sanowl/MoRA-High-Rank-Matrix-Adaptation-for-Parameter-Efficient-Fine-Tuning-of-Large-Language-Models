import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import logging

logging.basicConfig(level=logging.INFO)

class MoRA(nn.Module):
    hidden_size: int
    rank: int
    group_type: int = 0

    def setup(self):
        self.matrix = self.param('matrix', jax.random.uniform, (self.rank, self.rank))

    def __call__(self, x):
        compressed_x = self.compress(x)
        output = jnp.dot(compressed_x, self.matrix)
        return self.decompress(output)

    def compress(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_padded = jnp.pad(x, ((0, 0), (0, 0), (0, self.hidden_size - hidden_size)))

        if self.group_type == 0:
            compressed_x = x_padded.reshape(batch_size, seq_len, -1, self.rank).sum(axis=2)
        else:
            compressed_x = x_padded.reshape(batch_size, seq_len, self.rank, -1).sum(axis=3)

        return compressed_x

    def decompress(self, x):
        batch_size, seq_len, rank_size = x.shape

        if self.group_type == 0:
            decompressed_x = jnp.repeat(x, self.hidden_size // self.rank, axis=2)
        else:
            decompressed_x = jnp.tile(x, (1, 1, self.hidden_size // self.rank))

        decompressed_x = decompressed_x[:, :, :self.hidden_size]

        return decompressed_x

    def change_group_type(self):
        self.group_type = 1 - self.group_type

class MoRALinear(nn.Module):
    in_features: int
    out_features: int
    rank: int
    group_type: int = 0
    use_bias: bool = True

    def setup(self):
        self.weight = self.param('weight', jax.random.uniform, (self.out_features, self.in_features))
        self.bias = self.param('bias', jax.random.zeros, (self.out_features,)) if self.use_bias else None
        self.mora = MoRA(self.in_features, self.rank, self.group_type)

    def __call__(self, x):
        x = self.mora(x)
        output = jnp.dot(x, self.weight.T)
        if self.use_bias:
            output += self.bias
        return output

    def change_group_type(self):
        self.mora.change_group_type()

    def merge_weights(self):
        weight = self.weight
        if self.mora.group_type == 0:
            weight_merged = weight.reshape(self.out_features // self.rank, self.rank, -1).transpose(1, 0, 2).reshape(self.out_features, -1)
        else:
            weight_merged = weight.reshape(self.rank, self.out_features // self.rank, -1).transpose(1, 0, 2).reshape(self.out_features, -1)
        self.weight = weight_merged
        self.mora.matrix = jax.random.uniform(jax.random.PRNGKey(0), (self.mora.rank, self.mora.rank))

def apply_mora_linear(model):
    for attr_name in dir(model):
        attr = getattr(model, attr_name)
        if isinstance(attr, nn.Dense):
            setattr(model, attr_name, MoRALinear(attr.features, attr.kernel.shape[0], rank=128, group_type=0, use_bias=attr.bias is not None))

def merge_and_reset_mora_linear(model):
    for attr_name in dir(model):
        attr = getattr(model, attr_name)
        if isinstance(attr, MoRALinear):
            attr.merge_weights()
            attr.change_group_type()

def update_model_mora_linear(state, model, step, merge_steps):
    if step % merge_steps == 0:
        merge_and_reset_mora_linear(model)
        state = state.replace(apply_fn=model.apply, params=model.init(jax.random.PRNGKey(0), jnp.ones((1, 10))).unfreeze())

    return state

class YourModel(nn.Module):
    def setup(self):
        self.dense1 = nn.Dense(768)
        self.bn1 = nn.BatchNorm(use_running_average=True)
        self.dense2 = nn.Dense(768)
        self.bn2 = nn.BatchNorm(use_running_average=True)
        self.dense3 = nn.Dense(10)

    def __call__(self, x):
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dense3(x)
        return x

def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones((1, 50, 768)))['params']
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate),
        optax.exponential_decay(learning_rate, transition_steps=100, decay_rate=0.99)
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch, target):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch)
        loss = jnp.mean((logits - target) ** 2)
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state

# Initialize and apply MoRA to the model
model = YourModel()
apply_mora_linear(model)

# Create train state
rng = jax.random.PRNGKey(0)
state = create_train_state(rng, model, learning_rate=0.001)
merge_steps = 1000
num_steps = 2000
input_data = jnp.random.randn(32, 50, 768)
target = jnp.random.randn(32, 10)

# Training loop
for step in range(num_steps):
    try:
        state = train_step(state, input_data, target)
        state = update_model_mora_linear(state, model, step, merge_steps)

        if step % 100 == 0:
            loss = jnp.mean((state.apply_fn({'params': state.params}, input_data) - target) ** 2)
            logging.info(f'Step {step}, Loss: {loss}')
    except Exception as e:
        logging.error(f"Error during training step {step}: {e}")
        raise
