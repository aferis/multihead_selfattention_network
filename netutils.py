from rllib.functions import networks
from rllib.classes.errors.ConfigurationError import ConfigurationError
import tensorflow as tf



def create_basic_input(num_self_features,
                       num_job_features,
                       num_jobs):
    """
    Basic feature inputs without embeddings. All input variables are processed as a single vector.
    """

    # Create self input and embeddings
    self_input, self_output = networks.build_standard_input(num_input_vars=num_self_features,
                                                            input_name="plant_machine",
                                                            num_goal_input_vars=0)
    self_output = tf.expand_dims(self_output, axis=1)


    # Create job input and embeddings
    feature_input, feature_output = networks.build_2D_shared_feature_input(num_input_vars=num_job_features,
                                                                           input_name="job_feature",
                                                                           num_inputs=num_jobs,
                                                                           hidden_layers=[])

    # Concatenate self with jobs
    concat_output = tf.keras.layers.concatenate([self_output, feature_output], axis=1)

    # Concatenate all jobs into single dimension
    shape = networks.get_shape_list(concat_output)
    output = tf.reshape(concat_output, (shape[0], shape[1]*shape[2]))

    inputs = [*self_input, feature_input]

    return inputs, output



def create_standard_feature_network(num_self_features,
                                    num_self_embd_layers,
                                    num_job_features,
                                    num_jobs,
                                    num_job_embd_layers,
                                    embd_activation="relu",
                                    layer_normalization=False):
    """
    Creates feature inputs with embeddings that are shared across jobs. The embeddings are then
    concatenated into a single vector.

    """

    # Create self input and embeddings
    self_input, self_output = networks.build_standard_input(num_input_vars=num_self_features,
                                                            input_name="plant_machine",
                                                            num_goal_input_vars=0)

    self_output = networks.add_fully_connected_network(self_output,
                                                       num_output_vars=None,
                                                       hidden_layers=num_self_embd_layers,
                                                       hidden_layer_activation=embd_activation,
                                                       layer_normalization=layer_normalization)

    self_output = tf.expand_dims(self_output, axis=1)


    # Create job input and embeddings
    feature_input, feature_output = networks.build_2D_shared_feature_input(num_input_vars=num_job_features,
                                                                           input_name="job_feature",
                                                                           num_inputs=num_jobs,
                                                                           hidden_layers=num_job_embd_layers,
                                                                           hidden_layer_activation=embd_activation,
                                                                           layer_normalization=layer_normalization)

    # Concatenate self with jobs
    concat_output = tf.keras.layers.concatenate([self_output, feature_output], axis=1)

    # Concatenate all jobs into single dimension
    shape = networks.get_shape_list(concat_output)
    output = tf.reshape(concat_output, (shape[0], shape[1]*shape[2]))

    inputs = [*self_input, feature_input]

    return inputs, output



def create_multihead_selfattention_network(num_self_features,
                                           num_embd_layers,
                                           num_job_features,
                                           num_jobs,
                                           num_attention_heads,
                                           num_post_sa_layer,
                                           embd_activation="relu",
                                           post_sa_activation="linear",
                                           job_mask_dim=None,
                                           layer_normalization=False,
                                           pre_sa_layer_normalization=False,
                                           post_sa_layer_normalization=False,
                                           post_sa_residual_connection=False,
                                           pooling_layer=None,
                                           pooling_mask=False):
    """
    Creates feature inputs for plant state, action buffer and action mask. The features of the plant state and the action
    buffer are embedded separately and then concatenated into a single tensor. The action mask is reshaped and passed
    - alongside the concatenated plant state features and job features - through the multi-headed self-attention block.
    Finally, the output of the multi-headed self-attention is pooled.

    """

    # Create self input and embeddings
    self_input, self_output = networks.build_standard_input(num_input_vars=num_self_features,
                                                            input_name="plant_machine",
                                                            num_goal_input_vars=0)

    self_output = networks.add_fully_connected_network(self_output,
                                                       num_output_vars=None,
                                                       hidden_layers=num_embd_layers,
                                                       hidden_layer_activation=embd_activation,
                                                       layer_normalization=layer_normalization)

    self_output = tf.expand_dims(self_output, axis=1)

    # Create job input and action embeddings
    feature_input, feature_output = networks.build_2D_shared_feature_input(num_input_vars=num_job_features,
                                                                           input_name="job_feature",
                                                                           num_inputs=num_jobs,
                                                                           hidden_layers=num_embd_layers,
                                                                           hidden_layer_activation=embd_activation,
                                                                           layer_normalization=layer_normalization)

    # Create an action mask input and add a padding mask for self features
    job_mask_input = []
    mask_concat = None
    attention_mask = None

    if job_mask_dim is not None:
        job_mask_input, job_mask_output = networks.build_standard_input(num_input_vars=job_mask_dim,
                                                                        input_name="action_mask",
                                                                        num_goal_input_vars=0)

        # Create a one dimensional self mask with a value of 1 and then concatinate the self mask and the job mask
        bs = tf.shape(job_mask_output)[0]
        self_mask = tf.ones([bs, 1])
        mask_concat = tf.keras.layers.concatenate([self_mask, job_mask_output])     # Shape = [B, S]
        mask_shape = networks.get_shape_list(mask_concat)[-1]


    # Save the inputs of the network into a list
    inputs = [*self_input, feature_input, *job_mask_input]

    # Concatenate self with jobs
    concat_output = tf.keras.layers.concatenate([self_output, feature_output], axis=1)

    # Normalize embeddings prior to computing query, key and value tensors
    if pre_sa_layer_normalization is True:
        inp_qkv = tf.keras.layers.LayerNormalization()(concat_output)
    else:
        inp_qkv = concat_output

    # Get the last embedding dimension for creating query, key and value
    n_embd_last = [num_embd_layers[-1]]

    # Creating the tensors query [B, T, dim], key [B, S, dim] and value [B, S, dim]
    # B = Batch Size, T = Target, S = Source, dim = Feature dimensions
    query = networks.add_fully_connected_network(inp_qkv,
                                                 num_output_vars=None,
                                                 hidden_layers=n_embd_last,
                                                 hidden_layer_activation="linear",
                                                 layer_normalization=layer_normalization,
                                                 residual_connection=False)

    key = networks.add_fully_connected_network(inp_qkv,
                                               num_output_vars=None,
                                               hidden_layers=n_embd_last,
                                               hidden_layer_activation="linear",
                                               layer_normalization=layer_normalization,
                                               residual_connection=False)

    value = networks.add_fully_connected_network(inp_qkv,
                                                 num_output_vars=None,
                                                 hidden_layers=n_embd_last,
                                                 hidden_layer_activation="linear",
                                                 layer_normalization=layer_normalization,
                                                 residual_connection=False)


    # Get key dimensions (same as query- and value dimensions) for the multi-head attention layer
    # Number of the last embedding layer must be divisible by the number of attention heads!
    if n_embd_last[0] % num_attention_heads == 0:
        key_dim = int(n_embd_last[0] / num_attention_heads)
    else:
        raise ConfigurationError("Key and value depth must be divisible by the number of attention heads!")

    # Create an instance of multi-head attention layer
    attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=num_attention_heads,
                                                         key_dim=key_dim,
                                                         value_dim=key_dim,
                                                         use_bias=True)

    # Create an attention mask of shape [B, T, S].
    # If the attention is a self-attention (the dimensions of query, key and value should be the same), then T = S.
    if query.get_shape().as_list() == key.get_shape().as_list() == value.get_shape().as_list() and mask_concat is not None:
        attention_mask = tf.stack([mask_concat for _ in range(mask_shape)], axis=-1)    # Shape = [B, S, T]
        attention_mask = tf.transpose(attention_mask, perm=[0,2,1])                     # Shape = [B, T, S]
    else:
        attention_mask = None

    # Pass query, key, value and attention mask through the multi-head attention layer
    attention_output, attention_scores = attention_layer(query=query, value=value, key=key,
                                                         attention_mask=attention_mask,
                                                         return_attention_scores=True)

    # Post self-attention fully connected layer
    if num_post_sa_layer > 0:
        attention_output = networks.add_fully_connected_network(attention_output,
                                                                num_output_vars=None,
                                                                hidden_layers=n_embd_last,
                                                                hidden_layer_activation=post_sa_activation,
                                                                layer_normalization=layer_normalization)

    # Residual connection between the inputs and the outputs of the multi-head attention layer
    if post_sa_residual_connection is True:
        attention_output = concat_output + attention_output

    # If available, create more post self-attention layers and also add a residual connection from after the first post self-attention layer to after the last post self-attention layer
    if num_post_sa_layer > 1:
        mlp = attention_output
        for _ in range(num_post_sa_layer - 1):
            mlp = networks.add_fully_connected_network(mlp,
                                                       num_output_vars=None,
                                                       hidden_layers=n_embd_last,
                                                       hidden_layer_activation=post_sa_activation,
                                                       layer_normalization=layer_normalization)
        attention_output = attention_output + mlp

    # Normalize the residually connected inputs and outputs of the multi-head attention layer
    if post_sa_layer_normalization is True:
        attention_output = tf.keras.layers.LayerNormalization()(attention_output)

    # Masking and then pooling of the entities
    if pooling_layer is not None:
        if pooling_layer == "average":
            if mask_concat is not None and pooling_mask is True:
                entity_mask = tf.expand_dims(mask_concat, -1)               # Expand dimensions
                attention_output_masked = attention_output * entity_mask    # Mask the attention output
                summed = tf.reduce_sum(attention_output_masked, -2)         # Pooling of the masked attention output
                denom = tf.reduce_sum(entity_mask, -2) + 1e-5               # Pooling of the attention mask
            else:
                bs = tf.shape(attention_output)[0]                          # Take bs = Batch size
                NE = attention_output.get_shape().as_list()[1]              # Take NE = Number of entities
                summed = tf.reduce_sum(attention_output, -2)                # Pooling of the unmasked attention output
                denom = tf.cast(tf.fill([bs,1], NE), tf.float32)            # Denominator to devide "summed" by NE
            entity_avg_pooling = summed / denom                      # Average pooling of the entities
            output = entity_avg_pooling                              # Output average pooled entities
        elif pooling_layer == "maximum":
            if mask_concat is not None and pooling_mask is True:
                entity_mask = tf.expand_dims(mask_concat, -1)                                           # Expand dimensions
                has_unmasked_entities = tf.sign(tf.reduce_sum(entity_mask, axis=-2, keepdims=True))     # = 0, if the entity mask only consists of zeros (= 1 otherwise)
                offset = (entity_mask - 1) * 1e9                                                        # Create an offset of -inf to eliminate the entities which should be masked
                attention_output_masked = (attention_output + offset) * has_unmasked_entities           # Mask the attention output
                attention_output_to_pool = attention_output_masked
            else:
                attention_output_to_pool = attention_output
            entity_max_pooling = tf.reduce_max(attention_output_to_pool, -2)                            # Maximum pooling of the masked entities
            output = entity_max_pooling
        else:
            pass
    else:
        shape = networks.get_shape_list(attention_output)
        output = tf.reshape(attention_output, (shape[0], shape[1] * shape[2]))


    return inputs, output