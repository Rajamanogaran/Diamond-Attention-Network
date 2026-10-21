import tensorflow as tf
from tensorflow.keras import layers, models, Model
import numpy as np


def create_diamond_mask(kernel_size):
    """Create diamond-shaped mask for kernels"""
    mask = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            if abs(i - center) + abs(j - center) <= center:
                mask[i, j] = 1.0
    return mask


class DiamondConv2D(layers.Layer):
    """
    Diamond-shaped convolution layer with attention-based weight sharing.
    
    Fixed:
    - Removed built-in ReLU (let the caller decide activation)
    - Proper weight initialization accounting for diamond mask sparsity
    - Better attention mechanism with temperature scaling
    - Diversity loss hook to prevent kernel collapse
    """
    def __init__(self, filters, kernel_size=5, num_kernels=4, 
                 use_attention=True, **kwargs):
        super(DiamondConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.use_attention = use_attention
        self.diamond_mask = create_diamond_mask(kernel_size)
        self.diamond_mask_tensor = None

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if input_channels is None:
            raise ValueError("Input channels must be known at build time")

        self.input_channels = input_channels

        self.diamond_mask_tensor = tf.constant(
            self.diamond_mask.reshape(self.kernel_size, self.kernel_size, 1, 1),
            dtype=tf.float32
        )

        # Calculate effective fan_in for proper initialization
        # Diamond mask reduces the number of active weights
        active_pixels = float(np.sum(self.diamond_mask))
        fan_in = active_pixels * input_channels
        fan_out = active_pixels * self.filters
        # Glorot/Xavier scale adjusted for diamond mask
        init_scale = np.sqrt(2.0 / (fan_in + fan_out))

        self.kernels = []
        self.biases = []

        for i in range(self.num_kernels):
            kernel = self.add_weight(
                name=f'diamond_kernel_{i}',
                shape=(self.kernel_size, self.kernel_size, 
                       input_channels, self.filters),
                initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=init_scale
                ),
                trainable=True
            )
            self.kernels.append(kernel)

            bias = self.add_weight(
                name=f'diamond_bias_{i}',
                shape=(self.filters,),
                initializer='zeros',
                trainable=True
            )
            self.biases.append(bias)

        if self.use_attention and self.num_kernels > 1:
            # Better attention: use both channel stats and a learned temperature
            reduced_dim = max(input_channels // 4, 8)
            self.att_fc1 = self.add_weight(
                name='att_fc1',
                shape=(input_channels * 2, reduced_dim),  # avg + max
                initializer='glorot_uniform',
                trainable=True
            )
            self.att_bias1 = self.add_weight(
                name='att_bias1',
                shape=(reduced_dim,),
                initializer='zeros',
                trainable=True
            )
            self.att_fc2 = self.add_weight(
                name='att_fc2',
                shape=(reduced_dim, self.num_kernels),
                initializer='glorot_uniform',
                trainable=True
            )
            self.att_bias2 = self.add_weight(
                name='att_bias2',
                shape=(self.num_kernels,),
                initializer='zeros',
                trainable=True
            )
            # Learnable temperature for softmax sharpness
            self.temperature = self.add_weight(
                name='temperature',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True,
                constraint=tf.keras.constraints.MinMaxNorm(
                    min_value=0.1, max_value=10.0
                )
            )

        self.built = True

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]

        conv_outputs = []
        for kernel, bias in zip(self.kernels, self.biases):
            masked_kernel = kernel * self.diamond_mask_tensor
            output = tf.nn.conv2d(
                inputs, masked_kernel,
                strides=[1, 1, 1, 1], padding='SAME'
            )
            output = tf.nn.bias_add(output, bias)
            conv_outputs.append(output)

        # [batch, H, W, filters, num_kernels]
        stacked = tf.stack(conv_outputs, axis=-1)

        if self.use_attention and self.num_kernels > 1:
            avg_pool = tf.reduce_mean(inputs, axis=[1, 2])  # [batch, C]
            max_pool = tf.reduce_max(inputs, axis=[1, 2])   # [batch, C]
            pooled = tf.concat([avg_pool, max_pool], axis=-1)  # [batch, 2C]

            att = tf.matmul(pooled, self.att_fc1) + self.att_bias1
            att = tf.nn.relu(att)
            att = tf.matmul(att, self.att_fc2) + self.att_bias2
            # Temperature-scaled softmax
            att = att / self.temperature
            att_weights = tf.nn.softmax(att, axis=-1)  # [batch, num_kernels]

            att_weights = tf.reshape(
                att_weights, [batch_size, 1, 1, 1, self.num_kernels]
            )
            output = tf.reduce_sum(stacked * att_weights, axis=-1)
        else:
            output = tf.reduce_mean(stacked, axis=-1)

        # NO activation here - let the block handle it
        return output

    def get_kernel_diversity_loss(self):
        """
        Compute diversity loss to prevent all kernels from collapsing
        to the same weights. Call this during training and add to total loss.
        """
        if self.num_kernels <= 1:
            return 0.0

        # Compare masked kernels pairwise using cosine similarity
        masked_kernels = []
        for kernel in self.kernels:
            mk = kernel * self.diamond_mask_tensor
            mk_flat = tf.reshape(mk, [-1])
            mk_norm = mk_flat / (tf.norm(mk_flat) + 1e-8)
            masked_kernels.append(mk_norm)

        similarity_loss = 0.0
        count = 0
        for i in range(len(masked_kernels)):
            for j in range(i + 1, len(masked_kernels)):
                cos_sim = tf.reduce_sum(masked_kernels[i] * masked_kernels[j])
                similarity_loss += tf.square(cos_sim)
                count += 1

        return similarity_loss / max(count, 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'num_kernels': self.num_kernels,
            'use_attention': self.use_attention
        })
        return config


class ChannelAttention(layers.Layer):
    """Channel attention module (CBAM-style)"""
    def __init__(self, channels, reduction_ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        reduced = max(self.channels // self.reduction_ratio, 4)
        self.fc1_w = self.add_weight('fc1_w', (self.channels, reduced),
                                      initializer='glorot_uniform')
        self.fc1_b = self.add_weight('fc1_b', (reduced,),
                                      initializer='zeros')
        self.fc2_w = self.add_weight('fc2_w', (reduced, self.channels),
                                      initializer='glorot_uniform')
        self.fc2_b = self.add_weight('fc2_b', (self.channels,),
                                      initializer='zeros')
        self.built = True

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2])
        max_pool = tf.reduce_max(inputs, axis=[1, 2])

        def shared_mlp(x):
            x = tf.matmul(x, self.fc1_w) + self.fc1_b
            x = tf.nn.relu(x)
            x = tf.matmul(x, self.fc2_w) + self.fc2_b
            return x

        attention = tf.nn.sigmoid(shared_mlp(avg_pool) + shared_mlp(max_pool))
        return inputs * tf.reshape(attention, [-1, 1, 1, self.channels])

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels,
                       'reduction_ratio': self.reduction_ratio})
        return config


class SpatialAttention(layers.Layer):
    """Spatial attention module"""
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv_w = self.add_weight(
            'conv_w', (self.kernel_size, self.kernel_size, 2, 1),
            initializer='glorot_uniform')
        self.conv_b = self.add_weight('conv_b', (1,), initializer='zeros')
        self.built = True

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=-1)
        attention = tf.nn.conv2d(concat, self.conv_w, 
                                  strides=[1,1,1,1], padding='SAME')
        attention = tf.nn.bias_add(attention, self.conv_b)
        return inputs * tf.nn.sigmoid(attention)

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


class DiamondAttentionBlock(layers.Layer):
    """
    Fixed Diamond Attention Block:
    - Pre-activation design (BN → ReLU → Conv) for better gradient flow
    - Proper residual connections
    - Returns diversity loss for training
    """
    def __init__(self, filters, kernel_size=5, num_kernels=4, **kwargs):
        super(DiamondAttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

    def build(self, input_shape):
        input_channels = input_shape[-1]

        self.bn1 = layers.BatchNormalization()
        self.diamond_conv = DiamondConv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            num_kernels=self.num_kernels,
            use_attention=True
        )
        self.bn2 = layers.BatchNormalization()

        # Second conv for deeper block
        self.conv2 = layers.Conv2D(
            self.filters, 3, padding='same',
            kernel_initializer='he_normal', use_bias=False
        )
        self.bn3 = layers.BatchNormalization()

        self.channel_attention = ChannelAttention(self.filters)
        self.spatial_attention = SpatialAttention()

        # Residual projection
        self.use_projection = (input_channels != self.filters)
        if self.use_projection:
            self.proj_conv = layers.Conv2D(
                self.filters, 1, padding='same',
                kernel_initializer='he_normal', use_bias=False
            )
            self.bn_proj = layers.BatchNormalization()

        self.built = True

    def call(self, inputs, training=None):
        # Pre-activation
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.diamond_conv(x, training=training)

        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x, training=training)

        # Attention
        x = self.bn3(x, training=training)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        # Residual
        if self.use_projection:
            residual = self.proj_conv(inputs, training=training)
            residual = self.bn_proj(residual, training=training)
        else:
            residual = inputs

        return tf.nn.relu(x + residual)

    def get_diversity_loss(self):
        return self.diamond_conv.get_kernel_diversity_loss()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'num_kernels': self.num_kernels
        })
        return config


class DiamondNetwork(Model):
    """
    Complete Diamond Network as a proper Model subclass.
    Includes diversity regularization in the training step.
    """
    def __init__(self, input_shape_hw=(32, 32, 3), diamond_size=5,
                 num_kernels=4, num_diamond_blocks=3,
                 base_filters=64, num_classes=10,
                 diversity_weight=0.01, **kwargs):
        super(DiamondNetwork, self).__init__(**kwargs)

        self.diversity_weight = diversity_weight
        self._input_shape_hw = input_shape_hw

        # Stem
        self.stem_conv1 = layers.Conv2D(
            base_filters, 3, padding='same', use_bias=False,
            kernel_initializer='he_normal'
        )
        self.stem_bn1 = layers.BatchNormalization()
        self.stem_conv2 = layers.Conv2D(
            base_filters, 3, padding='same', use_bias=False,
            kernel_initializer='he_normal'
        )
        self.stem_bn2 = layers.BatchNormalization()

        # Diamond blocks
        self.diamond_blocks = []
        self.pool_layers = []
        filters_list = [base_filters * (2 ** i) 
                        for i in range(num_diamond_blocks)]

        for i, filters in enumerate(filters_list):
            self.diamond_blocks.append(
                DiamondAttentionBlock(
                    filters=filters,
                    kernel_size=diamond_size,
                    num_kernels=num_kernels,
                    name=f'diamond_block_{i}'
                )
            )
            if i < num_diamond_blocks - 1:
                self.pool_layers.append(
                    layers.MaxPooling2D(pool_size=(2, 2))
                )
            else:
                self.pool_layers.append(None)

        self.dropout_blocks = [
            layers.Dropout(0.1) for _ in range(num_diamond_blocks)
        ]

        # Head
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(256, kernel_initializer='he_normal')
        self.bn_fc1 = layers.BatchNormalization()
        self.drop_fc1 = layers.Dropout(0.4)
        self.fc2 = layers.Dense(128, kernel_initializer='he_normal')
        self.bn_fc2 = layers.BatchNormalization()
        self.drop_fc2 = layers.Dropout(0.4)
        self.classifier = layers.Dense(
            num_classes, activation='softmax',
            kernel_initializer='glorot_uniform'
        )

    def call(self, inputs, training=None):
        # Stem
        x = self.stem_conv1(inputs)
        x = self.stem_bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.stem_conv2(x)
        x = self.stem_bn2(x, training=training)
        x = tf.nn.relu(x)

        # Diamond blocks
        for block, pool, drop in zip(self.diamond_blocks,
                                      self.pool_layers,
                                      self.dropout_blocks):
            x = block(x, training=training)
            x = drop(x, training=training)
            if pool is not None:
                x = pool(x)

        # Head
        x = self.global_pool(x)
        x = self.fc1(x)
        x = self.bn_fc1(x, training=training)
        x = tf.nn.relu(x)
        x = self.drop_fc1(x, training=training)
        x = self.fc2(x)
        x = self.bn_fc2(x, training=training)
        x = tf.nn.relu(x)
        x = self.drop_fc2(x, training=training)
        return self.classifier(x)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Main classification loss
            loss = self.compiled_loss(y, y_pred)

            # Diversity regularization: penalize similar kernels
            diversity_loss = 0.0
            for block in self.diamond_blocks:
                diversity_loss += block.get_diversity_loss()
            
            total_loss = loss + self.diversity_weight * diversity_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        
        # Gradient clipping to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
        self.compiled_metrics.update_state(y, y_pred)

        results = {m.name: m.result() for m in self.metrics}
        results['diversity_loss'] = diversity_loss
        results['total_loss'] = total_loss
        return results


def create_diamond_attention_network(
    input_shape=(32, 32, 3),
    diamond_size=5,
    num_kernels=4,
    num_diamond_blocks=3,
    base_filters=64,
    num_classes=10,
    diversity_weight=0.01
):
    """Create the diamond network"""
    net = DiamondNetwork(
        input_shape_hw=input_shape,
        diamond_size=diamond_size,
        num_kernels=num_kernels,
        num_diamond_blocks=num_diamond_blocks,
        base_filters=base_filters,
        num_classes=num_classes,
        diversity_weight=diversity_weight,
        name='DiamondAttentionNetwork'
    )
    # Build the model by passing a dummy input
    dummy = tf.zeros((1,) + input_shape)
    _ = net(dummy)
    return net


def visualize_diamond_mask(kernel_size=5):
    """Visualize the diamond-shaped mask"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mask = create_diamond_mask(kernel_size)

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='Blues', interpolation='nearest')
    plt.title(f'Diamond Mask ({kernel_size}x{kernel_size})',
              fontsize=14, fontweight='bold')

    for i in range(kernel_size + 1):
        plt.axhline(i - 0.5, color='black', linewidth=0.5)
        plt.axvline(i - 0.5, color='black', linewidth=0.5)

    for i in range(kernel_size):
        for j in range(kernel_size):
            color = 'white' if mask[i, j] > 0.5 else 'black'
            plt.text(j, i, f'{mask[i, j]:.0f}', ha='center',
                     va='center', fontsize=12, color=color)

    plt.xticks(range(kernel_size))
    plt.yticks(range(kernel_size))
    plt.savefig('diamond_mask.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Diamond mask saved as 'diamond_mask.png'")


def visualize_learned_kernels(model):
    """Visualize learned diamond kernels"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    diamond_blocks = [l for l in model.diamond_blocks
                      if isinstance(l, DiamondAttentionBlock)]

    if not diamond_blocks:
        print("No diamond layers found")
        return

    for layer_idx, block in enumerate(diamond_blocks):
        dc = block.diamond_conv
        num_k = len(dc.kernels)

        fig, axes = plt.subplots(2, num_k, figsize=(num_k * 3, 6))
        if num_k == 1:
            axes = axes.reshape(2, 1)

        mask = dc.diamond_mask
        for ki, kernel in enumerate(dc.kernels):
            kw = kernel.numpy()
            kavg = np.mean(kw, axis=(2, 3))
            vmax = max(abs(kavg.min()), abs(kavg.max()), 1e-8)

            ax = axes[0, ki]
            im = ax.imshow(kavg, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'Kernel {ki + 1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

            masked = kavg * mask
            ax = axes[1, ki]
            im = ax.imshow(masked, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'Masked {ki + 1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle(f'Diamond Block {layer_idx + 1} Kernels',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'diamond_kernels_block_{layer_idx + 1}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Kernels for block {layer_idx + 1} saved")


def plot_training_history(history):
    """Plot training history"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[1].set_title('Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Training history saved as 'training_history.png'")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Diamond Neural Network - Fixed Version")
    print("=" * 60)

    np.random.seed(42)
    tf.random.set_seed(42)

    config = {
        'input_shape': (32, 32, 3),
        'diamond_size': 5,       # 5x5 diamond has 13 active pixels (good coverage)
        'num_kernels': 4,
        'num_diamond_blocks': 3,
        'base_filters': 64,      # More capacity
        'num_classes': 10,
        'batch_size': 128,       # Proper batch size
        'epochs': 2,
        'learning_rate': 1e-3,   # Standard Adam LR
        'diversity_weight': 0.01,
    }

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Visualize
    print("\n" + "-" * 40)
    visualize_diamond_mask(config['diamond_size'])

    # Load CIFAR-10
    print("\n" + "-" * 40)
    print("Loading CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Use full dataset for proper training
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Per-channel normalization (important for convergence)
    mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
    std = np.std(x_train, axis=(0, 1, 2), keepdims=True)
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.1, 0.1),
    ], name='augmentation')

    # Build augmented dataset pipeline
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(50000).batch(config['batch_size'])
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = val_ds.batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)

    # Create model
    print("\n" + "-" * 40)
    print("Creating model...")

    model = create_diamond_attention_network(
        input_shape=config['input_shape'],
        diamond_size=config['diamond_size'],
        num_kernels=config['num_kernels'],
        num_diamond_blocks=config['num_diamond_blocks'],
        base_filters=config['base_filters'],
        num_classes=config['num_classes'],
        diversity_weight=config['diversity_weight'],
    )

    model.summary()

    # Compile
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['learning_rate'],
    )
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        ),
    ]

    # Train
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    history = model.fit(
        train_ds,
        epochs=config['epochs'],
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(val_ds, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Visualizations
    plot_training_history(history)
    visualize_learned_kernels(model)

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Val Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("=" * 60)