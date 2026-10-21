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
    """
    def __init__(self, filters, kernel_size=5, num_kernels=4, use_attention=True, **kwargs):
        super(DiamondConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.use_attention = use_attention
        
        # Create diamond mask
        self.diamond_mask = create_diamond_mask(kernel_size)
        self.diamond_mask_tensor = None
        
    def build(self, input_shape):
        # Get input channels - handle None case
        input_channels = input_shape[-1]
        if input_channels is None:
            input_channels = 32  # Default fallback
        
        self.input_channels = input_channels
        
        # Create diamond mask tensor
        self.diamond_mask_tensor = tf.constant(
            self.diamond_mask.reshape(self.kernel_size, self.kernel_size, 1, 1),
            dtype=tf.float32
        )
        
        # Multiple diamond kernels
        self.kernels = []
        self.biases = []
        
        for i in range(self.num_kernels):
            kernel = self.add_weight(
                name=f'diamond_kernel_{i}',
                shape=(self.kernel_size, self.kernel_size, input_channels, self.filters),
                initializer='glorot_uniform',
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
        
        # Attention weights for kernel combination
        if self.use_attention:
            self.attention_dense1 = self.add_weight(
                name='attention_dense1',
                shape=(input_channels, self.filters),
                initializer='glorot_uniform',
                trainable=True
            )
            self.attention_dense2 = self.add_weight(
                name='attention_dense2',
                shape=(self.filters, self.num_kernels),
                initializer='glorot_uniform',
                trainable=True
            )
        
        self.built = True
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # Apply each diamond kernel
        conv_outputs = []
        for kernel, bias in zip(self.kernels, self.biases):
            # Apply diamond mask to kernel
            masked_kernel = kernel * self.diamond_mask_tensor
            
            # Standard convolution with diamond-masked kernel
            output = tf.nn.conv2d(
                inputs,
                masked_kernel,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            output = tf.nn.bias_add(output, bias)
            conv_outputs.append(output)
        
        # Stack outputs: [batch, H, W, filters, num_kernels]
        stacked_outputs = tf.stack(conv_outputs, axis=-1)
        
        if self.use_attention:
            # Compute attention weights based on input features
            # Global average pooling
            pooled_input = tf.reduce_mean(inputs, axis=[1, 2])  # [batch, channels]
            
            # Attention scores
            attention = tf.matmul(pooled_input, self.attention_dense1)  # [batch, filters]
            attention = tf.nn.relu(attention)
            attention = tf.matmul(attention, self.attention_dense2)  # [batch, num_kernels]
            attention_weights = tf.nn.softmax(attention, axis=-1)  # [batch, num_kernels]
            
            # Reshape for broadcasting: [batch, 1, 1, 1, num_kernels]
            attention_weights = tf.reshape(attention_weights, [batch_size, 1, 1, 1, self.num_kernels])
            
            # Weighted combination
            output = tf.reduce_sum(stacked_outputs * attention_weights, axis=-1)
        else:
            # Simple average
            output = tf.reduce_mean(stacked_outputs, axis=-1)
        
        return tf.nn.relu(output)
    
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
    """Channel attention module"""
    def __init__(self, channels, reduction_ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        reduced_channels = max(self.channels // self.reduction_ratio, 4)
        
        self.fc1_weights = self.add_weight(
            name='fc1_weights',
            shape=(self.channels, reduced_channels),
            initializer='glorot_uniform',
            trainable=True
        )
        self.fc1_bias = self.add_weight(
            name='fc1_bias',
            shape=(reduced_channels,),
            initializer='zeros',
            trainable=True
        )
        self.fc2_weights = self.add_weight(
            name='fc2_weights',
            shape=(reduced_channels, self.channels),
            initializer='glorot_uniform',
            trainable=True
        )
        self.fc2_bias = self.add_weight(
            name='fc2_bias',
            shape=(self.channels,),
            initializer='zeros',
            trainable=True
        )
        
        self.built = True
        
    def call(self, inputs):
        # Global average pooling
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2])  # [batch, channels]
        
        # Global max pooling
        max_pool = tf.reduce_max(inputs, axis=[1, 2])  # [batch, channels]
        
        # Shared MLP
        avg_out = tf.matmul(avg_pool, self.fc1_weights) + self.fc1_bias
        avg_out = tf.nn.relu(avg_out)
        avg_out = tf.matmul(avg_out, self.fc2_weights) + self.fc2_bias
        
        max_out = tf.matmul(max_pool, self.fc1_weights) + self.fc1_bias
        max_out = tf.nn.relu(max_out)
        max_out = tf.matmul(max_out, self.fc2_weights) + self.fc2_bias
        
        attention = tf.nn.sigmoid(avg_out + max_out)
        attention = tf.reshape(attention, [-1, 1, 1, self.channels])
        
        return inputs * attention
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'reduction_ratio': self.reduction_ratio
        })
        return config


class SpatialAttention(layers.Layer):
    """Spatial attention module"""
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv_weights = self.add_weight(
            name='conv_weights',
            shape=(self.kernel_size, self.kernel_size, 2, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.conv_bias = self.add_weight(
            name='conv_bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        
        self.built = True
        
    def call(self, inputs):
        # Channel-wise statistics
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)  # [batch, H, W, 1]
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)  # [batch, H, W, 1]
        concat = tf.concat([avg_out, max_out], axis=-1)  # [batch, H, W, 2]
        
        # Convolution
        attention = tf.nn.conv2d(
            concat,
            self.conv_weights,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        attention = tf.nn.bias_add(attention, self.conv_bias)
        attention = tf.nn.sigmoid(attention)
        
        return inputs * attention
    
    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


class DiamondAttentionBlock(layers.Layer):
    """
    Complete Diamond Attention Block
    """
    def __init__(self, filters, kernel_size=5, num_kernels=4, **kwargs):
        super(DiamondAttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        
    def build(self, input_shape):
        input_channels = input_shape[-1]
        if input_channels is None:
            input_channels = self.filters
        
        # Diamond convolution
        self.diamond_conv = DiamondConv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            num_kernels=self.num_kernels,
            use_attention=True
        )
        
        # Attention modules
        self.channel_attention = ChannelAttention(channels=self.filters)
        self.spatial_attention = SpatialAttention()
        
        # Batch normalization
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        
        # Residual projection
        self.use_projection = (input_channels != self.filters)
        if self.use_projection:
            self.projection_weights = self.add_weight(
                name='projection_weights',
                shape=(1, 1, input_channels, self.filters),
                initializer='glorot_uniform',
                trainable=True
            )
            self.projection_bias = self.add_weight(
                name='projection_bias',
                shape=(self.filters,),
                initializer='zeros',
                trainable=True
            )
            self.bn_proj = layers.BatchNormalization()
        
        self.built = True
        
    def call(self, inputs, training=None):
        # Diamond convolution
        x = self.diamond_conv(inputs)
        x = self.bn1(x, training=training)
        
        # Channel and spatial attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.bn2(x, training=training)
        
        # Residual connection
        if self.use_projection:
            residual = tf.nn.conv2d(
                inputs,
                self.projection_weights,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            residual = tf.nn.bias_add(residual, self.projection_bias)
            residual = self.bn_proj(residual, training=training)
        else:
            residual = inputs
        
        output = x + residual
        output = tf.nn.relu(output)
        
        return output
    
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


class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention for feature refinement"""
    def __init__(self, d_model, num_heads=4, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]
        
        # Query, Key, Value projections
        self.wq = self.add_weight(
            name='wq',
            shape=(self.d_model, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        self.wk = self.add_weight(
            name='wk',
            shape=(self.d_model, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        self.wv = self.add_weight(
            name='wv',
            shape=(self.d_model, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        self.wo = self.add_weight(
            name='wo',
            shape=(self.d_model, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Layer normalization parameters
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.d_model,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(self.d_model,),
            initializer='zeros',
            trainable=True
        )
        
        self.built = True
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        h, w = self.height, self.width
        
        # Reshape to sequence: [batch, H*W, channels]
        x = tf.reshape(inputs, [batch_size, h * w, self.d_model])
        
        # Compute Q, K, V
        q = tf.matmul(x, self.wq)
        k = tf.matmul(x, self.wk)
        v = tf.matmul(x, self.wv)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, h * w, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch_size, h * w, self.num_heads, self.head_dim])
        v = tf.reshape(v, [batch_size, h * w, self.num_heads, self.head_dim])
        
        # Transpose: [batch, num_heads, seq_len, head_dim]
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Attention scores
        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        scores = tf.matmul(q, k, transpose_b=True) / scale
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        attention_output = tf.matmul(attention_weights, v)
        
        # Reshape back
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, [batch_size, h * w, self.d_model])
        
        # Output projection
        output = tf.matmul(attention_output, self.wo)
        
        # Residual connection and layer norm
        output = output + x
        mean = tf.reduce_mean(output, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(output - mean), axis=-1, keepdims=True)
        output = (output - mean) / tf.sqrt(variance + 1e-6)
        output = output * self.gamma + self.beta
        
        # Reshape back to spatial: [batch, H, W, channels]
        output = tf.reshape(output, [batch_size, h, w, self.d_model])
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config


def create_diamond_attention_network(
    input_shape=(32, 32, 3),
    diamond_size=5,
    num_kernels=4,
    num_diamond_blocks=3,
    base_filters=32,
    num_classes=10
):
    """
    Create Diamond Neural Network with Attention-based Weight Sharing
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial stem with explicit channel dimension
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='stem_conv1')(inputs)
    x = layers.BatchNormalization(name='stem_bn1')(x)
    x = layers.Conv2D(base_filters, 3, padding='same', activation='relu', name='stem_conv2')(x)
    x = layers.BatchNormalization(name='stem_bn2')(x)
    
    # Current spatial dimensions
    current_h, current_w = input_shape[0], input_shape[1]
    
    # Diamond attention blocks with increasing filters
    filters_list = [base_filters * (2 ** i) for i in range(num_diamond_blocks)]
    
    for i, filters in enumerate(filters_list):
        # Diamond attention block
        x = DiamondAttentionBlock(
            filters=filters,
            kernel_size=diamond_size,
            num_kernels=num_kernels,
            name=f'diamond_block_{i}'
        )(x)
        
        # Multi-head self-attention (only if spatial dims are reasonable)
        if current_h >= 4 and current_w >= 4:
            x = MultiHeadSelfAttention(
                d_model=filters,
                num_heads=4,
                name=f'mhsa_{i}'
            )(x)
        
        # Dropout for regularization
        x = layers.Dropout(0.2, name=f'dropout_{i}')(x)
        
        # Downsample (except for the last block)
        if i < num_diamond_blocks - 1:
            x = layers.MaxPooling2D(pool_size=(2, 2), name=f'pool_{i}')(x)
            current_h = current_h // 2
            current_w = current_w // 2
    
    # Global feature aggregation
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Classification head
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization(name='fc_bn1')(x)
    x = layers.Dropout(0.5, name='fc_dropout1')(x)
    
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization(name='fc_bn2')(x)
    x = layers.Dropout(0.5, name='fc_dropout2')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs, name='DiamondAttentionNetwork')
    return model


def visualize_diamond_mask(kernel_size=5):
    """Visualize the diamond-shaped mask"""
    import matplotlib.pyplot as plt
    
    mask = create_diamond_mask(kernel_size)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='Blues', interpolation='nearest')
    plt.title(f'Diamond Mask ({kernel_size}x{kernel_size})', fontsize=14, fontweight='bold')
    
    # Add grid
    for i in range(kernel_size + 1):
        plt.axhline(i - 0.5, color='black', linewidth=0.5)
        plt.axvline(i - 0.5, color='black', linewidth=0.5)
    
    # Add values
    for i in range(kernel_size):
        for j in range(kernel_size):
            plt.text(j, i, f'{mask[i, j]:.0f}', ha='center', va='center', 
                    fontsize=12, color='white' if mask[i, j] > 0.5 else 'black')
    
    plt.xticks(range(kernel_size))
    plt.yticks(range(kernel_size))
    plt.savefig('diamond_mask.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Diamond mask saved as 'diamond_mask.png'")


def visualize_learned_kernels(model):
    """Visualize learned diamond kernels from the model"""
    import matplotlib.pyplot as plt
    
    diamond_layers = []
    for layer in model.layers:
        if isinstance(layer, DiamondAttentionBlock):
            diamond_layers.append(layer)
    
    if not diamond_layers:
        print("No diamond layers found")
        return
    
    for layer_idx, block in enumerate(diamond_layers):
        diamond_conv = block.diamond_conv
        num_kernels = len(diamond_conv.kernels)
        
        fig, axes = plt.subplots(2, num_kernels, figsize=(num_kernels * 3, 6))
        if num_kernels == 1:
            axes = axes.reshape(2, 1)
        
        mask = diamond_conv.diamond_mask
        
        for kernel_idx, kernel in enumerate(diamond_conv.kernels):
            kernel_weights = kernel.numpy()
            kernel_avg = np.mean(kernel_weights, axis=(2, 3))
            
            # Original kernel
            ax = axes[0, kernel_idx]
            vmax = max(abs(kernel_avg.min()), abs(kernel_avg.max()))
            im = ax.imshow(kernel_avg, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'Kernel {kernel_idx + 1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Diamond-masked kernel
            masked_kernel = kernel_avg * mask
            ax = axes[1, kernel_idx]
            im = ax.imshow(masked_kernel, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'Masked {kernel_idx + 1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.suptitle(f'Diamond Block {layer_idx + 1} Kernels', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'diamond_kernels_block_{layer_idx + 1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Kernels for block {layer_idx + 1} saved")


def plot_training_history(history):
    """Plot training history"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training history saved as 'training_history.png'")


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Diamond Neural Network with Attention-Based Weight Sharing")
    print("=" * 60)
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Configuration
    config = {
        'input_shape': (32, 32, 3),
        'diamond_size': 3,
        'num_kernels': 3,
        'num_diamond_blocks': 1,
        'base_filters': 12,
        'num_classes': 10,
        'batch_size': 5,
        'epochs': 10,
        'learning_rate': 0.01
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Visualize diamond mask
    print("\n" + "-" * 40)
    print("Visualizing diamond mask...")
    visualize_diamond_mask(config['diamond_size'])
    
    # Create model
    print("\n" + "-" * 40)
    print("Creating model...")
    
    model = create_diamond_attention_network(
        input_shape=config['input_shape'],
        diamond_size=config['diamond_size'],
        num_kernels=config['num_kernels'],
        num_diamond_blocks=config['num_diamond_blocks'],
        base_filters=config['base_filters'],
        num_classes=config['num_classes']
    )
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Load dataset
    print("\n" + "-" * 40)
    print("Loading CIFAR-10 dataset...")
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize
    x_train = x_train[0:5000].astype('float32') / 255.0
    x_test = x_test[5000:6000].astype('float32') / 255.0
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_diamond_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n" + "=" * 60)
    print("Training Started")
    print("=" * 60)
    
    history = model.fit(
        x_train, y_train[0:5000],
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=(x_test, y_test[5000:6000]),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Visualizations
    print("\n" + "-" * 40)
    print("Generating visualizations...")
    plot_training_history(history)
    visualize_learned_kernels(model)
    
    # Save model
    model.save('diamond_attention_network_final.h5')
    print("\nModel saved as 'diamond_attention_network_final.h5'")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print("=" * 60)