import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming these feature extractors are correctly defined elsewhere
# If not using these, the relevant section can be simplified/removed.
from builder.models.feature_extractor.psd_feature import *
from builder.models.feature_extractor.spectrogram_feature_binary import *
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE

class SIMPLECNN1D(nn.Module):
    """
    A simple 1D CNN model inspired by the structure of CNN1D_BLSTM,
    but without recurrent layers (LSTMs). Designed for basic feature
    extraction and classification.
    """
    def __init__(self, args, device):
        super(SIMPLECNN1D, self).__init__()
        self.args = args
        self.num_data_channel = args.num_channel
        self.output_dim = args.output_dim
        self.dropout_rate = args.dropout # Get dropout rate from args
        self.feature_extractor_name = args.enc_model

        # --- Feature Extractor Handling (Replicating logic from CNN1D_BLSTM) ---
        self.feat_model = None
        input_cnn_channels = 0 # Channels going into the first Conv1d

        if self.feature_extractor_name == "raw":
            # For Conv1d, input channels = number of data channels
            input_cnn_channels = self.num_data_channel
        else:
            # Initialize feature extractor models if needed
            self.feat_models = nn.ModuleDict([
                ['psd1', PSD_FEATURE1()],
                ['psd2', PSD_FEATURE2()],
                ['stft1', SPECTROGRAM_FEATURE_BINARY1()],
                ['stft2', SPECTROGRAM_FEATURE_BINARY2()],
                ['sincnet', SINCNET_FEATURE(args=args,
                                            num_eeg_channel=self.num_data_channel)]
            ])
            if self.feature_extractor_name in self.feat_models:
                self.feat_model = self.feat_models[self.feature_extractor_name]

                # Determine the number of features per channel from the extractor
                # Replicating the logic from CNN1D_BLSTM to calculate combined channel dimension
                feature_num = 0
                if args.enc_model == "psd1" or args.enc_model == "psd2":
                    feature_num = 7
                elif args.enc_model == "sincnet":
                    # This requires knowing SincNet's output feature dimension per channel.
                    # Let's make an assumption or require a new arg. Assume 8 features per channel for now.
                    # Ideally, SincNet should define its output feature size.
                    sincnet_out_features_per_channel = getattr(args, 'sincnet_out_features', 8) # Default 8 if not set
                    feature_num = sincnet_out_features_per_channel
                    print(f"[SimpleCNN1D INFO] Assuming SincNet output features per channel: {feature_num}")
                elif args.enc_model == "stft1":
                    feature_num = 50 # Number of frequency bins likely
                elif args.enc_model == "stft2":
                    feature_num = 100 # Number of frequency bins likely
                else:
                    raise ValueError(f"Unhandled feature extractor '{self.feature_extractor_name}' in channel calculation.")

                # Combine features and channels as input to Conv1d, like CNN1D_BLSTM
                input_cnn_channels = feature_num * self.num_data_channel
                print(f"[SimpleCNN1D INFO] Feature extractor '{self.feature_extractor_name}': Features={feature_num}, Channels={self.num_data_channel} => Input CNN Channels={input_cnn_channels}")

            else:
                 raise ValueError(f"Feature extractor '{self.feature_extractor_name}' not found in predefined modules.")


        # --- Simple 1D CNN Blocks ---
        # Using ReLU activation like the working example
        activation = 'relu'
        self.activations = nn.ModuleDict([
                ['relu', nn.ReLU(inplace=True)],
                # Add others if needed, but keep ReLU for similarity
        ])
        self.activation_fn = self.activations[activation]

        # Define CNN blocks directly (simpler than helper function)
        # Block 1
        channels_1 = 64 # Number of filters
        kernel_1 = 15   # Kernel size (adjust as needed)
        self.conv1 = nn.Conv1d(input_cnn_channels, channels_1, kernel_size=kernel_1, padding=kernel_1//2, bias=False) # Often False before BN
        self.bn1 = nn.BatchNorm1d(channels_1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4) # Pool + Stride
        self.dropout1 = nn.Dropout(self.dropout_rate)

        # Block 2
        channels_2 = 128
        kernel_2 = 7
        self.conv2 = nn.Conv1d(channels_1, channels_2, kernel_size=kernel_2, padding=kernel_2//2, bias=False)
        self.bn2 = nn.BatchNorm1d(channels_2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        # Store final channel count for classifier input calculation
        self.final_cnn_channels = channels_2

        # --- Pooling before Classifier ---
        # Use Adaptive Pooling to handle variable length after CNNs and get fixed size
        self.global_pool = nn.AdaptiveAvgPool1d(1) # Output shape: (Batch, final_cnn_channels, 1)

        # --- Classifier (Mimicking structure of CNN1D_BLSTM) ---
        classifier_hidden_dim = 64 # Intermediate hidden dimension
        self.classifier = nn.Sequential(
            # Input size is self.final_cnn_channels after pooling and flattening
            nn.Linear(self.final_cnn_channels, classifier_hidden_dim, bias=True),
            nn.BatchNorm1d(classifier_hidden_dim), # BatchNorm stabilization
            self.activation_fn, # ReLU activation
            nn.Dropout(self.dropout_rate), # Optional Dropout
            nn.Linear(classifier_hidden_dim, self.output_dim, bias=True) # Final classification layer
        )

    def forward(self, x):
        """
        Forward pass for SimpleCNN1D.
        Input x expected shape: (batch_size, sequence_length/samples, num_channels)
        """
        # Permute to (batch_size, num_channels, sequence_length/samples) for Conv1d
        x = x.permute(0, 2, 1)

        # Apply feature extractor if specified and exists
        if self.feature_extractor_name != "raw" and self.feat_model is not None:
            x = self.feat_model(x)
            # --- Reshape features to match expected CNN input ---
            # Based on CNN1D_BLSTM, expects (B, C*F, T')
            # This assumes feat_model outputs something like (B, C, F, T') or (B, C, Features)
            # This part is sensitive to the actual output shape of the feature extractors.
            if x.ndim == 4: # e.g., (B, C, F, T) from STFT
                # Reshape: Combine C and F dimensions
                target_channels = self.conv1.in_channels # Get expected channels
                x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
                if x.shape[1] != target_channels:
                     # This check is important!
                     raise ValueError(f"Feature shape mismatch after reshape: Expected {target_channels} channels, got {x.shape[1]}")
            elif x.ndim == 3: # e.g., (B, Features, T) from SincNet? -> Needs C dim combined
                 # Check if the dimension matches the expected combined C*F dimension
                 target_channels = self.conv1.in_channels
                 if x.shape[1] != target_channels:
                     # Attempt to reshape if C was dropped? Highly speculative.
                     # Example: If num_channels was folded into batch: (B*C, F, T) -> (B, C*F, T) ??
                     # This needs knowledge of the specific extractor. Let's raise an error for now.
                     raise ValueError(f"Unexpected 3D feature shape {x.shape}, expected {target_channels} channels after C*F combine.")
                 # If shape[1] *already* matches the combined C*F, proceed.
            else:
                 raise ValueError(f"Unsupported feature shape {x.shape} from extractor {self.feature_extractor_name}")

        # --- Pass through CNN blocks ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_fn(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation_fn(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # --- Global Pooling ---
        # Input shape: (Batch, final_cnn_channels, Reduced_Time)
        x = self.global_pool(x) # Output shape: (Batch, final_cnn_channels, 1)

        # --- Flatten ---
        # Remove the last dimension (size 1) and keep Batch, Features
        x = torch.flatten(x, 1) # Output shape: (Batch, final_cnn_channels)

        # --- Classifier ---
        output = self.classifier(x)

        # Return output and a placeholder (0) for hidden state for consistency
        return output, 0

    def init_state(self, device):
        """
        Initializes state if the model were recurrent.
        SimpleCNN1D is not recurrent, so return placeholder.
        """
        # No hidden state needed
        return 0