import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable # Note: Variable is deprecated, direct tensor usage is standard now. Retained for consistency if needed by framework.

# Assuming these feature extractors are correctly defined elsewhere
from builder.models.feature_extractor.psd_feature import *
from builder.models.feature_extractor.spectrogram_feature_binary import *
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE

class EEGNET(nn.Module):
    """
    EEGNet model implementation based on the paper:
    Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018).
    EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.
    Journal of neural engineering, 15(5), 056013.

    Adapted to fit the structure of the EEG_real_time_seizure_detection framework.
    NOTE: EEGNet is primarily designed for raw time-domain EEG. Using it with
          other feature extractors (STFT, PSD) might require modifications or
          may not leverage the architecture's strengths effectively.

    Modification: Increased BatchNorm epsilon for potentially improved numerical stability.
    """
    def __init__(self, args, device):
        super(EEGNET, self).__init__()
        self.args = args
        self.num_data_channel = args.num_channel
        self.output_dim = args.output_dim
        self.dropout = args.dropout
        self.feature_extractor_name = args.enc_model

        # --- Feature Extractor Handling (similar to example) ---
        self.feat_model = None
        if self.feature_extractor_name != "raw":
            print(f"[EEGNet INFO] Initializing feature extractor: {self.feature_extractor_name}")
            # Check if feature extractor modules are needed
            feat_models = nn.ModuleDict([
                ['psd1', PSD_FEATURE1()],
                ['psd2', PSD_FEATURE2()],
                ['stft1', SPECTROGRAM_FEATURE_BINARY1()],
                ['stft2', SPECTROGRAM_FEATURE_BINARY2()],
                ['sincnet', SINCNET_FEATURE(args=args,
                                            num_eeg_channel=self.num_data_channel)]
            ])
            if self.feature_extractor_name in feat_models:
                self.feat_model = feat_models[self.feature_extractor_name]
            else:
                 print(f"[EEGNet WARNING] Feature extractor '{self.feature_extractor_name}' not found in predefined modules.")
                 # Decide how to handle unknown extractors if necessary
                 # For now, self.feat_model remains None

        # --- EEGNet Core Parameters ---
        # These might need tuning based on data/sampling rate
        # F1: Number of temporal filters
        # D: Depth multiplier for spatial filters
        # F2: Number of pointwise filters
        # kernLength: Length of temporal convolution kernel (~0.5 sec)
        # poolKern: Pooling kernel size
        # norm_rate: Max norm constraint (not typically implemented directly in PyTorch layers)
        self.F1 = args.eegnet_F1 if hasattr(args, 'eegnet_F1') else 8
        self.D = args.eegnet_D if hasattr(args, 'eegnet_D') else 2
        self.F2 = self.F1 * self.D # Often F2 = F1 * D
        # Get sample rate from args calculated in get_data_preprocessed if possible
        self.sampling_rate = args.sample_rate if hasattr(args, 'sample_rate') else 200
        self.kernLength = args.eegnet_kernLength if hasattr(args, 'eegnet_kernLength') else self.sampling_rate // 2 # ~0.5 second kernel
        self.poolKern1 = args.eegnet_poolKern1 if hasattr(args, 'eegnet_poolKern1') else 4
        self.poolKern2 = args.eegnet_poolKern2 if hasattr(args, 'eegnet_poolKern2') else 8
        self.separableKernLength = args.eegnet_separableKernLength if hasattr(args, 'eegnet_separableKernLength') else 16

        # Activation (using dict like example)
        activation = 'elu' # ELU is common in EEGNet
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()],
                ['relu', nn.ReLU(inplace=True)],
                ['tanh', nn.Tanh()],
                ['sigmoid', nn.Sigmoid()],
                ['leaky_relu', nn.LeakyReLU(0.2)],
                ['elu', nn.ELU()]
        ])
        self.activation = self.activations[activation]

        # --- BatchNorm Epsilon Value ---
        # Increase slightly from default 1e-5 for potentially better stability
        self.batch_norm_eps = 1e-4 # Changed from 1e-5

        # --- EEGNet Architecture Blocks ---
        # Block 1: Temporal Convolution + Depthwise Spatial Convolution
        # Input shape for Conv2d: (Batch, 1, Channels, Samples)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernLength), padding=(0, self.kernLength // 2), bias=False),
            nn.BatchNorm2d(self.F1, eps=self.batch_norm_eps), # Added eps
            # Depthwise Convolution
            nn.Conv2d(self.F1, self.F1 * self.D, (self.num_data_channel, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D, eps=self.batch_norm_eps), # Added eps
            self.activation,
            nn.AvgPool2d((1, self.poolKern1)),
            nn.Dropout(self.dropout)
        )

        # Block 2: Separable Convolution (Depthwise temporal + Pointwise)
        # Input shape: (Batch, F1*D, 1, Samples_after_Pool1)
        self.block2 = nn.Sequential(
            # Separable Conv: Depthwise Temporal part
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.separableKernLength),
                      padding=(0, self.separableKernLength // 2), groups=self.F1 * self.D, bias=False),
            # Separable Conv: Pointwise part
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2, eps=self.batch_norm_eps), # Added eps
            self.activation,
            nn.AvgPool2d((1, self.poolKern2)),
            nn.Dropout(self.dropout)
        )

        # Calculate classifier input size dynamically
        self.classifier_input_dim = self._get_classifier_input_size(args)
        print(f"[EEGNet INFO] Calculated classifier input dimension: {self.classifier_input_dim}")

        # Classifier Block
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, self.output_dim)
            # Can add BatchNorm or other layers here if needed
        )


    def _get_classifier_input_size(self, args):
        """
        Calculates the input dimension for the classifier by passing a dummy tensor
        through the convolutional blocks. Runs on CPU.
        MUST use args.window_size_sig calculated in get_data_preprocessed.
        """
        bs = 1
        n_channels = self.num_data_channel

        # --- Use the calculated signal window size in samples ---
        if not hasattr(args, 'window_size_sig') or args.window_size_sig <= 0:
             raise AttributeError("args.window_size_sig not found or invalid...")
        else:
             n_samples = args.window_size_sig

        print(f"DEBUG [get_classifier_size]: Using n_samples = {n_samples} (from args.window_size_sig)")

        if n_samples <= 0: raise ValueError(...)
        if n_channels <= 0: raise ValueError(...)

        dummy_input_cpu = torch.randn(bs, n_channels, int(n_samples))
        dummy_input_cpu = torch.unsqueeze(dummy_input_cpu, dim=1)

        # --- FIX: Get the eps value locally ---
        local_batch_norm_eps = self.batch_norm_eps
        # -------------------------------------

        self.eval()
        with torch.no_grad():
            # --- Ensure temp blocks use the LOCAL eps value ---
            temp_block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.kernLength), padding=(0, self.kernLength // 2), bias=False),
                nn.BatchNorm2d(self.F1, eps=local_batch_norm_eps), # Use local variable
                nn.Conv2d(self.F1, self.F1 * self.D, (self.num_data_channel, 1), groups=self.F1, bias=False),
                nn.BatchNorm2d(self.F1 * self.D, eps=local_batch_norm_eps), # Use local variable
                self.activation,
                nn.AvgPool2d((1, self.poolKern1)),
                nn.Dropout(self.dropout)
            ).cpu()

            temp_block2 = nn.Sequential(
                nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.separableKernLength),
                          padding=(0, self.separableKernLength // 2), groups=self.F1 * self.D, bias=False),
                nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False),
                nn.BatchNorm2d(self.F2, eps=local_batch_norm_eps), # Use local variable
                self.activation,
                nn.AvgPool2d((1, self.poolKern2)),
                nn.Dropout(self.dropout)
            ).cpu()
            # --------------------------------------------

            output_block1 = temp_block1(dummy_input_cpu)
            output = temp_block2(output_block1)
        self.train()

        output = output.view(bs, -1)
        return output.size(1)


    def forward(self, x):
        """
        Forward pass for EEGNet.
        Input x expected shape: (batch_size, sequence_length/samples, num_channels)
        """
        # Permute to (batch_size, num_channels, sequence_length/samples)
        x = x.permute(0, 2, 1)

        # Apply feature extractor if specified and exists
        if self.feature_extractor_name != "raw" and self.feat_model is not None:
            x = self.feat_model(x)
            # --- Handling extracted features ---
            # EEGNet is designed for (N, C, T). Extracted features might be (N, C, F, T) or (N, C, Feat) etc.
            # Option 1: Try reshaping (might work for STFT-like features if F treated as C)
            # Example for (N, C, F, T) -> (N, C*F, T)
            if x.ndim == 4:
                print(f"[EEGNet WARNING] Reshaping 4D feature input {x.shape} for EEGNet.")
                x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3]) # (N, C*F, T)
                # Update effective number of channels for block1 potentially? Complicated.
                # Simpler: Flatten before classifier if not raw.
                print("[EEGNet WARNING] Non-raw features detected. Flattening after feature extraction and skipping EEGNet blocks.")
                x = x.view(x.size(0), -1) # Flatten features
                output = self.classifier(x)
                return output, 0 # Return 0 for hidden state placeholder
            elif x.ndim == 3:
                 # Assume (N, C, Features) - already suitable shape? Or (N, Features, C)? Needs check.
                 # Let's assume it's (N, C, Features) and proceed, but maybe flatten is safer.
                 print(f"[EEGNet WARNING] Non-raw 3D features detected {x.shape}. Flattening and skipping EEGNet blocks.")
                 x = x.view(x.size(0), -1) # Flatten features
                 output = self.classifier(x)
                 return output, 0 # Return 0 for hidden state placeholder
            else: # Unexpected shape
                 raise ValueError(f"Unsupported feature shape {x.shape} from extractor {self.feature_extractor_name}")

        # --- EEGNet Core Processing (only for RAW or appropriately shaped features) ---
        # Reshape for Conv2d: (Batch, 1, Channels, Samples)
        x = torch.unsqueeze(x, dim=1)

        # Pass through EEGNet blocks
        x = self.block1(x)
        x = self.block2(x)

        # Flatten for classifier
        x = x.view(x.size(0), -1) # Or x = torch.flatten(x, 1)

        # Classify
        output = self.classifier(x)

        # Return output and a placeholder for hidden state (consistent with framework)
        # EEGNet itself is stateless, so 0 is appropriate.
        return output, 0

    def init_state(self, device):
        """
        Initializes state if the model were recurrent (e.g., LSTM).
        EEGNet is not recurrent, so we return a placeholder.
        """
        # No hidden state needed for pure EEGNet
        return 0 # Or return None if the framework handles it