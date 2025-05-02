import torch
import torch.nn as nn

from builder.models.feature_extractor.psd_feature import *
from builder.models.feature_extractor.spectrogram_feature_binary import *
from builder.models.feature_extractor.sincnet_feature import SINCNET_FEATURE

class ALEXNET(nn.Module):
    def __init__(self, args, device):
        super(ALEXNET, self).__init__()

        self.args = args
        self.num_classes = self.args.output_dim
        self.in_channels = self.args.num_channel
        self.enc_model = self.args.enc_model
        self.features = True # Seems like this is always True unless raw? Might simplify logic.
        self.num_data_channel = self.args.num_channel

        self.feature_extractor = nn.ModuleDict([
                                ['psd1', PSD_FEATURE1()],
                                ['psd2', PSD_FEATURE2()],
                                ['stft1', SPECTROGRAM_FEATURE_BINARY1()],
                                ['stft2', SPECTROGRAM_FEATURE_BINARY2()],
                                ['sincnet', SINCNET_FEATURE(args=args,
                                                        num_eeg_channel=self.num_data_channel)
                                                        ]])

        # --- Define Conv1 and Net based on enc_model ---
        if self.enc_model == 'psd1' or self.enc_model =='psd2':
            self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=(1,21), stride=(1,4))
            # Adjust net output channels if needed - AlexNet usually increases channels then decreases
            # The provided self.net ends with 64 channels, let's assume that's intended.
            self.net = nn.Sequential(
                nn.ReLU(),
                nn.LocalResponseNorm(size=25, alpha=0.0001, beta=0.75, k=1),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
                nn.Conv2d(96, 128, (1,7), padding=(0,3)),
                nn.ReLU(),
                nn.LocalResponseNorm(size=25, alpha=0.0001, beta=0.75, k=1),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,1)),
                nn.Conv2d(128, 256, (1,7), padding=(0,3)),
                nn.ReLU(),
                nn.Conv2d(256, 128, (1,7), padding=(0,3)),
                nn.ReLU(),
                nn.Conv2d(128, 64, (1,7), padding=(0,3)), # Ends with 64 channels
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            )
        else: # Logic for raw, sincnet, stft1, stft2
            if self.enc_model == 'raw':
                self.features = False
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,51), stride=(1,4))
            elif self.enc_model == 'sincnet':
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,21), stride=(7,2))
            elif self.enc_model == 'stft1':
                # Original code had 96 out_channels here, mismatch with net input (64)
                # Assuming self.net below always expects 64 input channels after conv1+ReLU+LRN+Pool
                # If conv1 output is 96, need to adjust first layer of self.net
                # Let's keep conv1 output 64 to match potential self.net input expectation
                self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3) # Added padding assumption
                # self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=7, stride=2) # Original
            elif self.enc_model == 'stft2':
                 # Same channel mismatch potential as stft1
                self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3) # Added padding assumption
                # self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=96, kernel_size=7, stride=2) # Original
            else:
                raise ValueError(f'Unsupported feature extractor chosen: {self.enc_model}')

            # Define self.net (assuming it takes 64 channels after initial layers)
            # Note: The original code puts ReLU+LRN+MaxPool *after* conv1 for these cases.
            self.conv1_post = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=25, alpha=0.0001, beta=0.75, k=1), # LRN on 64 channels
                nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
            )
            # self.net now starts from the output of conv1_post (still 64 channels)
            self.net = nn.Sequential(
                # Input has 64 channels (from conv1_post)
                nn.Conv2d(64, 128, (1,15), padding=(0,7)),  # Output: 128 channels
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(size=25, alpha=0.0001, beta=0.75, k=1),
                nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)), # Output: 128 channels

                nn.Conv2d(128, 256, (1,15), padding=(0,7)), # Output: 256 channels
                nn.ReLU(inplace=True),
                # Optional LRN here? Doesn't affect channels
                nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)), # Output: 256 channels

                # --- Problem Area ---
                # This layer SHOULD output 128 channels:
                nn.Conv2d(256, 128, (1,15), padding=(0,7)), # Output: 128 channels (as defined)
                nn.ReLU(inplace=True), # Output: 128 channels

                # This layer EXPECTS 128 channels, but error says it gets 64:
                nn.Conv2d(128, 64, (1,15), padding=(0,7)), # <--- FAILING HERE
                # --- End Problem Area ---

                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool2d((1, 1)) # Or the removed pooling layer
            )

        # --- Dynamically determine the flattened size ---
        in_features_fc1 = self._get_conv_output_size(args, device)

        # --- Define Classifier using the dynamic size ---
        self.fc1 = nn.Linear(in_features=in_features_fc1, out_features=in_features_fc1 // 2)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True), # Standard dropout value for AlexNet FC layers
            nn.Linear(in_features_fc1 // 2, in_features_fc1 // 2),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True), # Add dropout here too
            nn.Linear(in_features_fc1 // 2, self.num_classes),
        )

        self.init_weights() # Use a standard weight init function

    # Helper method to calculate output size
    def _get_conv_output_size(self, args, device):
        # Create a dummy input matching the expected input format *after* permute in forward
        bs = 1 # Dummy batch size
        dummy_seq_len = getattr(args, 'window_size', 256) # Now set to 128
        if dummy_seq_len == 0: # Add a check for invalid window size
            raise ValueError("args.window_size is 0, cannot create dummy input.")
        if self.num_data_channel == 0: # Add a check for invalid channel count
             raise ValueError("args.num_channel is 0, cannot create dummy input.")

        # --- FIX HERE: Keep dummy_input on CPU during this calculation ---
        # Layer parameters are on CPU at this point in __init__
        dummy_input_cpu = torch.randn(bs, self.num_data_channel, dummy_seq_len)
        # ----------------------------------------------------------------

        # Pass through feature extractor if used
        self.eval() # Set to eval mode for calculation
        with torch.no_grad():
            # --- Use the CPU dummy input ---
            x = dummy_input_cpu
            # -----------------------------

            if self.enc_model != "raw":
                 # Note: Feature extractors might need specific device handling
                 # if they have internal parameters not part of the main model structure
                 # But generally, applying them to a CPU tensor should work here.
                x = self.feature_extractor[self.enc_model](x)
                if self.enc_model == "sincnet":
                     x = x.reshape(x.shape[0], 1, self.args.num_channel*self.args.sincnet_bandnum, x.shape[-1])
                # Add potential unsqueezing/reshaping for PSD/STFT if needed for Conv2d
            else: # raw model
                 x = torch.unsqueeze(x, dim=1) # x is still on CPU

            # Pass through convolutional layers (which are also on CPU at this stage)
            x = self.conv1(x)
            if hasattr(self, 'conv1_post'):
                 x = self.conv1_post(x)
            x = self.net(x)

        self.train() # Set back to train mode
        final_size = x.view(bs, -1).size(1)
        print(f"[{self.enc_model}] Detected conv output size: {x.shape}, Flattened: {final_size}") # Debug print
        return final_size


    # Replace init_bias with a more standard initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # AlexNet used normal init for FC
                nn.init.constant_(m.bias, 1) # AlexNet often initialized FC bias to 1
        # Special bias init from original code (careful if necessary)
        # You might need to access specific layers if the structure changed
        # Example: If net[3] is Conv2d: nn.init.constant_(self.net[3].bias, 1)


    def forward(self, x):
        # Input x expected shape: (batch_size, sequence_length, num_channels)

        # Permute to (batch_size, num_channels, sequence_length) - suitable for 1D processing/feature extraction
        x = x.permute(0, 2, 1)

        # Feature Extraction
        if self.args.enc_model != "raw":
            x = self.feature_extractor[self.args.enc_model](x)
            # Reshape/prepare for Conv2d (NCHW format)
            if self.args.enc_model == "sincnet":
                 # Assuming SincNet output (N, C_out_sinc, L_out_sinc)
                 # Reshape to (N, 1, C_out_sinc, L_out_sinc) for Conv2d kernel (7, 21)
                x = x.reshape(x.shape[0], 1, self.args.num_channel*self.args.sincnet_bandnum, x.shape[-1])
            # Add handling for PSD/STFT outputs if they aren't already NCHW
            # elif self.args.enc_model.startswith('psd'): x = x.unsqueeze(-1) # (N, C, H, 1)
            # elif self.args.enc_model.startswith('stft'): pass # Assumed NCHW already
        else:
             # Raw input (N, C, L) -> Unsqueeze for Conv2d (N, 1, C, L)? -> Needs check based on conv1 kernel size
             # Conv1 for raw: kernel=(1,51). Expects (N, C_in=1, H_in, W_in)
             # If x is (N, C, L), maybe need to view as (N, 1, C, L)?
             x = torch.unsqueeze(x, dim=1) # Assumes (N, 1, num_channel, seq_len)

        # Convolutional Layers
        x = self.conv1(x)
        if hasattr(self, 'conv1_post'): # Apply post-conv1 layers if defined
            x = self.conv1_post(x)
        x = self.net(x)

        # Flatten and Classify
        x = x.view(x.shape[0], -1)  # Flatten C, H, W dimensions
        x = self.fc1(x)
        x = self.classifier(x)

        return x, 0 # Return 0 for maps/consistency? OK if not used.

    def init_state(self, device):
        return 0 # Seems unused, OK.