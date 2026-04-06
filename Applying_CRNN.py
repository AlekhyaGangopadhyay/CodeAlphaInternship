"""
================================================================================
CRNN (Convolutional Recurrent Neural Network) for Handwritten Word Recognition
================================================================================

Architecture:
  - CNN Backbone: Multi-layer Conv2D for visual feature extraction
  - RNN Sequence Modeler: Bidirectional LSTM for sequence dependencies
  - CTC Decoder: Connectionist Temporal Classification for variable-length output

Dataset: IAM Handwriting Words Dataset
  - Images: Grayscale handwritten word crops (PNG)
  - Labels: Transcriptions from words_new.txt
"""

import os
import sys
import time
import random
import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """All hyperparameters and paths in one place."""
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    WORDS_DIR = os.path.join(BASE_DIR, "iam_words", "words")
    LABELS_FILE = os.path.join(BASE_DIR, "words_new.txt")
    SAVE_DIR = os.path.join(BASE_DIR, "crnn_output")

    # Image preprocessing
    IMG_HEIGHT = 32        # Fixed height for all images
    IMG_WIDTH = 128        # Fixed width (padded/resized)

    # Model architecture
    CNN_OUTPUT_CHANNELS = 512
    RNN_HIDDEN_SIZE = 256
    RNN_NUM_LAYERS = 2
    DROPOUT = 0.3

    # Training
    BATCH_SIZE = 64
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    LR_STEP_SIZE = 10
    LR_GAMMA = 0.1
    MAX_LABEL_LENGTH = 32  # Max characters in a word
    NUM_WORKERS = 0        # DataLoader workers (0 for Windows compatibility)
    TRAIN_SPLIT = 0.9      # 90% train, 10% validation
    MAX_SAMPLES = None     # Set to integer to limit dataset size (e.g., 10000 for quick test)

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    SEED = 42


def set_seed(seed):
    """Set reproducibility seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Character Encoding
# ============================================================================

class CharsetEncoder:
    """
    Maps characters to integer indices and back.
    Index 0 is reserved for CTC blank token.
    """

    def __init__(self):
        # Build charset from printable ASCII characters commonly found in IAM
        chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
        chars += list(".,;:'\"!?-/()&# ")
        # Remove duplicates, sort for consistency
        self.chars = sorted(set(chars))
        # 0 = CTC blank
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.chars)}
        self.blank_idx = 0
        self.num_classes = len(self.chars) + 1  # +1 for blank

    def encode(self, text):
        """Convert text string to list of indices."""
        encoded = []
        for ch in text:
            if ch in self.char_to_idx:
                encoded.append(self.char_to_idx[ch])
            # Skip unknown characters
        return encoded

    def decode(self, indices):
        """Convert indices back to text (CTC greedy decode)."""
        chars = []
        prev_idx = self.blank_idx
        for idx in indices:
            if idx != self.blank_idx and idx != prev_idx:
                if idx in self.idx_to_char:
                    chars.append(self.idx_to_char[idx])
            prev_idx = idx
        return "".join(chars)

    def decode_batch(self, log_probs):
        """
        Greedy CTC decoding for a batch.
        log_probs: (T, N, C) - time steps, batch, classes
        Returns list of decoded strings.
        """
        # Get best path
        _, max_indices = torch.max(log_probs, dim=2)  # (T, N)
        max_indices = max_indices.permute(1, 0).cpu().numpy()  # (N, T)

        decoded = []
        for seq in max_indices:
            decoded.append(self.decode(seq))
        return decoded


# ============================================================================
# Dataset
# ============================================================================

class IAMWordsDataset(Dataset):
    """
    PyTorch Dataset for IAM Words.

    Parses the label file, maps word IDs to image paths,
    and loads/preprocesses images on-the-fly.
    """

    def __init__(self, samples, charset, img_height=32, img_width=128):
        self.samples = samples  # List of (img_path, transcription)
        self.charset = charset
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, transcription = self.samples[idx]

        # Load image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Return a blank image if loading fails
            img = np.ones((self.img_height, self.img_width), dtype=np.uint8) * 255

        # Resize maintaining aspect ratio, pad to fixed width
        img = self._preprocess(img)

        # Normalize to [0, 1] and add channel dimension
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, H, W)

        # Encode transcription
        label = self.charset.encode(transcription)
        label_length = len(label)

        return (
            torch.FloatTensor(img),
            torch.IntTensor(label),
            label_length
        )

    def _preprocess(self, img):
        """Resize to fixed height, pad/crop to fixed width."""
        h, w = img.shape

        # Resize height to target, scale width proportionally
        new_w = max(1, int(w * self.img_height / h))
        img = cv2.resize(img, (new_w, self.img_height), interpolation=cv2.INTER_AREA)

        # Pad or crop width
        if new_w < self.img_width:
            # Pad right with white (255)
            pad_w = self.img_width - new_w
            img = np.pad(img, ((0, 0), (0, pad_w)), mode='constant', constant_values=255)
        elif new_w > self.img_width:
            # Resize to exact width
            img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)

        return img


def collate_fn(batch):
    """
    Custom collate function for variable-length labels.
    Returns:
        images: (N, 1, H, W) tensor
        labels: concatenated 1D tensor of all labels
        label_lengths: (N,) tensor of each label's length
        input_lengths: (N,) tensor of input sequence lengths (from CNN output width)
    """
    images, labels, label_lengths = zip(*batch)

    images = torch.stack(images, 0)
    labels = torch.cat(labels, 0)
    label_lengths = torch.IntTensor(label_lengths)

    return images, labels, label_lengths


def parse_labels_and_build_samples(config):
    """
    Parse the IAM words label file and build (image_path, transcription) pairs.
    """
    samples = []
    skipped = 0

    print(f"[DATA] Parsing labels from: {config.LABELS_FILE}")
    print(f"[DATA] Looking for images in: {config.WORDS_DIR}")

    with open(config.LABELS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 9:
                skipped += 1
                continue

            word_id = parts[0]       # e.g., a01-000u-00-00
            status = parts[1]        # ok or err
            transcription = parts[-1]  # Last field is the word transcription

            # Only use correctly segmented words
            if status != "ok":
                skipped += 1
                continue

            # Skip very short or very long labels
            if len(transcription) < 1 or len(transcription) > config.MAX_LABEL_LENGTH:
                skipped += 1
                continue

            # Build image path: a01-000u-00-00 -> words/a01/a01-000u/a01-000u-00-00.png
            parts_id = word_id.split('-')
            folder1 = parts_id[0]                          # a01
            folder2 = parts_id[0] + '-' + parts_id[1]      # a01-000u
            img_filename = word_id + '.png'
            img_path = os.path.join(config.WORDS_DIR, folder1, folder2, img_filename)

            if os.path.exists(img_path):
                samples.append((img_path, transcription))
            else:
                skipped += 1

    print(f"[DATA] Valid samples found: {len(samples)}")
    print(f"[DATA] Skipped entries: {skipped}")

    if config.MAX_SAMPLES and len(samples) > config.MAX_SAMPLES:
        random.shuffle(samples)
        samples = samples[:config.MAX_SAMPLES]
        print(f"[DATA] Limited to {config.MAX_SAMPLES} samples")

    return samples


# ============================================================================
# CRNN Model Architecture
# ============================================================================

class CRNN(nn.Module):
    """
    CRNN = CNN (feature extraction) + RNN (sequence modeling) + Linear (CTC output)

    Architecture Overview:
    ┌─────────────────────────────────────────────┐
    │  Input: (N, 1, 32, 128) grayscale images    │
    ├─────────────────────────────────────────────┤
    │  CNN Block 1: Conv(64) → BN → ReLU → Pool  │
    │  CNN Block 2: Conv(128) → BN → ReLU → Pool │
    │  CNN Block 3: Conv(256) → BN → ReLU         │
    │  CNN Block 4: Conv(256) → BN → ReLU → Pool │
    │  CNN Block 5: Conv(512) → BN → ReLU         │
    │  CNN Block 6: Conv(512) → BN → ReLU → Pool │
    │  CNN Block 7: Conv(512) → BN → ReLU         │
    ├─────────────────────────────────────────────┤
    │  Reshape: (N, C, 1, W') → (W', N, C)       │
    ├─────────────────────────────────────────────┤
    │  BiLSTM Layer 1: 512 → 256×2                │
    │  BiLSTM Layer 2: 512 → 256×2                │
    ├─────────────────────────────────────────────┤
    │  Linear: 512 → num_classes                  │
    │  LogSoftmax over classes                     │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, num_classes, img_height=32, img_width=128,
                 rnn_hidden=256, rnn_layers=2, dropout=0.3):
        super(CRNN, self).__init__()

        self.img_height = img_height
        self.img_width = img_width

        # ── CNN Feature Extractor ──
        # Designed to reduce height to 1 while preserving width information

        self.cnn = nn.Sequential(
            # Block 1: (1, 32, 128) → (64, 16, 64)
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (64, 16, 64) → (128, 8, 32)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (128, 8, 32) → (256, 8, 32)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4: (256, 8, 32) → (256, 4, 16)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: (256, 4, 16) → (512, 4, 16)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 6: (512, 4, 16) → (512, 2, 16)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Only pool height

            # Block 7: (512, 2, 16) → (512, 1, 16)
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ── RNN Sequence Modeler ──
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0,
            batch_first=False
        )

        # ── Output Layer ──
        self.output = nn.Linear(rnn_hidden * 2, num_classes)  # *2 for bidirectional
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: (N, 1, H, W) input images
        Returns:
            log_probs: (T, N, C) log probabilities for CTC
        """
        # CNN: (N, 1, H, W) → (N, 512, 1, W')
        conv = self.cnn(x)

        # Reshape for RNN: (N, C, 1, W') → (N, C, W') → (W', N, C)
        batch, channels, height, width = conv.size()
        assert height == 1, f"CNN output height should be 1, got {height}"
        conv = conv.squeeze(2)       # (N, C, W')
        conv = conv.permute(2, 0, 1)  # (W', N, C) = (T, N, C)

        # RNN: (T, N, C) → (T, N, H*2)
        rnn_out, _ = self.rnn(conv)

        # Output: (T, N, H*2) → (T, N, num_classes)
        output = self.output(rnn_out)
        log_probs = self.log_softmax(output)

        return log_probs

    def get_output_length(self):
        """Calculate CNN output width for CTC input_lengths."""
        # Width: 128 → pool/2 → 64 → pool/2 → 32 → pool/2 → 16 → (no width pool) → 16 → 16
        # Width reductions: 128 / 2 / 2 / 2 = 16
        return self.img_width // 8


# ============================================================================
# Metrics
# ============================================================================

def calculate_cer(predicted, target):
    """
    Character Error Rate using edit distance.
    CER = edit_distance(pred, target) / len(target)
    """
    if len(target) == 0:
        return 0.0 if len(predicted) == 0 else 1.0

    # Dynamic programming edit distance
    m, n = len(predicted), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predicted[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[m][n] / n


def calculate_wer(predicted, target):
    """
    Word Error Rate.
    For single words: 0 if exact match, 1 otherwise.
    """
    return 0.0 if predicted == target else 1.0


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, charset, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cer = 0
    total_samples = 0
    num_batches = len(dataloader)

    for batch_idx, (images, labels, label_lengths) in enumerate(dataloader):
        images = images.to(device)

        # Forward pass
        log_probs = model(images)  # (T, N, C)
        T, N, C = log_probs.size()

        # Input lengths: all sequences have the same length from CNN
        input_lengths = torch.full((N,), T, dtype=torch.int32)

        # CTC Loss
        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [WARN] NaN/Inf loss at batch {batch_idx}, skipping...")
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * N

        # Calculate CER for monitoring (every 50 batches)
        if batch_idx % 50 == 0:
            with torch.no_grad():
                decoded = charset.decode_batch(log_probs)
                # Reconstruct ground truth strings
                gt_strings = []
                offset = 0
                for length in label_lengths:
                    gt_indices = labels[offset:offset + length].numpy()
                    gt_str = "".join([charset.idx_to_char.get(idx, '') for idx in gt_indices])
                    gt_strings.append(gt_str)
                    offset += length

                batch_cer = np.mean([calculate_cer(p, g) for p, g in zip(decoded, gt_strings)])

            print(f"  Epoch [{epoch+1}/{total_epochs}] Batch [{batch_idx+1}/{num_batches}] "
                  f"Loss: {loss.item():.4f} | CER: {batch_cer:.4f}")

        total_samples += N

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss


def evaluate(model, dataloader, criterion, device, charset):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_cer = 0
    total_wer = 0
    total_samples = 0
    sample_predictions = []

    with torch.no_grad():
        for images, labels, label_lengths in dataloader:
            images = images.to(device)

            log_probs = model(images)
            T, N, C = log_probs.size()

            input_lengths = torch.full((N,), T, dtype=torch.int32)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * N

            # Decode predictions
            decoded = charset.decode_batch(log_probs)

            # Reconstruct ground truth strings
            offset = 0
            for i, length in enumerate(label_lengths):
                gt_indices = labels[offset:offset + length].numpy()
                gt_str = "".join([charset.idx_to_char.get(idx, '') for idx in gt_indices])

                cer = calculate_cer(decoded[i], gt_str)
                wer = calculate_wer(decoded[i], gt_str)
                total_cer += cer
                total_wer += wer

                # Save some sample predictions
                if len(sample_predictions) < 20:
                    sample_predictions.append((gt_str, decoded[i], cer))

                offset += length

            total_samples += N

    avg_loss = total_loss / max(total_samples, 1)
    avg_cer = total_cer / max(total_samples, 1)
    avg_wer = total_wer / max(total_samples, 1)

    return avg_loss, avg_cer, avg_wer, sample_predictions


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    config = Config()
    set_seed(config.SEED)

    # Create output directory
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    log_path = os.path.join(config.SAVE_DIR, "training_log.txt")

    def log(msg):
        print(msg)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log("=" * 70)
    log("  CRNN Training for Handwritten Word Recognition (IAM Dataset)")
    log("=" * 70)
    log(f"  Device: {config.DEVICE}")
    log(f"  Image size: {config.IMG_HEIGHT}x{config.IMG_WIDTH}")
    log(f"  Batch size: {config.BATCH_SIZE}")
    log(f"  Epochs: {config.NUM_EPOCHS}")
    log(f"  Learning rate: {config.LEARNING_RATE}")
    log(f"  RNN hidden: {config.RNN_HIDDEN_SIZE} × 2 (BiLSTM)")
    log(f"  RNN layers: {config.RNN_NUM_LAYERS}")
    log("")

    # ── Build character encoder ──
    charset = CharsetEncoder()
    log(f"[CHARSET] {charset.num_classes} classes (including CTC blank)")
    log(f"[CHARSET] Characters: {''.join(charset.chars)}")
    log("")

    # ── Load and parse dataset ──
    all_samples = parse_labels_and_build_samples(config)

    if len(all_samples) == 0:
        log("[ERROR] No valid samples found! Check paths and label file.")
        return

    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * config.TRAIN_SPLIT)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    log(f"[SPLIT] Training: {len(train_samples)} samples")
    log(f"[SPLIT] Validation: {len(val_samples)} samples")
    log("")

    # ── Create datasets and dataloaders ──
    train_dataset = IAMWordsDataset(train_samples, charset,
                                     config.IMG_HEIGHT, config.IMG_WIDTH)
    val_dataset = IAMWordsDataset(val_samples, charset,
                                   config.IMG_HEIGHT, config.IMG_WIDTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )

    # ── Build model ──
    model = CRNN(
        num_classes=charset.num_classes,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        rnn_hidden=config.RNN_HIDDEN_SIZE,
        rnn_layers=config.RNN_NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(config.DEVICE)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[MODEL] CRNN Architecture:")
    log(f"  Total parameters: {total_params:,}")
    log(f"  Trainable parameters: {trainable_params:,}")
    log(f"  CNN output width (T): {model.get_output_length()}")
    log("")

    # ── Loss, optimizer, scheduler ──
    criterion = nn.CTCLoss(blank=charset.blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP_SIZE,
                                           gamma=config.LR_GAMMA)

    # ── Training loop ──
    best_cer = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_cer': [], 'val_wer': []}

    log("=" * 70)
    log("  Starting Training")
    log("=" * 70)

    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            config.DEVICE, charset, epoch, config.NUM_EPOCHS
        )

        # Validate
        val_loss, val_cer, val_wer, sample_preds = evaluate(
            model, val_loader, criterion, config.DEVICE, charset
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_cer'].append(val_cer)
        history['val_wer'].append(val_wer)

        # Log epoch results
        log("-" * 70)
        log(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        log(f"  Train Loss: {train_loss:.4f}")
        log(f"  Val Loss:   {val_loss:.4f}")
        log(f"  Val CER:    {val_cer:.4f} ({val_cer*100:.2f}%)")
        log(f"  Val WER:    {val_wer:.4f} ({val_wer*100:.2f}%)")

        # Show sample predictions
        log(f"\n  Sample Predictions (GT → Predicted):")
        for gt, pred, cer in sample_preds[:10]:
            marker = "✓" if gt == pred else "✗"
            log(f"    {marker} '{gt}' → '{pred}' (CER: {cer:.2f})")
        log("")

        # Save best model
        if val_cer < best_cer:
            best_cer = val_cer
            best_model_path = os.path.join(config.SAVE_DIR, "crnn_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer': val_cer,
                'val_wer': val_wer,
                'charset_chars': charset.chars,
            }, best_model_path)
            log(f"  ★ New best model saved! CER: {val_cer:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(config.SAVE_DIR, f"crnn_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer': val_cer,
            }, ckpt_path)
            log(f"  Checkpoint saved: {ckpt_path}")

    # ── Final Summary ──
    log("\n" + "=" * 70)
    log("  Training Complete!")
    log("=" * 70)
    log(f"  Best Validation CER: {best_cer:.4f} ({best_cer*100:.2f}%)")
    log(f"  Best model saved to: {os.path.join(config.SAVE_DIR, 'crnn_best.pth')}")

    # Save training history
    history_path = os.path.join(config.SAVE_DIR, "training_history.txt")
    with open(history_path, 'w', encoding='utf-8') as f:
        f.write("epoch,train_loss,val_loss,val_cer,val_wer\n")
        for i in range(len(history['train_loss'])):
            f.write(f"{i+1},{history['train_loss'][i]:.6f},{history['val_loss'][i]:.6f},"
                    f"{history['val_cer'][i]:.6f},{history['val_wer'][i]:.6f}\n")
    log(f"  Training history saved to: {history_path}")

    # ── Final evaluation with detailed results ──
    log("\n" + "=" * 70)
    log("  Final Evaluation on Validation Set")
    log("=" * 70)

    # Load best model for final evaluation
    best_ckpt = torch.load(os.path.join(config.SAVE_DIR, "crnn_best.pth"),
                           map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(best_ckpt['model_state_dict'])

    val_loss, val_cer, val_wer, sample_preds = evaluate(
        model, val_loader, criterion, config.DEVICE, charset
    )

    log(f"  Final Val Loss: {val_loss:.4f}")
    log(f"  Final Val CER:  {val_cer:.4f} ({val_cer*100:.2f}%)")
    log(f"  Final Val WER:  {val_wer:.4f} ({val_wer*100:.2f}%)")
    log(f"\n  Sample Predictions:")
    for gt, pred, cer in sample_preds:
        marker = "✓" if gt == pred else "✗"
        log(f"    {marker} '{gt}' → '{pred}' (CER: {cer:.2f})")

    log("\n" + "=" * 70)
    log("  Done! All outputs saved to: " + config.SAVE_DIR)
    log("=" * 70)


if __name__ == "__main__":
    main()
