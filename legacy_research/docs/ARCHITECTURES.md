# Architectures

The source CIFAR implementations are retained in `models/`: ResNet-56, VGG-16-BN, and MobileNetV2. Constructors accept `num_classes`; MobileNetV2 also accepts `width_mult`. They consume 3-channel image batches and emit one logit vector per input. CIFAR configs use 32×32 images. Custom datasets can resize to the configured `image_size`; verify model behavior if using dimensions unlike the CIFAR default.
