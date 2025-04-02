source GLSMEnv/bin/activate

python src/train_quantizer.py --augmentation pitch_shift --augment_parameter default || true
python src/train_quantizer.py --augmentation pitch_shift --augment_parameter UED30 || true
python src/train_quantizer.py --augmentation time_stretch --augment_parameter default || true
python src/train_quantizer.py --augmentation time_stretch --augment_parameter UED30 || true
python src/train_quantizer.py --augmentation reverb --augment_parameter default || true
python src/train_quantizer.py --augmentation clip --augment_parameter default || true
python src/train_quantizer.py --augmentation lowpass --augment_parameter default || true
python src/train_quantizer.py --augmentation noise --augment_parameter default || true
python src/train_quantizer.py --augmentation noise --augment_parameter UED30 || true