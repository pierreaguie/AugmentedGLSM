

source GLSMEnv/bin/activate

# Generate datasets with pitch shifting at different semi-tones
python src/create_dataset.py --augment_parameter 1
python src/create_dataset.py --augment_parameter 2
python src/create_dataset.py --augment_parameter 3
python src/create_dataset.py --augment_parameter 4
python src/create_dataset.py --augment_parameter -1
python src/create_dataset.py --augment_parameter -2
python src/create_dataset.py --augment_parameter -3
python src/create_dataset.py --augment_parameter -4