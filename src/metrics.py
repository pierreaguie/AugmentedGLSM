import torch
import torch.nn.functional as F
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm

from transform import add_noise, add_reverb, pitch_shift, time_stretch


def unit_edit_distance(dataset, encoder_quantizer, augmentation):
    """
    Compute the deduplicated Unit Edit Distance (UED).
    
    Args:
        dataset (list of torch.Tensor): List of input samples x from the evaluation set D.
        encoder (callable): Continuous encoder f: R^T → R^T'.
        quantizer (callable): Quantizer E: R^T' → {1, ..., K}^T'.
        augmenter (callable): Input augmentation g: R^T' → R^T'.
    
    Returns:
        float: The UED score.
    """
    total_ued = 0.0
    total_frames = 0
    
    for x, sample_rate in tqdm(dataset):
        # Encode the input
        x = x.detach()
        encoded_x = encoder_quantizer(x)
        units = encoded_x["units"]

        #Encode the augmented input
        if augmentation=='time_stretch':
            augmented_x = time_stretch(x, sample_rate)
        elif augmentation=='pitch_shift':
            augmented_x = pitch_shift(x, sample_rate)
        elif augmentation=='add_noise':
            augmented_x = add_noise(x)
        elif augmentation=='add_reverb':
            augmented_x = add_reverb(x, sample_rate)
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation}")
        augmented_x = augmented_x.detach()
        encoded_augmented_x = encoder_quantizer(augmented_x)
        units_augmented = encoded_augmented_x["units"]

        # Compute Levenshtein distance
        lev_dist = levenshtein_distance(units.tolist(), units_augmented.tolist())
        
        # Normalize by the number of frames
        T_prime_x = len(units)
        total_ued += lev_dist / T_prime_x
        total_frames += 1
    
    return total_ued / total_frames if total_frames > 0 else 0.0


#Compute UED with pre-augmented dataset
def unit_edit_distance_from_two_datasets(dataset, dataset_augmented, encoder_quantizer, store_file, verbose=False):

    total_ued = 0.0
    total_frames = 0
    n = len(dataset)
    
    for i in tqdm(range(n)):
        try:
            x, _ = dataset[i]
            augmented_x, _ = dataset_augmented[i]
            if x is not None and augmented_x is not None:
                # Encode the input
                x = x.detach()
                augmented_x = augmented_x.detach()
                encoded_x = encoder_quantizer(x)

                units = encoded_x["units"]

                encoded_augmented_x = encoder_quantizer(augmented_x)

                units_augmented = encoded_augmented_x["units"]

                # Compute Levenshtein distance
                lev_dist = levenshtein_distance(units.tolist(), units_augmented.tolist())
                
                # Normalize by the number of frames
                T_prime_x = len(units)
                total_ued += lev_dist / T_prime_x
                total_frames += 1
        except Exception as e:
            print(f"Erreur à l'itération {i}: {e}")
            continue

        if verbose:
            if i%20==0:
                print('current ued : ', total_ued/total_frames)
    
    # Write the current UED to a .txt file
    with open(f"{store_file}", "w") as file:
        file.write(f"Current UED: {total_ued / total_frames if total_frames > 0 else 0.0}\n")
    
    return total_ued / total_frames if total_frames > 0 else 0.0