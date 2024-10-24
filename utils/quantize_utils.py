from typing import List, Tuple

import numpy as np
from numpy import float64, int64, ndarray

from collections import OrderedDict

import torch
import copy
from utils.homomorphic_encryption_utils import HEScheme
from typing import Dict
from tenseal.tensors.bfvvector import BFVVector
from collections import OrderedDict


def unquantize_matrix(
        matrix: ndarray, bit_width: int = 8, r_max: float64 = 0.5
) -> ndarray:
    matrix = matrix.astype(int)
    og_sign = np.sign(matrix)
    uns_matrix = np.multiply(matrix, og_sign)
    uns_result = np.multiply(
        uns_matrix, np.divide(r_max, (pow(2, bit_width - 1) - 1.0))
    )
    result = og_sign * uns_result
    return result.astype(np.float32)


def quantize_matrix_stochastic(
        matrix: ndarray, bit_width: int = 8, r_max: float64 = 0.5
) -> Tuple[ndarray, ndarray]:
    og_sign = np.sign(matrix)
    uns_matrix = np.multiply(matrix, og_sign)
    uns_result = np.multiply(
        uns_matrix, np.divide((pow(2, bit_width - 1) - 1.0), r_max)
    )
    result = np.multiply(og_sign, uns_result)
    return result, og_sign


def quantize_per_layer(party, r_maxs, bit_width=16):
    result = []
    for component, r_max in zip(party, r_maxs):
        x, _ = quantize_matrix_stochastic(component, bit_width=bit_width, r_max=r_max)
        result.append(x)

    return np.array(result, dtype=object)


def clip_with_threshold(grads, thresholds):
    return [np.clip(x, -1 * y, y) for x, y in zip(grads, thresholds)]


def get_alpha_gaus(values: ndarray, values_size: int64, num_bits: int) -> float64:
    """
    Calculating optimal alpha(clipping value) in Gaussian case

    Args:
    values: input ndarray
    num_bits: number of bits used for quantization

    Return:
    Optimal clipping value
    """

    # Dictionary that stores optimal clipping values for N(0, 1)
    alpha_gaus = {
        2: 1.71063516,
        3: 2.02612148,
        4: 2.39851063,
        5: 2.76873681,
        6: 3.12262004,
        7: 3.45733738,
        8: 3.77355322,
        9: 4.07294252,
        10: 4.35732563,
        11: 4.62841243,
        12: 4.88765043,
        13: 5.1363822,
        14: 5.37557768,
        15: 5.60671468,
        16: 5.82964388,
        17: 6.04501354,
        18: 6.25385785,
        19: 6.45657762,
        20: 6.66251328,
        21: 6.86053901,
        22: 7.04555454,
        23: 7.26136857,
        24: 7.32861916,
        25: 7.56127906,
        26: 7.93151212,
        27: 7.79833847,
        28: 7.79833847,
        29: 7.9253003,
        30: 8.37438905,
        31: 8.37438899,
        32: 8.37438896,
    }
    # That's how ACIQ paper calculate sigma, based on the range (efficient but not accurate)
    gaussian_const = (0.5 * 0.35) * (1 + (np.pi * np.log(4)) ** 0.5)
    # sigma = ((np.max(values) - np.min(values)) * gaussian_const) / ((2 * np.log(values.size)) ** 0.5)
    sigma = ((np.max(values) - np.min(values)) * gaussian_const) / (
            (2 * np.log(values_size)) ** 0.5
    )
    return alpha_gaus[num_bits] * sigma


def calculate_clip_threshold_aciq_g(
        grads: ndarray, grads_sizes: List[int64], bit_width: int = 8
) -> List[float64]:
    res = []
    for idx in range(len(grads)):
        res.append(get_alpha_gaus(grads[idx], grads_sizes[idx], bit_width))
    # return [aciq.get_alpha_gaus(x, bit_width) for x in grads]
    return res


def quant_weights(clients_weights, num_clients):
    for idx in range(len(clients_weights)):
        clients_weights[idx] = [
            # clients_weights[idx][x].numpy() for x in clients_weights[idx]
            clients_weights[idx][x].cpu().numpy() for x in clients_weights[idx]
        ]

        # clipping_thresholds = encryption.calculate_clip_threshold(grads_0)
    theta = 2.5
    clients_weights_mean = []
    clients_weights_mean_square = []
    for client_idx in range(len(clients_weights)):
        temp_mean = [
            np.mean(clients_weights[client_idx][layer_idx])
            for layer_idx in range(len(clients_weights[client_idx]))
        ]
        temp_mean_square = [
            np.mean(clients_weights[client_idx][layer_idx] ** 2)
            for layer_idx in range(len(clients_weights[client_idx]))
        ]
        clients_weights_mean.append(temp_mean)
        clients_weights_mean_square.append(temp_mean_square)
    clients_weights_mean = np.array(clients_weights_mean)
    clients_weights_mean_square = np.array(clients_weights_mean_square)

    layers_size = np.array([_.size for _ in clients_weights[0]])
    clipping_thresholds = (
            theta
            * np.sqrt(
        np.maximum(
            np.sum(clients_weights_mean_square * layers_size, 0) / (layers_size * num_clients) - (
                    np.sum(clients_weights_mean * layers_size, 0) / (layers_size * num_clients)
            ) ** 2,
            0
        )
    )
    )
    r_maxs = [x * num_clients for x in clipping_thresholds]

    clients_weights = [
        clip_with_threshold(item, clipping_thresholds) for item in clients_weights
    ]

    q_width = 64
    clients_weights = [
        quantize_per_layer(item, r_maxs, bit_width=q_width)
        for item in clients_weights
    ]

    return clients_weights, r_maxs


def dequant_weights(grads, r_maxs, q_width):
    grads = [grads[x].numpy() for x in grads]
    result = []
    for component, r_max in zip(grads, r_maxs):
        result.append(
            unquantize_matrix(component, bit_width=q_width, r_max=r_max).astype(
                np.float32
            )
        )
    return np.array(result, dtype=object)


def change_back_to_ordered_dict(clients_weights, layer_names):
    client_weights_ordereddict = []
    for client_weights in clients_weights:
        ordered_dict = OrderedDict()
        for i, layer_name in enumerate(layer_names):
            ordered_dict[layer_name] = client_weights[i]
        client_weights_ordereddict.append(ordered_dict)
    return client_weights_ordereddict


def aggregate_ckks_bfv(encrypted_weights, client_weights) -> Dict[str, BFVVector]:
    w_sum = {}
    for encrypted_client_weights, client_weight in zip(
            encrypted_weights, client_weights
    ):
        for key in encrypted_client_weights:
            if key not in w_sum:
                w_sum[key] = 0
            w_sum[key] = w_sum[key] + encrypted_client_weights[key] * client_weight
    return w_sum


from phe import paillier


def aggregate_paillier(encrypted_weights, client_weights) -> Dict[str, paillier.EncryptedNumber]:
    public_key, private_key = paillier.generate_paillier_keypair()

    w_sum = {}
    for encrypted_client_weights, client_weight in zip(encrypted_weights, client_weights):
        for key in encrypted_client_weights:
            if key not in w_sum:
                w_sum[key] = public_key.encrypt(0)
            w_sum[key] += encrypted_client_weights[key].encrypted_number * client_weight

    return w_sum


def weights_to_ordered_dict(clients_weights):
    clients_ordered_dicts = []
    for client_weights in clients_weights:
        client_od = OrderedDict()
        for layer_name, weights_tensor in client_weights.items():
            client_od[layer_name] = weights_tensor.cpu().numpy()
        clients_ordered_dicts.append(client_od)
    return clients_ordered_dicts


def aggregated_quantize_encrypt(local_weights, num_users, args):
    # Initialize homomorphic encryption
    he = HEScheme(he_scheme_name=args.he_scheme, poly_modulus_degree=args.poly_modulus_degree)
    if args.he_scheme == "ckks":
        context = he.context
        secret_key = (
            context.secret_key()
        )  # save the secret key before making context public
        context.make_context_public()  # make the context object public so it can be shared across clients
    elif args.he_scheme == "paillier":
        secret_key = he.private_key
    elif args.he_scheme == "bfv":
        context = he.context
        secret_key = (
            context.secret_key()
        )  # save the secret key before making context public
        context.make_context_public()

    local_weights = copy.deepcopy(local_weights)
    weight_shapes = {k: v.shape for k, v in local_weights[0].items()}
    # Quantize_weights
    quantize_weights, r_maxs = quant_weights(local_weights, num_users)
    quantize_weights = change_back_to_ordered_dict(quantize_weights, weight_shapes.keys())
    # Encrypt client weights
    encrypted_weights = he.encrypt_client_weights(quantize_weights)
    # Aggregate encrypted weights
    client_weights = [1] * len(encrypted_weights)
    global_averaged_encrypted_weights = aggregate_ckks_bfv(encrypted_weights, client_weights)
    # Decrypt and average the weights
    global_averaged_weights = he.decrypt_and_average_weights(
        global_averaged_encrypted_weights,
        weight_shapes,
        num_users,
        secret_key)
    # Dequantize_weights
    dequantize_weights = dequant_weights(global_averaged_weights, r_maxs, 64)
    ordered_dict = OrderedDict()
    for i, layer_name in enumerate(weight_shapes.keys()):
        ordered_dict[layer_name] = torch.from_numpy(dequantize_weights[i])

    return ordered_dict


def aggregated_encrypt(local_weights, num_users, args):
    """
    Aggregates encrypted weights for clients using the specified homomorphic encryption scheme.
    """
    # Initialize homomorphic encryption
    he = HEScheme(he_scheme_name=args.he_scheme, poly_modulus_degree=args.poly_modulus_degree)
    if args.he_scheme in ["ckks", "bfv"]:
        context = he.context
        secret_key = context.secret_key()  # save the secret key before making context public
        context.make_context_public()  # make the context object public so it can be shared across clients
    elif args.he_scheme == "paillier":
        secret_key = he.private_key  # Paillier uses private_key as secret_key

    # Prepare weights for encryption
    local_weights = copy.deepcopy(local_weights)
    weight_shapes = {k: v.shape for k, v in local_weights[0].items()}

    # Encrypt client weights
    ordered_dict_w = weights_to_ordered_dict(local_weights)
    encrypted_weights = he.encrypt_client_weights(ordered_dict_w)

    # Count the number of vectors (i.e., total number of weights)
    num_vectors = sum([np.prod(v.shape) for v in local_weights[0].values()])

    # Calculate the amount of data processed based on the encryption scheme
    if args.he_scheme == "ckks":
        # CKKS encrypts each vector into a ciphertext, and the size is dependent on poly_modulus_degree
        # Typically, the ciphertext size is 2 * poly_modulus_degree * log2(coeff_modulus) (in bits)
        coeff_modulus_log = 60  # Example value for coefficient modulus (log2(coeff_modulus))
        data_processed = 2 * args.poly_modulus_degree * coeff_modulus_log * num_vectors / 8  # Convert to bytes

    elif args.he_scheme == "paillier":
        # Paillier encryption increases the size by the key size (e.g., 2048 bits per encrypted value)
        paillier_key_size = 2048  # Assuming a 2048-bit Paillier key
        data_processed = num_vectors * paillier_key_size / 8  # Convert to bytes

    elif args.he_scheme == "bfv":
        # Similar to CKKS but with its own parameters
        # Ciphertext size for BFV depends on the poly_modulus_degree and the encryption parameters
        coeff_modulus_log = 60  # Example value for coefficient modulus (log2(coeff_modulus))
        data_processed = 2 * args.poly_modulus_degree * coeff_modulus_log * num_vectors / 8  # Convert to bytes

    else:
        raise ValueError(f"Unsupported encryption scheme: {args.he_scheme}")

    # Print the number of vectors and the data processed based on the encryption scheme
    # print(f"Number of vectors (weights): {num_vectors}")
    # print(f"Amount of data processed with {args.he_scheme}: {data_processed / (1024 ** 2):.2f} MB")  # Convert bytes to MB

    # Aggregate encrypted weights
    client_weights = [1] * len(encrypted_weights)

    if args.he_scheme in ["ckks", "bfv"]:
        global_averaged_encrypted_weights = aggregate_ckks_bfv(encrypted_weights, client_weights)
    elif args.he_scheme == "paillier":
        global_averaged_encrypted_weights = aggregate_paillier(encrypted_weights, client_weights)

    # Decrypt and average weights
    global_averaged_weights = he.decrypt_and_average_weights(
        global_averaged_encrypted_weights,
        weight_shapes,
        num_users,
        secret_key)

    # Convert weights back to an ordered dictionary form
    ordered_dict = OrderedDict(global_averaged_weights)

    return ordered_dict

