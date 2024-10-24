from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import phe as paillier
import tenseal as ts
import torch
from tenseal.enc_context import SecretKey
from tenseal.tensors.bfvvector import BFVVector


# create graphs of what you are doing
# use <T> kind to initialize he
class HEScheme:
    def __init__(
            self,
            he_scheme_name: str,
            bits_scale: int = 40,
            # Polynomial Degree (N) 多项式度数 (N) 加密密文中系数的数量
            # poly_modulus_degree: int = 8192,
            # poly_modulus_degree: int = 4096,
            poly_modulus_degree: int = 16384,
            coeff_mod_bit_sizes: List[int] = [40, 40, 40, 40, 40],
            global_scale: int = 40,
            create_galois_keys: bool = False,
            # 模数 (q): 密文元素的大小
            plain_modulus: int = 1032193,
    ) -> None:
        self.bits_scale = bits_scale
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = global_scale
        self.plain_modulus = plain_modulus
        self.create_galois_keys = create_galois_keys

        if he_scheme_name == "ckks":
            self.init_ckks()
            self.encrypt_client_weights = self.encrypt_client_weights_ckks_bfv
            self.encrypt_feature = self.encrypt_client_weights_ckks_bfv
            self.decrypt_and_average_weights = self.decrypt_and_average_weights_ckks_bfv
            self.decrypt_and_average_feature = self.decrypt_and_average_weights_ckks_bfv
        elif he_scheme_name == "paillier":
            self.init_paillier()
            self.encrypt_client_weights = self.encrypt_client_weights_paillier
            self.encrypt_feature = self.encrypt_feature_paillier
            self.decrypt_and_average_weights = self.decrypt_and_average_weights_paillier
            self.decrypt_and_average_feature = self.decrypt_and_average_feature_paillier
        elif he_scheme_name == "bfv":
            self.init_bfv()
            self.encrypt_client_weights = self.encrypt_client_weights_ckks_bfv
            self.encrypt_feature = self.encrypt_client_weights_ckks_bfv
            self.decrypt_and_average_weights = self.decrypt_and_average_weights_ckks_bfv
            self.decrypt_and_average_feature = self.decrypt_and_average_weights_ckks_bfv

    def init_ckks(self) -> None:
        # controls precision of the fractional part
        bits_scale = self.bits_scale

        # Create TenSEAL context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            # coeff_mod_bit_sizes=[60, bits_scale, bits_scale, 60]
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes
            # coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
            # coeff_mod_bit_sizes = [31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
        )
        self.context.global_scale = 2 ** self.global_scale
        if self.create_galois_keys:
            self.context.generate_galois_keys()
        self.encrypt_function = ts.ckks_vector
        # self.decrypt_func =
        # pack all channels into a single flattened vector
        # enc_x = ts.CKKSVector.pack_vectors(enc_channels)

    def init_paillier(self):
        secret_number_list = [3.141592653, 300, -4.6e-12]
        self.public_key, self.private_key = paillier.generate_paillier_keypair()

    def init_bfv(self) -> None:
        # TODO Set correct hyperparameter values for BFV, convert float to int for processing
        self.context = ts.context(
            ts.SCHEME_TYPE.BFV, poly_modulus_degree=2 ** 13, plain_modulus=65537
        )
        # self.context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=2 ** 13, plain_modulus=536903681)
        self.encrypt_function = ts.bfv_vector
        # encrypted_vector = ts.bfv_vector(context, plain_vector)

    def serialize_encrypted_data(self):
        pass

    def deserialize_encrypted_data(self):
        pass

    def encrypt_client_weights_ckks_bfv(
            self, clients_weights: List[OrderedDict]
    ) -> list:
        encr = []
        for client_weights in clients_weights:
            encr_state_dict = {}
            # image_feature_extractor.extractor.0.0.weight
            for key, value in client_weights.items():
                val = value.flatten()
                encr_state_dict[key] = self.encrypt_function(self.context, val)
            encr.append(encr_state_dict)
        return encr

    def float_to_fixed_point(self, x, precision_bits):
        scaling_factor = 2 ** precision_bits
        x_scaled = x * scaling_factor
        x_int = x_scaled.apply_(np.round).to(torch.int64)

        return x_int

    def fixed_point_to_float(self, x_int, precision_bits):
        scaling_factor = 2 ** precision_bits
        x_float = x_int / scaling_factor

        return x_float

    # def encrypt_client_weights_paillier(self, clients_weights) -> list:
    #     all_enc_dicts = []

    #     enc_dict = dict()
    #     for clients_weight in clients_weights:
    #         for name, data in clients_weight.items():
    #             # params = data.cpu().numpy()
    #             # 如果data是PyTorch张量，则转换为NumPy数组
    #             if hasattr(data, 'cpu'):
    #                 params = data.cpu().numpy()
    #             else:
    #                 # 如果data已经是NumPy数组，直接使用
    #                 params = data
    #             enc_list = []
    #             for x in np.nditer(params):
    #                 x = self.public_key.encrypt(float(x))
    #                 enc_list.append(x)
    #             enc_dict[name] = enc_list
    #         all_enc_dicts.append(enc_dict)
    #     return all_enc_dicts

    def encrypt_client_weights_paillier(self, clients_weights) -> list:
        all_enc_dicts = []

        for clients_weight in clients_weights:
            enc_dict = dict()  # 移动此行到循环内部，以确保为每个客户端的权重创建新的字典
            for name, data in clients_weight.items():
                # 检查数据类型是否为PyTorch张量，如果是，则先转换到CPU，再转换为NumPy数组
                if hasattr(data, 'cpu'):
                    params = data.cpu().numpy()
                else:
                    # 如果数据已经是NumPy数组，则直接使用
                    params = data
                enc_list = []
                for x in np.nditer(params):
                    x = self.public_key.encrypt(float(x))
                    enc_list.append(x)
                enc_dict[name] = enc_list
            all_enc_dicts.append(enc_dict)
        return all_enc_dicts

    def decrypt_and_average_weights_ckks_bfv(
            self,
            encr_weights: Dict[str, BFVVector],
            shapes: Dict[str, torch.Size],
            client_weights: int,
            secret_key: Optional[SecretKey] = None,
    ) -> Dict[str, torch.Tensor]:
        decry_model = {}
        for key, value in encr_weights.items():
            decry_model[key] = torch.reshape(
                torch.tensor(value.decrypt(secret_key)), shapes[key]
            )
            # average weights
            decry_model[key] = torch.div(decry_model[key], client_weights)
        return decry_model

    def decrypt_and_average_weights_paillier(
            self, encr_weights, shapes, client_weights, secret_key=None
    ):
        decry_model = {}

        for key, value in encr_weights.items():
            dec_list = []
            for x in value:
                x = self.private_key.decrypt(x)
                dec_list.append(x)
            # reshape vector
            dec_params = np.asarray(dec_list, dtype=np.float32).reshape(shapes[key])
            decry_model[key] = torch.div(torch.from_numpy(dec_params), client_weights)

        return decry_model

    def encrypt_feature_ckks_bfv(self):
        pass

    def encrypt_feature_paillier(self):
        pass

    def decrypt_and_average_feature_ckks_bfv(self):
        pass

    def decrypt_and_average_feature_paillier(self):
        pass

    # 对单个参数进行加密
    def encrypt_single_value_ckks(self, value: float) -> ts.ckks_vector:
        encrypted_value = self.encrypt_function(self.context, [value])
        return encrypted_value

    def decrypt_single_value_ckks(self, encrypted_value: ts.ckks_vector,
                                  secret_key: Optional[SecretKey] = None) -> float:
        decrypted_values = encrypted_value.decrypt(secret_key)
        return decrypted_values[0]
