import argparse

def args_parser():

    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--num_rounds', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_clients', type=int, default=16, help="number of users")
    parser.add_argument('--batch_size', type=int, default=1024, help="batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--num_select', type=int, default=10, help="the selection of clients")

    # initialize he
    parser.add_argument('--he_scheme', type=str, default='ckks', help="he_scheme_name: ckks/paillier/bfv ")
    parser.add_argument('--poly_modulus_degree', type=int, default=8192, help="Polynomial Degree")

    # dateset
    parser.add_argument('--dataset', type=str, default='creditcard_train_SMOTE_2.csv', help="dataset")

    args = parser.parse_args()
    return args
