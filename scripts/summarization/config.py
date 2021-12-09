pg_train_config = {
    'vocab_size':50000, 'emb_dim':128, 'hidden_dim':256, 'max_dec_len':100,
    'enc_n_layers':1, 'dec_n_layers':1, 'max_oovs':0, 'pad_id': 0
}

configs = {'pointer-generator':pg_train_config}