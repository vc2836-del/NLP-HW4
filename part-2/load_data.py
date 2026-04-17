import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

SQL_SCHEMA_PREFIX = (
    "restriction ( no_discounts , minimum_stay , stopovers , restriction_code , application , "
    "maximum_stay , saturday_stay_required , advance_purchase ) | "
    "flight_stop ( departure_airline , stop_number , arrival_flight_number , flight_id , "
    "arrival_time , departure_flight_number , stop_time , arrival_airline , stop_days , "
    "stop_airport , departure_time ) | "
    "food_service ( meal_code , compartment , meal_number , meal_description ) | "
    "month ( month_number , month_name ) | "
    "code_description ( code , description ) | "
    "city ( city_name , country_name , state_code , time_zone_code , city_code ) | "
    "flight_fare ( flight_id , fare_id ) | "
    "state ( country_name , state_code , state_name ) | "
    "fare_basis ( discounted , class_type , season , basis_days , booking_class , night , "
    "premium , fare_basis_code , economy ) | "
    "date_day ( day_number , month_number , day_name , year ) | "
    "time_interval ( period , end_time , begin_time ) | "
    "flight ( to_airport , aircraft_code_sequence , dual_carrier , flight_id , stops , "
    "flight_days , connections , arrival_time , time_elapsed , flight_number , from_airport , "
    "airline_flight , airline_code , meal_code , departure_time ) | "
    "dual_carrier ( low_flight_number , high_flight_number , main_airline , service_name , "
    "dual_airline ) | "
    "aircraft ( aircraft_code , capacity , wing_span , engines , aircraft_description , "
    "basic_type , weight , pressurized , length , propulsion , pay_load , cruising_speed , "
    "range_miles , wide_body , manufacturer ) | "
    "fare ( to_airport , restriction_code , round_trip_required , fare_id , from_airport , "
    "one_direction_cost , fare_basis_code , round_trip_cost , fare_airline ) | "
    "compartment_class ( compartment , class_type ) | "
    "flight_leg ( flight_id , leg_number , leg_flight ) | "
    "days ( days_code , day_name ) | "
    "airport_service ( minutes_distant , airport_code , direction , city_code , miles_distant ) | "
    "airport ( airport_code , airport_name , airport_location , minimum_connect_time , "
    "country_name , state_code , time_zone_code ) | "
    "time_zone ( time_zone_name , hours_from_gmt , time_zone_code ) | "
    "airline ( note , airline_code , airline_name ) | "
    "equipment_sequence ( aircraft_code , aircraft_code_sequence ) | "
    "ground_service ( airport_code , transport_type , city_code , ground_fare ) | "
    "class_of_service ( booking_class , class_description , rank ) | "

    "translate to SQL: "
)

def preprocess_sql(sql):
    sql = sql.replace("<=", " LESSEQUAL ")
    sql = sql.replace(">=", " GREATEREQUAL ")
    sql = sql.replace("<", " LESSTHAN ")
    sql = sql.replace(">", " GREATERTHAN ")
    sql = " ".join(sql.split())
    return sql

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.encoder_inputs, self.decoder_inputs, self.decoder_targets = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        path_nl = os.path.join(data_folder, f"{split}.nl")
        lines_nl = load_lines(path_nl)
        lines_nl = [SQL_SCHEMA_PREFIX + line.lower() for line in lines_nl]
        
        encoder_inputs = [tokenizer.encode(line, return_tensors = 'pt', max_length = 1280, truncation = True).squeeze(0) for line in lines_nl]
        decoder_inputs = []
        decoder_targets = []

        if split == "test":
            id_bos = tokenizer.pad_token_id
            for _ in lines_nl:
                decoder_inputs.append(torch.tensor([id_bos]))
                decoder_targets.append(torch.tensor([], dtype=torch.long))
        else:
            path_sql = os.path.join(data_folder, f"{split}.sql")
            lines_sql = load_lines(path_sql)

            for line in lines_sql:
                line = preprocess_sql(line)
                target_ids = tokenizer.encode(line, return_tensors = 'pt', max_length=512, truncation=True).squeeze(0)
                bos = torch.tensor([tokenizer.pad_token_id])
                decoder_inputs.append(torch.cat((bos, target_ids[:-1])))
                decoder_targets.append(target_ids)
            
        return encoder_inputs, decoder_inputs, decoder_targets
        
    
    def __len__(self):
        # TODO
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        # TODO
        if self.split == "test":
            return self.encoder_inputs[idx], self.decoder_inputs[idx]
        else:
            return self.encoder_inputs[idx], self.decoder_inputs[idx], self.decoder_targets[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_inputs = [item[0] for item in batch]
    decoder_inputs = [item[1] for item in batch]
    decoder_targets = [item[2] for item in batch]

    encoder_ids = pad_sequence(encoder_inputs, batch_first = True, padding_value = PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_input_ids = pad_sequence(decoder_inputs, batch_first = True, padding_value = PAD_IDX)
    decoder_target_ids = pad_sequence(decoder_targets, batch_first = True, padding_value = PAD_IDX)

    initial_decoder_inputs = decoder_input_ids[:, 0:1]
    return encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_inputs = [item[0] for item in batch]
    decoder_inputs = [item[1] for item in batch]

    encoder_ids = pad_sequence(encoder_inputs, batch_first = True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = pad_sequence(decoder_inputs, batch_first = True, padding_value = PAD_IDX)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x