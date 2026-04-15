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
    "flight ( flight_id , airline_code , from_airport , to_airport , departure_time , "
    "arrival_time , stops , flight_number , meal_code , aircraft_code_sequence , "
    "flight_days , time_elapsed , connections , dual_carrier , airline_flight ) | "

    "airport ( airport_code , airport_name , airport_location , state_code , "
    "country_name , time_zone_code , minimum_connect_time ) | "

    "airline ( airline_code , airline_name , note ) | "

    "city ( city_code , city_name , state_code , country_name , time_zone_code ) | "

    "airport_service ( airport_code , city_code , direction , miles_distant , minutes_distant ) | "

    "fare ( fare_id , fare_airline , from_airport , to_airport , fare_basis_code , "
    "round_trip_required , round_trip_cost , one_direction_cost , restriction_code ) | "

    "flight_fare ( flight_id , fare_id ) | "

    "fare_basis ( fare_basis_code , booking_class , class_type , premium , economy , "
    "discounted , night , season , basis_days ) | "

    "class_of_service ( booking_class , class_description , rank ) | "

    "food_service ( meal_code , meal_description , compartment , meal_number ) | "

    "ground_service ( airport_code , city_code , transport_type , ground_fare ) | "

    "restriction ( restriction_code , advance_purchase , stopovers , saturday_stay_required , "
    "no_discounts , minimum_stay , maximum_stay , application ) | "

    "dual_carrier ( main_airline , dual_airline , service_name , low_flight_number , high_flight_number ) | "

    "code_description ( code , description ) | "

    "aircraft ( aircraft_code , aircraft_description , basic_type , manufacturer , propulsion , "
    "wide_body , pressurized , capacity , wing_span , engines , weight , length , "
    "pay_load , cruising_speed , range_miles ) | "

    "equipment_sequence ( aircraft_code_sequence , aircraft_code ) | "

    "flight_stop ( flight_id , stop_number , stop_airport , stop_days , stop_time , "
    "arrival_time , departure_time , arrival_airline , arrival_flight_number , "
    "departure_airline , departure_flight_number ) | "

    "flight_leg ( flight_id , leg_number , leg_flight ) | "

    "state ( state_code , state_name , country_name ) | "

    "time_zone ( time_zone_code , time_zone_name , hours_from_gmt ) | "

    "date_day ( month_number , day_number , year , day_name ) | "

    "days ( days_code , day_name ) | "

    "month ( month_number , month_name ) | "

    "time_interval ( period , begin_time , end_time ) | "

    "compartment_class ( compartment , class_type ) | "

    "translate to SQL: "
)

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
                decoder_targets.append(torch.tensor([]))
        else:
            path_sql = os.path.join(data_folder, f"{split}.sql")
            lines_sql = load_lines(path_sql)

            for line in lines_sql:
                line = line.replace(" < ", " LESSTHAN ")
                line = line.replace(" > ", " GREATERTHAN ")
                target_ids = tokenizer.encode(line, return_tensors = 'pt').squeeze(0)
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
    return train_x, train_y, dev_x, dev_y, test_x