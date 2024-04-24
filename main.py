import pandas as pd
import random
from transformers import AutoTokenizer, GPT2LMHeadModel, BertForMaskedLM
import torch
from sentence_transformers import SentenceTransformer, util
import heapq

# Function to get the value of a specific cell
def get_value(df, column_name, row_index):
    return df.at[row_index, column_name]

# Function to get a row as a dictionary
def get_row(df, row_index):
    return df.iloc[row_index].to_dict()

# Function to get 1000 random rows as a list of dictionaries
def get_rows_from_sheet(df):
    row_indices = random.sample(range(len(df)), 1000)
    return [df.iloc[idx].to_dict() for idx in row_indices]


def get_collumnAs_array(arrayof_dicts, column_name):

    values_array = []

    for row in range(len(arrayof_dicts)):
        values_array.append(arrayof_dicts[row][column_name])

    return values_array

def calculate_similarity(model, text1, embeddings2):

    # Encode the texts to obtain embeddings
    embeddings1 = model.encode([text1], convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.pytorch_cos_sim(embeddings1, embeddings2)[0]

    return similarities

def calculate_similarities(model, listof_texts, key_value, key_value2, model_name):

    global bert_top5
    global bert_top1
    global gpt_top1
    global gpt_top5

    similarities = [[0] * 1000 for _ in range(1000)]

    values_array = get_collumnAs_array(listof_texts, key_value2)

    model.tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 

    embeddings2 = model.encode(values_array, convert_to_tensor=True)

    for row in range(len(listof_texts)):
        print("index " + str(row))
        values = calculate_similarity(model, listof_texts[row][key_value], embeddings2) 
        similarities[row] = values

    print("debug")

    highest_five_indices = []

    for j in range(len(similarities)):
        highest_five_indices.append(
            heapq.nlargest(5, range(len(similarities[j])), key=similarities[j].__getitem__)
        )
        #print(j)
    print("debug2")
    for i in range(len(highest_five_indices)):
        print("index " + str(i) + " top 5 indexes")
        print(highest_five_indices[i])
        if model_name == "bert":
            if i == highest_five_indices[i][0]:
                bert_top1 += 1
                print("bert_top1 1 artirildi " + str(bert_top1))
            elif i in highest_five_indices[i]:
                bert_top5 += 1
                print("bert_top5 1 artirildi " + str(bert_top5))
        elif model_name == "gpt":
            if i == highest_five_indices[i][0]:
                gpt_top1 += 1
                print("gpt_top1 1 artirildi " + str(gpt_top1))
            elif i in highest_five_indices[i]:
                gpt_top5 += 1
                print("gpt_top5 1 artirildi " + str(gpt_top5))

    return similarities


#shift command / block command
file_name = 'soru_cevap.xlsx'
sheet_name = 'Sheet1'

# Read the Excel file into a DataFrame
df = pd.read_excel(file_name, sheet_name=sheet_name)

# BERT ve GPT modellerini yükleyelim
bert_model = SentenceTransformer('ytu-ce-cosmos/turkish-mini-bert-uncased')
gpt_model = SentenceTransformer('ytu-ce-cosmos/turkish-gpt2')

bert_top1 = 0
bert_top5 = 0

gpt_top1 = 0
gpt_top5 = 0


all_rows = get_rows_from_sheet(df)
question1000answer = calculate_similarities(gpt_model, all_rows, "soru", "insan cevabı","gpt")


print("gpt_top1 " + str(gpt_top1))
print("gpt_top5 " + str(gpt_top5))


