import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Read the Excel file into a DataFrame
file_name = 'soru_cevap.xlsx'
sheet_name = 'Sheet1'
df = pd.read_excel(file_name, sheet_name=sheet_name)

# Rastgele 1000 satır seçme işlemi
sampled_df = df.sample(n=1000)

# Load the pre-trained Turkish BERT model
model = SentenceTransformer("ytu-ce-cosmos/turkish-medium-bert-uncased")
model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token

# Extract question, human answer, and machine answer from each row
questions = sampled_df['soru'].tolist()
human_answers = sampled_df['insan cevabı'].tolist()
machine_answers = sampled_df['makine cevabı'].tolist()

# Encode questions, human answers, and machine answers to obtain embeddings
print("1")
question_embeddings = model.encode(questions, convert_to_tensor=False)
print("2")
human_answer_embeddings = model.encode(human_answers, convert_to_tensor=False)
print("3")
machine_answer_embeddings = model.encode(machine_answers, convert_to_tensor=False)
print("4")

# Apply t-SNE to reduce dimensions to 2D
tsne = TSNE(n_components=2, random_state=42)
question_tsne = tsne.fit_transform(question_embeddings)
human_answer_tsne = tsne.fit_transform(human_answer_embeddings)
machine_answer_tsne = tsne.fit_transform(machine_answer_embeddings)

# Visualize the embeddings in 2D using scatter plot
plt.figure(figsize=(10, 6))

# Plot questions
plt.scatter(question_tsne[:, 0], question_tsne[:, 1], label='Sorular', alpha=0.7, color='blue')

# Plot human answers
plt.scatter(human_answer_tsne[:, 0], human_answer_tsne[:, 1], label='İnsan Cevapları', alpha=0.7, color='green')

# Plot machine answers
plt.scatter(machine_answer_tsne[:, 0], machine_answer_tsne[:, 1], label='Makine Cevapları', alpha=0.7, color='red')

plt.title('Rastgele 1000 Örnekleme ile Sorular ve Cevapların t-SNE Görselleştirmesi')
plt.legend()
plt.show()
