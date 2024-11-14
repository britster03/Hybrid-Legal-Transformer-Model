# legal_llm.py

# import necessary libraries
import spacy
from rdflib import Graph, Namespace, RDF, URIRef
import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import higher

# define constants
LEGAL_CORPUS_PATH = 'legal_corpus.txt'
KG_SERIALIZE_PATH = 'legal_kg.ttl'
EMBEDDING_DIM = 64  # dimension for knowledge graph embeddings
BATCH_SIZE = 2
NUM_EPOCHS_NODE2VEC = 100
NUM_EPOCHS_TRAINING = 3
LEARNING_RATE = 5e-5

# initialize spacy model
nlp = spacy.load('en_core_web_sm')

# initialize rdf graph
g = Graph()

# define a custom namespace
EX = Namespace("http://legalkg.local/#")
g.bind("ex", EX)

# function to extract entities and build the knowledge graph
def build_knowledge_graph(corpus):
    for text in corpus:
        doc = nlp(text)
        for ent in doc.ents:
            # create a uri for each entity
            ent_uri = URIRef(EX[ent.text.replace(" ", "_")])
            # add entity type to the graph
            g.add((ent_uri, RDF.type, URIRef(EX[ent.label_])))
    return g

# read the legal corpus
with open(LEGAL_CORPUS_PATH, 'r', encoding='utf-8') as f:
    corpus = f.readlines()

# build the knowledge graph
knowledge_graph = build_knowledge_graph(corpus)

# serialize the knowledge graph to a turtle file
knowledge_graph.serialize(destination=KG_SERIALIZE_PATH, format='turtle')
print(f"knowledge graph saved to {KG_SERIALIZE_PATH}")

# map entities to unique IDs
entity_list = list(set([str(s) for s, p, o in knowledge_graph]))
entity2id = {entity: idx for idx, entity in enumerate(entity_list)}

# build edge index for PyTorch Geometric
edge_index = []
for subj, pred, obj in knowledge_graph:
    subj_id = entity2id[str(subj)]
    obj_id = entity2id[str(obj)]
    edge_index.append([subj_id, obj_id])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# create PyTorch Geometric data object
data = Data(edge_index=edge_index, num_nodes=len(entity2id))

# define and train Node2Vec model
node2vec = Node2Vec(edge_index, embedding_dim=EMBEDDING_DIM, walk_length=20,
                    context_size=10, walks_per_node=10, num_negative_samples=1,
                    sparse=True)

loader = node2vec.loader(batch_size=128, shuffle=True)
optimizer_node2vec = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

def train_node2vec():
    node2vec.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer_node2vec.zero_grad()
        loss = node2vec.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer_node2vec.step()
        total_loss += loss.item()
    return total_loss / len(loader)

print("training node2vec...")
for epoch in range(1, NUM_EPOCHS_NODE2VEC + 1):
    loss = train_node2vec()
    if epoch % 10 == 0:
        print(f'epoch: {epoch:03d}, loss: {loss:.4f}')

# get entity embeddings
entity_embeddings = node2vec.embedding.weight.data

# initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # set pad token

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# freeze original embeddings
for param in model.parameters():
    param.requires_grad = False

# define hybrid embedding layer
class HybridEmbedding(nn.Module):
    def __init__(self, original_embedding, lkg_embeddings):
        super(HybridEmbedding, self).__init__()
        self.original_embedding = original_embedding
        self.lkg_embeddings = lkg_embeddings  # precomputed embeddings
        self.linear = nn.Linear(
            original_embedding.embedding_dim + lkg_embeddings.size(1),
            original_embedding.embedding_dim
        )
    
    def forward(self, input_ids, entity_ids):
        token_embeds = self.original_embedding(input_ids)
        lkg_embeds = self.lkg_embeddings[entity_ids]
        combined = torch.cat((token_embeds, lkg_embeds), dim=-1)
        combined = self.linear(combined)
        return combined

# replace the model's embedding layer
model.transformer.wte = HybridEmbedding(model.transformer.wte, entity_embeddings)

# define symbolic reasoning module
class SymbolicReasoningModule(nn.Module):
    def __init__(self, embedding_dim):
        super(SymbolicReasoningModule, self).__init__()
        self.rule_layer = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.ReLU()
    
    def forward(self, input_embeddings):
        reasoning_output = self.activation(self.rule_layer(input_embeddings))
        return reasoning_output

# instantiate symbolic reasoning module
srm = SymbolicReasoningModule(EMBEDDING_DIM).to(device)

# define adaptive attention mechanism
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

class AdaptiveAttention(GPT2Attention):
    def __init__(self, config, lkg_embedding_dim):
        super(AdaptiveAttention, self).__init__(config)
        self.lkg_linear = nn.Linear(lkg_embedding_dim, config.hidden_size)
    
    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
                use_cache=False, output_attentions=False, lkg_embeddings=None):
        # standard attention
        output_attn = super().forward(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = output_attn[0]
        
        if lkg_embeddings is not None:
            # project lkg embeddings
            lkg_proj = self.lkg_linear(lkg_embeddings)
            # adjust attention output
            attn_output = attn_output + lkg_proj
        
        outputs = (attn_output,) + output_attn[1:]
        return outputs

# replace attention layers with adaptive attention
for block in model.transformer.h:
    block.attn = AdaptiveAttention(model.config, lkg_embedding_dim=EMBEDDING_DIM).to(device)

# define hybrid model combining transformer and symbolic reasoning
class HybridModel(nn.Module):
    def __init__(self, transformer_model, srm):
        super(HybridModel, self).__init__()
        self.transformer = transformer_model
        self.srm = srm
    
    def forward(self, input_ids, entity_ids, attention_mask=None):
        embeddings = self.transformer.transformer.wte(input_ids, entity_ids)
        transformer_outputs = self.transformer(inputs_embeds=embeddings, attention_mask=attention_mask)
        srm_outputs = self.srm(embeddings)
        combined_outputs = transformer_outputs.last_hidden_state + srm_outputs
        return combined_outputs

# instantiate hybrid model
hybrid_model = HybridModel(model, srm).to(device)

# define custom dataset
class LegalDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128
        )
        input_ids = encoding['input_ids'].squeeze()  # shape: [128]
        attention_mask = encoding['attention_mask'].squeeze()  # shape: [128]
        # map tokens to entities (simplified)
        # here, we randomly assign entity IDs for demonstration
        entity_ids = torch.randint(0, len(entity2id), (128,))
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'entity_ids': entity_ids}

# split corpus into training and validation
split_idx = int(0.8 * len(corpus))
train_texts = corpus[:split_idx]
val_texts = corpus[split_idx:]

# create datasets
train_dataset = LegalDataset(train_texts, tokenizer)
val_dataset = LegalDataset(val_texts, tokenizer)

# define expert feedback function (placeholder)
def expert_feedback_fn(text):
    # in real scenario, replace with actual evaluation
    return random.uniform(0, 1)

# define reward computation
def compute_reward(generated_texts, expert_feedback_fn):
    rewards = [expert_feedback_fn(text) for text in generated_texts]
    return torch.tensor(rewards, dtype=torch.float32).to(device)

# define training loop with expert feedback
def train_model_with_feedback(model, tokenizer, dataset, optimizer, num_epochs):
    model.train()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity_ids = batch['entity_ids'].to(device)
            
            # forward pass
            outputs = model(input_ids, entity_ids, attention_mask=attention_mask)
            # compute loss (simplified)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            
            # generate text for feedback
            generated_ids = model.transformer.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # compute rewards
            rewards = compute_reward(generated_texts, expert_feedback_fn)
            
            # adjust loss with rewards
            adjusted_loss = loss - rewards.mean()
            
            # backpropagation
            optimizer.zero_grad()
            adjusted_loss.backward()
            optimizer.step()
            
            total_loss += adjusted_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch+1}/{num_epochs}, loss: {avg_loss:.4f}")

# define meta-learning training (simplified)
def meta_train(model, tasks, num_inner_steps=1, inner_lr=1e-4, meta_lr=1e-5):
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    for task_texts in tasks:
        task_dataset = LegalDataset(task_texts, tokenizer)
        task_loader = DataLoader(task_dataset, batch_size=2, shuffle=True)
        with higher.innerloop_ctx(model, meta_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
            for i, batch in enumerate(task_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                entity_ids = batch['entity_ids'].to(device)
                
                outputs = fmodel(input_ids, entity_ids, attention_mask=attention_mask)
                loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
                diffopt.step(loss)
                if i >= num_inner_steps:
                    break
            # compute meta-loss (simplified)
            meta_loss = loss
            meta_loss.backward()
            meta_optimizer.step()

# main execution
if __name__ == "__main__":
    # train model with expert feedback
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=LEARNING_RATE)
    print("training model with expert feedback...")
    train_model_with_feedback(hybrid_model, tokenizer, train_dataset, optimizer, NUM_EPOCHS_TRAINING)
    
    # evaluate the model
    def evaluate_model(model, tokenizer, dataset):
        model.eval()
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
        total_loss = 0
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                entity_ids = batch['entity_ids'].to(device)
                
                outputs = model(input_ids, entity_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"validation loss: {avg_loss:.4f}")
    
    print("evaluating model...")
    evaluate_model(hybrid_model, tokenizer, val_dataset)
    
    # save the model
    model_save_path = 'hybrid_legal_model'
    hybrid_model.transformer.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"model saved to {model_save_path}")
