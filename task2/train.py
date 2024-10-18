import jieba
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


train_data_path = "./train.txt"
test_data_path = "./test.txt"
dev_data_path = "./dev.txt"

train_data = []
train_label = []
test_data = []
test_label = []
dev_data = []
dev_label = []

### load data
with open(train_data_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().split("\t")
        train_data.append(line[0])
        train_label.append(line[1])
        
with open(test_data_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().split("\t")
        test_data.append(line[0])
        test_label.append(line[1])
        
with open(dev_data_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().split("\t")
        dev_data.append(line[0])
        dev_label.append(line[1])

### tokenize data
def tokenize_sentences(sentences):
    return [" ".join(jieba.cut(sentence)) for sentence in sentences]

tokenized_train_texts = tokenize_sentences(train_data)
tokenized_dev_texts = tokenize_sentences(dev_data)
tokenized_test_texts = tokenize_sentences(test_data)



#### 构建词表
total_words = " ".join(tokenized_train_texts+tokenized_dev_texts+tokenized_test_texts).split()

def build_vocab(sentences):
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence.split())
    return {word: idx + 1 for idx, word in enumerate(vocab)}  # 从1开始

vocab = build_vocab(tokenized_train_texts + tokenized_dev_texts + tokenized_test_texts)
vocab_size = len(vocab) + 1  # 词汇表大小

print("vocab size: ", vocab_size)
# 转换文本为整数序列
def texts_to_sequences(texts, vocab):
    return [[vocab[word] for word in sentence.split() if word in vocab] for sentence in texts]

train_sequences = texts_to_sequences(tokenized_train_texts, vocab)
dev_sequences = texts_to_sequences(tokenized_dev_texts, vocab)
test_sequences = texts_to_sequences(tokenized_test_texts, vocab)

print("train sequence: ", train_sequences[0])


def pad_sequences(sequences, maxlen):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < maxlen:
            seq = seq + [0] * (maxlen - len(seq))  # 用0填充
        else:
            seq = seq[:maxlen]  # 截断
        padded_sequences.append(seq)
    return padded_sequences

max_length = max(max(len(seq) for seq in train_sequences), 
                 max(len(seq) for seq in dev_sequences), 
                 max(len(seq) for seq in test_sequences))

X_train = pad_sequences(train_sequences, maxlen=max_length)
X_dev = pad_sequences(dev_sequences, maxlen=max_length)
X_test = pad_sequences(test_sequences, maxlen=max_length)


label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_label)
y_dev = label_encoder.transform(dev_label)
y_test = label_encoder.transform(test_label)

print("y_train: ", y_train)



#### 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])
    

train_dataset = TextDataset(X_train, y_train)
dev_dataset = TextDataset(X_dev, y_dev)
test_dataset = TextDataset(X_test, y_test)
    
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


print("train_loader: ", train_loader)
print("train_dataset_size: ", len(train_dataset))
print("dataset_example: ", train_dataset[0], train_dataset[1])

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv = nn.Conv1d(embed_size, 100, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(100 * ((max_length - 2) // 2), num_classes) 

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1) 
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

### 一些超参数
embed_size = 128
num_classes = len(label_encoder.classes_)
print("num_classes: ", num_classes)

model = TextCNN(vocab_size, embed_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)




# 训练模型
num_epochs = 20
best_accuracy = 0.0
model.train()

patience = 5  
no_improvement_count = 0  
best_accuracy = 0.0  

for epoch in range(num_epochs):
    model.train()  # 确保进入训练模式
    for step, (texts, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 每10步进行验证
        if (step + 1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for val_texts, val_labels in dev_loader:
                    val_outputs = model(val_texts)
                    _, predicted = torch.max(val_outputs, 1)
                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.2f}%")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                no_improvement_count = 0 
                torch.save(model.state_dict(), 'best_model.pth')
                print("Best model saved!")
            else:
                no_improvement_count += 1 
            if no_improvement_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}, step {step+1}")
                break 

    if no_improvement_count >= patience:
        break  # 提前结束训练


# 最后的评估
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")