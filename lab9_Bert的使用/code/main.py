from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForNextSentencePrediction
import torch

# 1. 装载英文BERT tokenizer 和 bert-base-cased模型
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")

# 2. 用BERT为句子”Apple company does not sell the apple.”编码
sentence = "Apple company does not sell the apple."
input_ids = tokenizer.encode(sentence, add_special_tokens=True) # 将句子中的单词编码为预训练词汇表中的ID（这是因为神经网络需要数值输入），true表示添加特殊标记

# 3. 输出句子转化后的ID
print('')
print('3.')
print(f"ID of \"{sentence}\" is: ", input_ids)
print('')

# 4. 分别输出句子编码后单词‘CLS’,‘Apple’,‘apple’和‘SEP’,四个词对应的编码
# 特殊标记：起始标记（[CLS]），结束标记（[SEP]）和填充标记（[PAD]）
cls_token_id = tokenizer.convert_tokens_to_ids('[CLS]')
apple_token_id = tokenizer.convert_tokens_to_ids('Apple')
small_apple_token_id = tokenizer.convert_tokens_to_ids('apple')
sep_token_id = tokenizer.convert_tokens_to_ids('[SEP]')

car_token_id = tokenizer.convert_tokens_to_ids('car')
bicycle_token_id = tokenizer.convert_tokens_to_ids('bicycle')

print('')
print('4.')
print("CLS Token ID:", cls_token_id)
print("Apple Token ID:", apple_token_id)
print("apple Token ID:", small_apple_token_id)
print("SEP Token ID:", sep_token_id)
print('')

# 5. 计算‘Apple’和‘apple’，‘CLS’和‘Apple’,‘CLS’和‘SEP’之间的距离
embedding = model.get_input_embeddings()
apple_embedding = embedding(torch.tensor([apple_token_id])) # 将单词id输入到预训练模型中获得单词的词向量
small_apple_embedding = embedding(torch.tensor([small_apple_token_id]))
cls_embedding = embedding(torch.tensor([cls_token_id]))
sep_embedding = embedding(torch.tensor([sep_token_id]))
car_embedding = embedding(torch.tensor([car_token_id]))
bicycle_embedding = embedding(torch.tensor([bicycle_token_id]))

cosine_similarity = torch.nn.functional.cosine_similarity # 余弦距离

similarity_apple_vs_small_apple = cosine_similarity(apple_embedding, small_apple_embedding) # 计算余弦距离
similarity_cls_vs_apple = cosine_similarity(cls_embedding, apple_embedding)
similarity_cls_vs_sep = cosine_similarity(cls_embedding, sep_embedding)
similarity = cosine_similarity(car_embedding, bicycle_embedding)

print('')
print('5.')
print("Similarity (Apple vs apple):", similarity_apple_vs_small_apple.item())
print("Similarity (CLS vs Apple):", similarity_cls_vs_apple.item())
print("Similarity (CLS vs SEP):", similarity_cls_vs_sep.item())
# print(similarity)
print('')

# 6. 输入句子“I have a [MASK] named Charlie.”，重新加载BertForMaskedLM模型，通过BERT预测[MASK]位置最可能的单词
masked_sentence = "I have a [MASK] named Charlie."
masked_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
model_masked_lm = BertForMaskedLM.from_pretrained("bert-base-cased")
input_ids_masked = tokenizer.encode(masked_sentence, add_special_tokens=True)

# 找到[MASK]在句子中的位置
masked_index = input_ids_masked.index(tokenizer.mask_token_id)

# 将输入转化为PyTorch张量
input_ids_tensor = torch.tensor([input_ids_masked])

# 计算模型的预测
with torch.no_grad():
    outputs = model_masked_lm(input_ids_tensor)
    predictions = outputs.logits[0, masked_index].topk(1)  # 获取最可能的前5个预测单词

# 解码预测结果
predicted_token_ids = predictions.indices.tolist()
predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)

print('')
print('6.')
for token in predicted_tokens:
    print("Predicted word of \"I have a [MASK] named Charlie.\" is:", token)
print('')

# 7. 输入句子“I have a small  ”，重新加载BertForNextSentencePrediction模型，通过BERT预测下一句。
sentence = "I have a cat, I like it very much"
model_masked_lm = BertForMaskedLM.from_pretrained("bert-base-cased")
# 将输入句子编码成模型可以理解的格式

while True:
    # 句子结尾添加 [MASK]
    inputs = tokenizer.encode(sentence, add_special_tokens=False)
    inputs.append(tokenizer.mask_token_id)
    
    # 找到[MASK]在句子中的位置
    masked_index = inputs.index(tokenizer.mask_token_id)
    # print(masked_index)

    # 将输入转化为PyTorch张量
    inputs_ids_tensor = torch.tensor([inputs])
    
    # 获取预测结果
    with torch.no_grad():
        outputs = model_masked_lm(inputs_ids_tensor)
        predictions = outputs.logits[0, masked_index].topk(1)  # 获取最可能的前5个预测单词
    
    # 获取预测结果对应的文本
    predicted_ids = predictions.indices.tolist()
    predicted_text = tokenizer.convert_ids_to_tokens(predicted_ids)

    inputs[masked_index-1]=predicted_ids[0]
    
    # 如果预测结果是标点符号或者句子结束，则停止预测，否则继续在末尾添加 [MASK] 进行预测
    if  predicted_text in [".", "!", "?"] or masked_index > 40:
        break
    
    # 否则，将预测结果添加到句子末尾
    sentence += " " + predicted_text[0]

# 打印最终的补全句子
print(sentence)

print('')
print('7.')
print("Probability of next sentence:", sentence)
print('')
