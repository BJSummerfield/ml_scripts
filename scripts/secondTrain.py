from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

import evaluate
import numpy as np

books = load_dataset("opus100", "en-ja")
subtitles = load_dataset("open_subtitles", lang1="en", lang2="ja").remove_columns(['id', 'meta'])
subtitlesWithTest = subtitles['train'].train_test_split(test_size=0.1)

combinedTrain = concatenate_datasets([books['train'], subtitlesWithTest['train']])
combinedTest = concatenate_datasets([books['test'], subtitlesWithTest['test']])

checkpoint = "staka/fugumt-en-ja"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

source_lang = "en"
target_lang = "ja"
prefix = "translate English to Japanese: "
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
metric = evaluate.load("sacrebleu")

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs
    
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
    
tokenized_train = combinedTrain.map(preprocess_function, batched=True)
tokenized_test = combinedTest.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)    
    
training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs/combined_training",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./outputs/combined_train")
