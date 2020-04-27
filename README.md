# therapy-reflections-generator
Output short, contextual and empathetic reflections by processing journal entries

### Introduction
* Use state-of-the-art generative models to learn pattern of therapy conversations.
* Implemented in pytorch.

### Model Details
* Model used is based on Open AI GPT-2 model.
* The input to the model during training consists of a journaling entry and its corresponding response concatenated as a single sequence.
* A custom separator token is used to separate these two sub-sequences, as the GPT-2 model is adept at inferring customized structure.
* During evaluation, the fine-tuned model generates a response using the journaling entry as the seed sentence.
* The seed sentence and the response are concatenated with the custom separator to compute the perplexity for the generated response.   

### File Structure
* **data** : Contains the data files used to train and evaluate the model.
* **scripts** : Contains scripts to preprocess data, and to train and evaluate the model.

### Steps to reproduce

1. Clone the `hugginface\transformers` repository from this [link](https://github.com/huggingface/transformers).
2. Reset to commit `827d6d6ef0` using the command `git reset --hard 827d6d6ef0`  
3. Install from source.
4. Clone this repository.
5. Preprocess [counsel_chat.csv](data/counsel_chat.csv) using [prepare_data.py](scripts/prepare_data.py) to generate train, val and test splits using: `python prepare_data.py`.
6. Run the training script [train.py](scripts/train.py) using the command:
   ```python train.py     --output_dir=output     --model_type=gpt2     --model_name_or_path=gpt2     --data_dir=../data     --do_train     --train_data_file=train.jsonl   --do_eval    --eval_data_file=val.jsonl --per_gpu_train_batch_size 8 --num_train_epochs 5 --overwrite_output_dir --num_return_sequences 5 --learning_rate 2e-5 -k 7 -p 0.7```
7. Run the test script:
    `python test.py     --output_dir=output_1     --model_type=gpt2     --model_name_or_path=output     --data_dir=../data    --eval_data_file=test.jsonl --overwrite_output_dir --num_return_sequences 5 -k 7 -p 0.7`

8. The evaluation metric used is perplexity.

#### Generator Parameters
Apart from the parameters mentioned above, the parameters used to control the generation of text are:


| Parameters        | Description           |
| :--------------:|:-------------:|
| length      | Number of tokens to generate |
| k      | Filter top-k tokens before sampling      |
| p | Nucleus filtering top-p tokens before sampling      |
| num_return_sequences  | Number of output sequences to generate |
| sep_token | Custom separator token to separate journal and response |

### Sample Generated Output

<code>**Journal**: I'm not suicidal and wouldn't take my own life, but sometimes, I've wished for an accident to occur and take it. I feel like I just shouldn't be here and wish I wouldn't have been born so I didn't have to go through life. For me, it's a chore, but I don't know if that's normal or a sign of something</code>

<code>**Reflection**: It sounds like you are in a very difficult place.  It is not uncommon for people to have</code>

---
 
<code>**Journal**: I'm socially awkward. I've always want to be popular. I've had the opportunity to be popular, but every time I get i,t I'm too scared. All I have to do is talk the popular people. They're really nice to me, but I'm too scared of change. I really want to work up the courage to lose my fear</code>

<code>**Reflection**: Having the courage to speak your fears as they are sounds like a large step in the right direction. </code>

---

<code>**Journal**: My girlfriend just quit drinking and she became really depressed. She told me that she wants to move. What can I do to help her? I want her to stay</code>

<code>**Reflection**: Hi there.  I am sorry to hear about your girlfriend's current situation</code>

### Challenges Faced
1. The strategy to truncate examples with long text for the purpose of training. In this POC, the first two sentences of the response (reflection) are extracted for this purpose.
2. The evaluation metrics to evaluate responses.
3. How to generate truly contextual responses. Some examples included random names of people.
4. How to avoid generating incorrect and highly insensitive responses.
5. Avoid generating responses that are continuations of the question.
6. Stop generating once a coherent sentence is generated. 

Example Response from an experiment with no filtering before sampling (k=0 and p=0) describing points 3 and 4 :

<code>**Journal**: My girlfriend just quit drinking and she became really depressed. She told me that she wants to move. What can I do to help her? I want her to stay</code>

<code>**Reflection**: Hi Ashley,It is with great sadness that we are unable to find a permanent solution to your problematic</code>

### Future Directions
1. Create better contextualized and personalized responses, a knowledge base containing the personality traits and the afflictions affecting a person can be created.
2. Maintain a history of prior utterances.
3. Sample more data for training from sources like [7cups.com](https://www.7cups.com/qa/)
4. Constrain generated responses using techniques like Textual Entailment to prevent generating incorrect and irrelevant responses.
5. Use IR techniques to narrow down candidate responses.
6. Explore other evaluation metrics like F-1 measure.
7. Scale to larger GPT-2 models, and consequently employ distributed training frameworks like horovod.