import pickle
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch





def respond_to_dialog(model, tokenizer_boltalka, texts):
    prefix = '\nx:'
    for i, t in enumerate(texts):
        prefix += t
        prefix += '\nx:' if i % 2 == 1 else '\ny:'
    tokens = tokenizer_boltalka(prefix, return_tensors='pt')
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    end_token_id = tokenizer_boltalka.encode('\n')[0]
    size = tokens['input_ids'].shape[1]
    output = model.generate(
        **tokens, 
        eos_token_id=end_token_id,
        do_sample=True, 
        max_length=size+128, 
        repetition_penalty=3.2, 
        temperature=1,
        num_beams=10,
        length_penalty=0.01,
        pad_token_id=tokenizer_boltalka.eos_token_id
    )
    decoded = tokenizer_boltalka.decode(output[0])
    result = decoded[len(prefix):]
    return result.strip()

def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 200
    
    
    with open('models/saved_GAN_char2idx.pkl', 'rb') as f:
        char2idx = pickle.load(f)
    with open('models/saved_GAN_idx2char.pkl', 'rb') as f:
        idx2char = pickle.load(f)
    
    
    
    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = 0.5

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))