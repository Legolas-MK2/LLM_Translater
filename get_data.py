import ollama
import pandas as pd
pd.set_option('display.max_colwidth', None)

# Parquet-Datei lesen
df = pd.read_parquet('train-00000-of-00001.parquet')

# Ersten Datensatz (erste Zeile) auslesen
first_row = df.iloc[0]
for i in range(100):
  print(df.iloc[i]['text'])
  print(df.iloc[i]['translation'])
  print()

exit()

response = ollama.chat(model='qwen2.5:72b', messages=[
  {
    'role': 'system',
    'content': 'You are a strict translation model. Your only task is to translate texts from English to German. Do not explain or elaborate, and do not add any additional information. Just provide a direct and accurate translation for each sentence.'
  },
  {
    'role': 'user',
    'content': 'The beauty of nature lies in its simplicity and complexity at the same time. While the changing colors of the autumn leaves mesmerize us, the intricate patterns of a snowflake remind us of nature’s attention to detail. In a world that moves so fast, taking a moment to appreciate these little wonders can bring peace and perspective.',
  }
])
print(response['message']['content'])

"""
1. qwen 14b
2. qwen 32b
3. llama3.2:latest (3b)
4. nemotron-mini:4b-instruct-q8_0
"""
