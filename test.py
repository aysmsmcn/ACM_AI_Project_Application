from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("Don't ever say its over if I'm breathing")

print(res)