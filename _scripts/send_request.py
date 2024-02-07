import requests

api_url = f"http://localhost:8000/generate"
best_of = 1
use_beam_search = False
output_len = 100
headers = {"User-Agent": "Benchmark Client"}


# prompt =  "Who are you? What can you do?"
prompt = """There is a single choice question about high school european history. Answer the question by replying A, B, C or D.
Q: This question refers to the following information.
In order to make the title of this discourse generally intelligible, I have translated the term "Protoplasm," which is the scientific name of the substance of which I am about to speak, by the words "the physical basis of life." I suppose that, to many, the idea that there is such a thing as a physical basis, or matter, of life may be novel—so widely spread is the conception of life as something which works through matter. … Thus the matter of life, so far as we know it (and we have no right to speculate on any other), breaks up, in consequence of that continual death which is the condition of its manifesting vitality, into carbonic acid, water, and nitrogenous compounds, which certainly possess no properties but those of ordinary matter.
Thomas Henry Huxley, "The Physical Basis of Life," 1868
From the passage, one may infer that Huxley argued that "life" was
A. a force that works through matter
B. essentially a philosophical notion
C. merely a property of a certain kind of matter
D. a supernatural phenomenon
Answer:"""

print(prompt)

pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": False,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": False,
            "stream": False,
        }

response = requests.post(api_url, json=pload)
print(type(response.text))
print(response.text)
# print(response.status_code)

