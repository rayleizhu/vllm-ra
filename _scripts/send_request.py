import requests

api_url = f"http://SH-IDC1-10-142-5-114:8000/generate"
best_of = 1
use_beam_search = False
output_len = 100
headers = {"User-Agent": "Benchmark Client"}


# prompt = "How to answer a single choice question? Can you give an example?"
prompt = "Who are you?"
# prompt = """There is a single choice question about high school european history. Answer the question by replying A, B, C or D.
# Q: This question refers to the following information.
# In order to make the title of this discourse generally intelligible, I have translated the term "Protoplasm," which is the scientific name of the substance of which I am about to speak, by the words "the physical basis of life." I suppose that, to many, the idea that there is such a thing as a physical basis, or matter, of life may be novel—so widely spread is the conception of life as something which works through matter. … Thus the matter of life, so far as we know it (and we have no right to speculate on any other), breaks up, in consequence of that continual death which is the condition of its manifesting vitality, into carbonic acid, water, and nitrogenous compounds, which certainly possess no properties but those of ordinary matter.
# Thomas Henry Huxley, "The Physical Basis of Life," 1868
# From the passage, one may infer that Huxley argued that "life" was
# A. a force that works through matter
# B. essentially a philosophical notion
# C. merely a property of a certain kind of matter
# D. a supernatural phenomenon
# # Answer:"""
# prompt="""There is a single choice question about high school european history. Answer the question by replying A, B, C or D.
# Question: This question refers to the following information.
# The following excerpt is from a pamphlet.
# You will do me the justice to remember, that I have always strenuously supported the Right of every man to his own opinion, however different that opinion might be to mine. He who denies to another this right, makes a slave of himself to his present opinion, because he precludes himself the right of changing it.
# The most formidable weapon against errors of every kind is Reason.nA. The ideas of personal liberty and nationalism conceived during the Enlightenment resulted in radical revolutions that could spread throughout Europe.
# B. The conquest of Europe by Napoleon led to the creation of new factions and shifted the European balance of power.
# C. The power of monarchs had grown to the point where it needed to be checked by other powers within each nation or domination of civilians would occur.
# D. The rising and falling economic cycle of the newly emerging capitalist economy could lead to civilian unrest that must be suppressed.
# Answer:"""


# print(prompt)

pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": output_len,
            "ignore_eos": False,
            "stream": False,
            # "do_sample": False
        }

response = requests.post(api_url, json=pload)
# print(type(response.text))
print(response.json())
# print(response.status_code)

