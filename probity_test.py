from datasets.base import ProbingExample, CharacterPositions, ProbingDataset
from datasets.position_finder import PositionFinder
from probity.probity.datasets.templated import TemplatedDataset, TemplateVariable, Template
from probity.probity.datasets.tokenized import TokenizedProbingDataset
from probity.probity.probes.linear_probe import LogisticProbe, LogisticProbeConfig
from probity.probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
from probity.probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig
from probity.probity.probes.inference import ProbeInference
from transformers import AutoTokenizer


code_examples = [
    "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)",
    "for i in range(10):\n    print(i**2)",
    "import numpy as np\nnp.array([1, 2, 3])",
    "class Person:\n    def __init__(self, name):\n        self.name = name",
    "try:\n    x = 1/0\nexcept ZeroDivisionError:\n    print('Error')",
    "with open('file.txt', 'r') as f:\n    data = f.read()",
    "lambda x: x * 2",
    "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
    "async def fetch_data():\n    return await api.get_data()",
    "@decorator\ndef my_function():\n    pass",
    "x = {'a': 1, 'b': 2}",
    "if x > 0 and y < 10:\n    print('Valid')",
    "while not done:\n    process()",
    "result = [i*i for i in range(10)]",
    "def map_function(f, items):\n    return [f(x) for x in items]"
]

non_code_examples = [
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    "It was the best of times, it was the worst of times.",
    "In a hole in the ground there lived a hobbit.",
    "The only thing we have to fear is fear itself.",
    "Four score and seven years ago our fathers brought forth on this continent, a new nation.",
    "I have a dream that one day this nation will rise up.",
    "Ask not what your country can do for you, ask what you can do for your country.",
    "That's one small step for man, one giant leap for mankind.",
    "Life is like a box of chocolates. You never know what you're gonna get.",
    "May the Force be with you.",
    "Elementary, my dear Watson.",
    "The greatest glory in living lies not in never falling, but in rising every time we fall.",
    "The way to get started is to quit talking and begin doing.",
    "You miss 100% of the shots you don't take."
]

# Create position finder for the end of each example
# PositionFinder can take a variety of inputs, including regexes, character positions, and templates
end_finder = PositionFinder.from_regex(r"[\S].{0,1}$")

# Create ProbingExamples
code_vs_text_examples = []

# Add code examples
for text in code_examples:
    positions_dict = {}
    end_pos = end_finder(text)
    if end_pos:
        positions_dict["END_POSITION"] = end_pos[0]

    code_vs_text_examples.append(ProbingExample(
        text=text,
        label=1,  # 1 for code
        label_text="code",
        character_positions=CharacterPositions(positions_dict) if positions_dict else None,
        attributes={"type": "code"}
    ))

# Add non-code examples
for text in non_code_examples:
    positions_dict = {}
    end_pos = end_finder(text)
    if end_pos:
        positions_dict["END_POSITION"] = end_pos[0]

    code_vs_text_examples.append(ProbingExample(
        text=text,
        label=0,  # 0 for non-code
        label_text="non_code",
        character_positions=CharacterPositions(positions_dict) if positions_dict else None,
        attributes={"type": "non_code"}
    ))

# Create the dataset
code_dataset = ProbingDataset(
    examples=code_vs_text_examples,
    task_type="classification",
    label_mapping={"non_code": 0, "code": 1},
    dataset_attributes={"description": "Code vs. non-code classification dataset"}
)

