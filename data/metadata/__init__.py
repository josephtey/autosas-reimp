import json
import glob
import os

PROBLEM_STMTS = {}
_path = os.path.join(os.path.dirname(__file__), 'problems', '*.txt')
for fname in glob.glob(_path):
    basename = os.path.basename(fname).split('.txt')[0]
    with open(fname) as f:
        PROBLEM_STMTS[basename] = f.read()

PROBLEM_PROMPTS = {
    'asap_01': (
        'After reading the group\'s procedure, describe what additional '
        'information you would need in order to replicate the experiment. '
        'Make sure to include at least three pieces of information.'
    ),
    'asap_02': (
        'a.	Draw a conclusion based on the student\'s data.\n'
        'b.	Describe two ways the student could have improved the '
        'experimental design and/or validity of the results.'
    ),
    'asap_03': (
        'Explain how pandas in China are similar to koalas in Australia '
        'and how they both are different from pythons. Support your '
        'response with information from the article.'
    ),
    'asap_04': (
        'Explain the significance of the word “invasive” to the rest '
        'of the article. Support your response with information '
        'from the article.'
    ),
    'asap_05': (
        'Starting with mRNA leaving the nucleus, list and describe '
        'four major steps involved in protein synthesis.'
    ),
    'asap_06': (
        'List and describe three processes used by cells to control the '
        'movement of substances across the cell membrane.'
    ),
    'asap_07': (
        'Identify ONE trait that can describe Rose based on her '
        'conversations with her sister Anna or Aunt Kolab. Include ONE '
        'detail from the story that supports your answer.'
    ),
    'asap_08': (
        'During the story, the reader gets background information '
        'about Mr. Leonard. Explain the effect that background information '
        'has on Paul. Support your response with details from the story.'
    ),
    'asap_09': (
        'How does the author organize the article? '
        'Support your response with details from the article.'
    ),
    'asap_10': (
        'Brandi and Jerry were designing a doghouse. Use the results '
        'from the experiment to describe the best paint color '
        'for the doghouse.\n'
        '\n'
        'In your description, be sure to:\n'
        '- Choose a paint color.\n'
        '- Describe how that color might affect the inside of the doghouse.\n'
        '- Use results from the experiment to support your description.'
    )
}
