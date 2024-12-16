# -*- coding: utf-8 -*-

import os
from bs4 import BeautifulSoup

from tools.splitCorpus import train_test_split
from tools.tokenizer import tokenizer2


"""
This program is implemented to extract the content and tokenize documents collected from the the project of Gutenberg. 
The program takes as input a .html file and returns as output a tokenized .txt file
"""


# Keywords and classes to exclude
exclude_keywords = ["Title", "Author", "Release date", "Language", "Credits", "Preface", "Translator", "Illustrator"]
exclude_classes = ["c", "toc", "rt", "footnote"]


def extract_paragraphs(input_file, output_file):
    # Read the HTML file
    with open(input_file, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Parse the HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract text from <p> tags, filtering out those with the specified keywords and classes
    filtered_paragraphs = [
        " ".join(paragraph.get_text().strip().split()) for paragraph in soup.find_all("p")
        if not any(keyword in paragraph.get_text() for keyword in exclude_keywords)
           and not any(cls in exclude_classes for cls in paragraph.get("class", []))
           and not (len(paragraph.get_text().strip()) == 0 or "[Copyright" in paragraph.get_text())
    ]

    # Replace double quotes and smart quotes with a dash
    filtered_paragraphs = [paragraph.replace('"', '-').replace('“', '-').replace('”', '') for paragraph in filtered_paragraphs]

    # Join the paragraphs into a single string, separated by a blank line
    output_content = "\n\n".join(filtered_paragraphs)

    # Write the output to a file
    with open(output_file, "w", encoding="utf-8") as output_file:
        output_file.write(output_content)

    print(f"Paragraphs extracted and saved to '{output_file}'.")


def process_directory(input_dir):
    # Iterate over all .txt files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.html'):
            html_file = os.path.join(input_dir, filename)
            print('Extracting : ', html_file)
            txt_file = os.path.join(input_dir, filename.replace('.html', '.txt'))
            extract_paragraphs(html_file, txt_file)

            print('Tokenizing : ', txt_file)
            tok_file = os.path.join(input_dir, filename.replace('.html', '.tok'))
            tokenizer2(txt_file, tok_file, 2)

            print('Spliting : ', tok_file)
            train_test_split(tok_file, 0.3)



process_directory("../data/authors/")
