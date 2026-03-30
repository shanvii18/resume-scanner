import pymupdf
doc = pymupdf.open(r"C:\Users\verma\Downloads\ankur kr resume.pdf") 
out = open("output.txt", "wb") # create a text output
for page in doc: # iterate the document pages
    text = page.get_text().encode("utf8") # get plain text (is in UTF-8)
    out.write(text) # write text of page
    out.write(bytes((12,))) # write page delimiter (form feed 0x0C)
out.close()
print(text)

#CLEANING

import re

def clean_text(text):

    # bytes → string convert
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")

    text = text.lower()                          # lowercase
    text = re.sub(r'\n+', ' ', text)             # newlines remove
    text = re.sub(r'\s+', ' ', text)             # extra spaces
    text = re.sub(r'[^a-z0-9+.# ]', '', text)    # special chars remove
    return text.strip()

cleaned_text = clean_text(text)
print(cleaned_text)

