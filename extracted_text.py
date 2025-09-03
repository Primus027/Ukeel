import pdfplumber
with pdfplumber.open("Public-Procurement-Rules-2008-English.pdf") as pdf:
    all_text = ""
    for page in pdf.pages: 
        # a list-like object where each element represents a page in the document
        page_text = page.extract_text() #Extracting contents from a page
        if page_text:
            all_text += page_text + '\n'
with open("amendment.txt", "w", encoding="utf-8") as f:
    f.write(all_text)
    # "w" = write mode
    