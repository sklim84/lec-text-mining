from tika import parser
data = parser.from_file("a.pdf")
content = data["content"].strip()
print(content)
