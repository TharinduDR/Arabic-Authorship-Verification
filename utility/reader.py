def read_book(path):
    lines = []
    with open(path, encoding='utf-8-sig') as file_in:
        for line in file_in:
            line = line.strip()
            line = line.replace("?", ".")
            line = line.replace("!", ".")
            line = line.replace("\t", "")
            line = line.replace("END_OF_PAGE", "")
            line = line.replace("END _ OF _ PAGE", "")
            sentences = line.split(".")
            for sentence in sentences:
                if len(sentence) != 0:
                    lines.append(sentence)
    return lines
