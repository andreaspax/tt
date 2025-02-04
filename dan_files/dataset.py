# Read the text8 file
def import_text8_dataset():
    with open('./text8', 'r', encoding='utf-8') as file:
        text8_data = file.read()
        return text8_data
    

